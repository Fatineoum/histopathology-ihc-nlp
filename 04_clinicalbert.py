"""
04_clinicalbert.py
IHC NLP Project : ClinicalBERT Fine-tuning
------------------------------------------
Fine-tunes Bio_ClinicalBERT (Alsentzer et al. 2019) as a token classifier
for IHC marker extraction. Each token is labelled:

    O        : not part of a marker
    B-POS    : first token of a positive marker
    I-POS    : continuation token of a positive marker
    B-NEG    : first token of a negative marker
    I-NEG    : continuation token of a negative marker

Labelling strategy (distant supervision):
    For each case, we have the report text and the ground truth marker lists.
    We search for each GT marker name in the text using flexible regex
    (allows hyphens/slashes as separators) and label those character spans.
    Positive markers override negative on overlap (handles the rare case
    where a marker appears in both GT lists due to RNOH/referred conflict).

Dependencies:
    pip install transformers torch scikit-learn pandas openpyxl --break-system-packages
    # For GPU (optional, replace cu118 with your CUDA version):
    # pip install torch --index-url https://download.pytorch.org/whl/cu118

Usage:
    python 04_clinicalbert.py --train          # download model and fine-tune
    python 04_clinicalbert.py --eval           # case-level evaluation
    python 04_clinicalbert.py --train --eval   # train then evaluate immediately

Outputs:
    models/clinicalbert_ihc/    : fine-tuned checkpoint (best validation F1)
    data/bert_predictions.csv   : per-case predictions (all cases)
    data/bert_errors.csv        : cases with at least one FP or FN
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sklearn_f1

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
except ImportError:
    raise ImportError(
        "Missing dependencies. Run:\n"
        "  pip install transformers torch --break-system-packages"
    )


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME     = "emilyalsentzer/Bio_ClinicalBERT"
DATA_DIR       = Path("data")
MODEL_DIR      = Path("models/clinicalbert_ihc")        # pilot model
MODEL_DIR_FULL = Path("models/clinicalbert_ihc_full")   # full-dataset model

MAX_LENGTH  = 512    # ClinicalBERT max context window (tokens)
BATCH_SIZE  = 16
LR          = 2e-5
EPOCHS      = 8
WARMUP_FRAC = 0.1
VAL_SPLIT   = 0.15   # validation split (used during training)
TEST_SPLIT  = 0.15   # held-out test split (evaluated once, after training)
SEED        = 42

LABELS   = ["O", "B-POS", "I-POS", "B-NEG", "I-NEG"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Qualifiers that may appear in GT marker cells but are not part of the marker name
STRIP_QUALIFIERS = {
    "focal", "focally", "weak", "weakly", "strong", "strongly",
    "diffuse", "diffusely", "patchy", "occasional", "rare",
    "scattered", "moderate", "moderately", "intense", "dim",
    "cytoplasmic", "nuclear", "membranous", "paranuclear",
    "perinuclear", "dot-like",
}


# ── Marker utilities ──────────────────────────────────────────────────────────

def clean_gt_marker(marker: str) -> str:
    """Strip qualifiers from a GT marker name, e.g. 'focal CD34' -> 'CD34'."""
    parts = [p for p in marker.strip().split() if p.lower() not in STRIP_QUALIFIERS]
    return " ".join(parts).strip()


def make_marker_pattern(marker: str) -> str:
    """
    Build a case-insensitive regex pattern for a marker name.

    Splits on whitespace/hyphen/slash, escapes each part, then rejoins with
    a flexible separator. Wraps with lookaround word boundaries.

    Examples:
        'SMA'         -> r'(?<![a-zA-Z0-9])SMA(?![a-zA-Z0-9])'
        'AE1/3'       -> r'(?<![a-zA-Z0-9])AE1[\\s\\-/]?3(?![a-zA-Z0-9])'
        'h-caldesmon' -> r'(?<![a-zA-Z0-9])h[\\s\\-/]?caldesmon(?![a-zA-Z0-9])'
    """
    cleaned = clean_gt_marker(marker)
    parts   = [re.escape(p) for p in re.split(r"[\s\-/]+", cleaned) if p]
    if not parts:
        return re.escape(marker)
    inner   = r"[\s\-/]?".join(parts)
    return r"(?<![a-zA-Z0-9])" + inner + r"(?![a-zA-Z0-9])"


def find_spans(text: str, marker: str) -> list:
    """Return list of (start, end) character spans where marker occurs in text."""
    if not marker.strip():
        return []
    pattern = make_marker_pattern(marker)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def build_char_labels(text: str, positives: list, negatives: list) -> list:
    """
    Create a character-level label list for report text.

    Each position gets one of: 'O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG'.
    Positives are applied after negatives so they override on overlap
    (handles RNOH/referred conflicts where the same marker appears twice
    with different polarities).
    """
    char_labels = ["O"] * len(text)

    def apply(marker_list, b_label, i_label):
        for marker in marker_list:
            for start, end in find_spans(text, marker):
                if start < len(char_labels):
                    char_labels[start] = b_label
                    for i in range(start + 1, min(end, len(char_labels))):
                        char_labels[i] = i_label

    apply(negatives, "B-NEG", "I-NEG")
    apply(positives, "B-POS", "I-POS")  # positives override on overlap

    return char_labels


# ── Dataset ───────────────────────────────────────────────────────────────────

class IHCDataset(Dataset):
    """
    PyTorch Dataset for IHC token classification.

    Tokenises each report with the ClinicalBERT tokenizer and aligns the
    character-level labels to token positions using offset_mapping.
    Non-first subword tokens and special tokens ([CLS], [SEP], [PAD]) receive
    label -100, which is ignored by PyTorch's cross-entropy loss.
    """

    def __init__(self, records: list, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.items      = []

        for rec in records:
            text      = str(rec["text"])
            positives = rec.get("positives", [])
            negatives = rec.get("negatives", [])

            char_labels = build_char_labels(text, positives, negatives)

            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offsets   = encoding["offset_mapping"][0].tolist()
            token_ids = encoding["input_ids"][0]
            attn_mask = encoding["attention_mask"][0]

            token_labels = []
            for start, end in offsets:
                if start == 0 and end == 0:
                    # Special token
                    token_labels.append(-100)
                elif start >= len(char_labels):
                    # Truncated beyond text length
                    token_labels.append(-100)
                else:
                    # Use the char label at the first character of this token.
                    # This handles subwords: only the first piece gets a real label,
                    # subsequent pieces get the same char position (which is I- or O),
                    # so they are treated as continuation tokens naturally.
                    token_labels.append(label2id[char_labels[start]])

            self.items.append({
                "input_ids":      token_ids,
                "attention_mask": attn_mask,
                "labels":         torch.tensor(token_labels, dtype=torch.long),
                "text":           text,
                "positives":      positives,
                "negatives":      negatives,
                "patient_id":     rec.get("patient_id", ""),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pilot() -> pd.DataFrame:
    path = DATA_DIR / "pilot_leiomyosarcoma.csv"
    df   = pd.read_csv(path, dtype=str).fillna("")
    df   = df[df["patient_id"] != "S00157601"].reset_index(drop=True)
    print(f"Pilot cases loaded: {len(df)}")
    return df


def load_full_dataset() -> pd.DataFrame:
    """
    Load the full cleaned ground truth for cross-diagnosis training.
    Excludes: anonymous-marker cases (x1/x2/x3), cases with no IHC text,
    and S00157601 (known data entry error).
    """
    path = DATA_DIR / "cleaned_ground_truth.csv"
    df   = pd.read_csv(path, dtype=str).fillna("")
    before = len(df)
    # Exclude flagged anonymous cases
    df = df[df["has_anonymous"].str.strip().str.lower() != "true"]
    # Exclude cases with no IHC text
    df = df[df["ihc_text"].str.strip() != ""]
    # Exclude known bad case
    df = df[df["patient_id"] != "S00157601"].reset_index(drop=True)
    print(f"Full dataset loaded: {len(df)} cases ({before - len(df)} excluded)")
    print(f"Diagnoses covered:   {df['diagnosis'].nunique()}")
    return df


def parse_markers(cell: str) -> list:
    if not cell.strip():
        return []
    return [m.strip() for m in cell.split(",") if m.strip()]


def df_to_records(df: pd.DataFrame) -> list:
    return [
        {
            "patient_id": row["patient_id"],
            "text":       row["ihc_text"],
            "positives":  parse_markers(row.get("positive_ihc", "")),
            "negatives":  parse_markers(row.get("negative_ihc", "")),
        }
        for _, row in df.iterrows()
    ]


# ── Metrics ───────────────────────────────────────────────────────────────────

def prf(tp, fp, fn):
    """Compute precision, recall, F1. Returns (0, 0, 0) on empty denominator."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def token_level_f1(model, loader) -> float:
    """
    Token-level micro F1 over B-POS/I-POS/B-NEG/I-NEG labels only (excludes O).
    Used as a fast proxy metric during training validation.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits   = model(input_ids=ids, attention_mask=mask).logits
            pred_ids = logits.argmax(dim=-1)

            valid = labels != -100
            all_preds.extend(pred_ids[valid].cpu().tolist())
            all_labels.extend(labels[valid].cpu().tolist())

    if not all_labels:
        return 0.0

    return sklearn_f1(
        all_labels, all_preds,
        average="micro",
        labels=[1, 2, 3, 4],  # B-POS, I-POS, B-NEG, I-NEG
        zero_division=0,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(df: pd.DataFrame, tokenizer, model_dir: Path = MODEL_DIR,
                 test_df: pd.DataFrame = None):
    """Fine-tune Bio_ClinicalBERT. Returns the trained model."""
    records              = df_to_records(df)
    train_recs, val_recs = train_test_split(
        records, test_size=VAL_SPLIT, random_state=SEED
    )
    print(f"Train: {len(train_recs)} cases | Val: {len(val_recs)} cases")

    # Count labelled tokens for a sanity check
    sample_labels = []
    for rec in train_recs[:20]:
        cl = build_char_labels(rec["text"], rec["positives"], rec["negatives"])
        sample_labels.extend(cl)
    n_marked = sum(1 for l in sample_labels if l != "O")
    print(f"Labelling check (first 20 cases): {n_marked}/{len(sample_labels)} "
          f"chars labelled as markers ({100*n_marked/max(1,len(sample_labels)):.1f}%)")

    train_ds = IHCDataset(train_recs, tokenizer)
    val_ds   = IHCDataset(val_recs,   tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_batch
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch
    )

    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    optimiser    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_f1   = token_level_f1(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_token_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  Checkpoint saved (epoch {epoch}, F1={val_f1:.4f})")

    print(f"\nBest validation token F1: {best_val_f1:.4f}")
    print(f"Model saved to: {model_dir}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_text(model, tokenizer, text: str) -> tuple:
    """
    Extract (pred_positive_set, pred_negative_set) from a single report text.

    Reconstructs marker text by reading character spans back from the original
    text using the tokenizer's offset mapping.
    """
    model.eval()
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets  = encoding["offset_mapping"][0].tolist()
    ids      = encoding["input_ids"].to(DEVICE)
    mask     = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits

    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()

    pred_pos    = set()
    pred_neg    = set()
    span_chars  = []
    span_label  = None

    def flush():
        if not span_chars or span_label is None:
            return
        start  = span_chars[0][0]
        end    = span_chars[-1][1]
        marker = re.sub(r"\s+", " ", text[start:end]).strip()
        if marker and len(marker) >= 2:
            marker_lower = marker.lower()
            if span_label == "POS":
                pred_pos.add(marker_lower)
            elif span_label == "NEG":
                pred_neg.add(marker_lower)

    for (start, end), pred_id in zip(offsets, pred_ids):
        if start == 0 and end == 0:
            # Special token: close any open span
            flush()
            span_chars = []
            span_label = None
            continue

        lbl = id2label[pred_id]

        if lbl.startswith("B-"):
            flush()
            span_label = lbl[2:]   # 'POS' or 'NEG'
            span_chars = [(start, end)]
        elif lbl.startswith("I-") and span_label == lbl[2:]:
            span_chars.append((start, end))
        else:
            flush()
            span_chars = []
            span_label = None

    flush()
    return pred_pos, pred_neg


# ── Case-level evaluation ─────────────────────────────────────────────────────

def evaluate_cases(df: pd.DataFrame, model, tokenizer) -> pd.DataFrame:
    """
    Run case-level evaluation using the same set-comparison logic as the
    rule-based pipeline (03_rule_based_pipeline.py). Returns a full results
    DataFrame and prints the summary table.
    """
    model.eval()
    rows   = []
    totals = {k: 0 for k in ["tp_pos", "fp_pos", "fn_pos", "tp_neg", "fp_neg", "fn_neg"]}

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 50 == 0:
            print(f"  Evaluating case {i}/{len(df)}...")

        text   = str(row["ihc_text"])
        gt_pos = {m.lower() for m in parse_markers(row.get("positive_ihc", ""))}
        gt_neg = {m.lower() for m in parse_markers(row.get("negative_ihc", ""))}

        pred_pos, pred_neg = predict_text(model, tokenizer, text)

        case = {
            "patient_id":    row["patient_id"],
            "ihc_text":      text,
            "gt_positive":   ", ".join(sorted(gt_pos)),
            "gt_negative":   ", ".join(sorted(gt_neg)),
            "pred_positive": ", ".join(sorted(pred_pos)),
            "pred_negative": ", ".join(sorted(pred_neg)),
            "tp_pos": len(pred_pos & gt_pos),
            "fp_pos": len(pred_pos - gt_pos),
            "fn_pos": len(gt_pos  - pred_pos),
            "tp_neg": len(pred_neg & gt_neg),
            "fp_neg": len(pred_neg - gt_neg),
            "fn_neg": len(gt_neg  - pred_neg),
        }
        for k in totals:
            totals[k] += case[k]
        rows.append(case)

    results = pd.DataFrame(rows)

    tp_p = totals["tp_pos"]; fp_p = totals["fp_pos"]; fn_p = totals["fn_pos"]
    tp_n = totals["tp_neg"]; fp_n = totals["fp_neg"]; fn_n = totals["fn_neg"]
    tp_a = tp_p + tp_n;      fp_a = fp_p + fp_n;      fn_a = fn_p + fn_n

    p_pos, r_pos, f1_pos = prf(tp_p, fp_p, fn_p)
    p_neg, r_neg, f1_neg = prf(tp_n, fp_n, fn_n)
    _,     _,     f1_all = prf(tp_a, fp_a, fn_a)

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS : ClinicalBERT : leiomyosarcoma HG pilot")
    print("=" * 55)
    print(f"{'':22s}  {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'Positive markers':22s}  {p_pos:>10.4f} {r_pos:>10.4f} {f1_pos:>10.4f}")
    print(f"{'Negative markers':22s}  {p_neg:>10.4f} {r_neg:>10.4f} {f1_neg:>10.4f}")
    print(f"{'Overall (micro)':22s}  {'':>10} {'':>10} {f1_all:>10.4f}")
    print(f"\nTP pos={tp_p}  FP pos={fp_p}  FN pos={fn_p}")
    print(f"TP neg={tp_n}  FP neg={fp_n}  FN neg={fn_n}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ClinicalBERT fine-tuning for IHC marker extraction"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Fine-tune Bio_ClinicalBERT"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Case-level evaluation"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full ground truth dataset (all diagnoses) instead of pilot"
    )
    args = parser.parse_args()

    if not (args.train or args.eval):
        parser.print_help()
        return

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = None
    test_df   = None

    if args.full:
        full_df = load_full_dataset()
        # Hold out a test set before any training
        train_val_df, test_df = train_test_split(
            full_df, test_size=TEST_SPLIT, random_state=SEED
        )
        print(f"Train+val: {len(train_val_df)} | Test (held out): {len(test_df)}")
        df        = train_val_df
        model_dir = MODEL_DIR_FULL
        pred_file = "bert_full_predictions.csv"
        err_file  = "bert_full_errors.csv"
    else:
        df        = load_pilot()
        model_dir = MODEL_DIR
        pred_file = "bert_predictions.csv"
        err_file  = "bert_errors.csv"

    if args.train:
        model = run_training(df, tokenizer, model_dir=model_dir)

    if args.eval:
        if model is None:
            if not model_dir.exists():
                print(f"\nNo checkpoint found at {model_dir}.")
                flag = "--train --eval --full" if args.full else "--train --eval"
                print(f"Run:  python 04_clinicalbert.py {flag}")
                return
            print(f"\nLoading checkpoint from {model_dir}...")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model     = AutoModelForTokenClassification.from_pretrained(
                model_dir
            ).to(DEVICE)

        # Evaluate on held-out test set if --full, otherwise full pilot
        eval_df = test_df if (args.full and test_df is not None) else df
        label   = "test set" if (args.full and test_df is not None) else "pilot"
        print(f"\nEvaluating on {label} ({len(eval_df)} cases)...")
        results = evaluate_cases(eval_df, model, tokenizer)

        errors = results[
            (results["fp_pos"] + results["fn_pos"] +
             results["fp_neg"] + results["fn_neg"]) > 0
        ].copy()

        results.to_csv(DATA_DIR / pred_file, index=False)
        errors.to_csv(DATA_DIR  / err_file,  index=False)
        print(f"\nSaved: data/{pred_file}  ({len(results)} rows)")
        print(f"Saved: data/{err_file}   ({len(errors)} rows)")


if __name__ == "__main__":
    main()
