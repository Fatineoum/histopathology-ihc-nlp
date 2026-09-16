"""
05_hybrid_pipeline.py
IHC NLP Project : Hybrid Extraction Pipeline
---------------------------------------------
Combines the rule-based pipeline (03_rule_based_pipeline.py) and ClinicalBERT
(04_clinicalbert.py) into a single extraction system.

Combination strategy (BERT-primary):
  1. Run both pipelines independently on each report
  2. Normalise all outputs via marker_utils.py
  3. Start with ClinicalBERT predictions as primary
  4. Rule-based supplements: adds markers BERT missed, unless BERT
     explicitly predicted the opposite polarity
  5. Resolve any remaining conflicts: ClinicalBERT wins

Additional outputs:
  - Equivocal markers: positive predictions found in equivocal language context
  - Ki-67 index: extracted percentage, not included in binary classification
  - Source: referred vs in-house flag (RNOH preference applied at database layer)

Usage:
    python 05_hybrid_pipeline.py --pilot       # run on leiomyosarcoma pilot
    python 05_hybrid_pipeline.py --full        # run on all ground truth cases
    python 05_hybrid_pipeline.py --compare     # show all three pipelines side by side
    python 05_hybrid_pipeline.py --no-immuno   # run on cleaned_no_immuno.csv (no GT eval)

Outputs:
    data/hybrid_predictions.csv     : per-case predictions + ki67 + equivocal
    data/hybrid_errors.csv          : cases with any errors (normalised comparison)
    data/pipeline_comparison.csv    : rule / bert / hybrid F1 side by side
    data/no_immuno_predictions.csv  : pipeline-only predictions for no-GT cases
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
import torch

from marker_utils import (
    normalise_marker,
    normalise_marker_list,
    parse_and_normalise,
    extract_ki67,
    strip_ki67,
    EXCLUDED_FROM_BINARY,
)

DATA_DIR  = Path("data")
MODEL_DIR = Path("models/clinicalbert_ihc_full")   # full-dataset model
MODEL_DIR_PILOT = Path("models/clinicalbert_ihc")  # fallback: pilot model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dynamic imports from numbered scripts ─────────────────────────────────────

def _import_module(filename: str, name: str):
    spec   = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rule_pipeline():
    try:
        m = _import_module("03_rule_based_pipeline.py", "rule_pipeline")
        return m.extract_from_text
    except Exception as e:
        print(f"Could not load rule-based pipeline: {e}")
        sys.exit(1)


def load_bert_pipeline():
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    # Prefer full-dataset model, fall back to pilot
    model_dir = MODEL_DIR if MODEL_DIR.exists() else MODEL_DIR_PILOT
    if not model_dir.exists():
        print(f"No ClinicalBERT checkpoint found. Run 04_clinicalbert.py --train first.")
        sys.exit(1)

    print(f"Loading ClinicalBERT from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    # Import predict_text from 04_clinicalbert
    bert_module = _import_module("04_clinicalbert.py", "bert_pipeline")
    predict_fn  = bert_module.predict_text

    return model, tokenizer, predict_fn


# ── Equivocal detection ───────────────────────────────────────────────────────

_EQUIVOCAL_RE = re.compile(
    r"""
    (?<!un)equivocal | borderline | indeterminate | cannot\s+exclude |
    possible | uncertain | suspicious | weak(?:ly)?\s+positive |
    focally?\s+weak | \?\s*positive | at\s+least\s+focally
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_equivocal(text: str, pred_positive: set) -> set:
    """
    Return the subset of pred_positive markers found in an equivocal context.
    Checks a 40-character window BEFORE each marker mention and 10 chars after.
    The asymmetric window prevents qualifiers for one marker bleeding into the
    context of the next marker in a comma-separated list.
    'unequivocally' is excluded via negative lookbehind in the regex.
    """
    equivocal = set()
    for marker in pred_positive:
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        for m in pattern.finditer(text):
            context = text[max(0, m.start() - 40): m.end() + 10]
            if _EQUIVOCAL_RE.search(context):
                equivocal.add(marker)
                break
    return equivocal


# ── Combination logic ─────────────────────────────────────────────────────────

def combine(rule_output: list, bert_pos_raw: set, bert_neg_raw: set) -> tuple:
    """
    Merge rule-based and ClinicalBERT outputs into a single prediction.

    Parameters
    ----------
    rule_output  : list of {marker, outcome, source} dicts from rule-based pipeline
    bert_pos_raw : set of positive marker strings from ClinicalBERT (unnormalised)
    bert_neg_raw : set of negative marker strings from ClinicalBERT (unnormalised)

    Returns
    -------
    combined_pos, combined_neg : normalised marker sets
    """
    # Separate rule-based by polarity (exclude referred for binary classification)
    rule_pos_raw = {
        r["marker"] for r in rule_output
        if r["outcome"] in ("positive", "focal_positive")
    }
    rule_neg_raw = {
        r["marker"] for r in rule_output
        if r["outcome"] == "negative"
    }

    # Strip Ki-67 from both pipelines before normalising
    rp = normalise_marker_list(strip_ki67(list(rule_pos_raw)))
    rn = normalise_marker_list(strip_ki67(list(rule_neg_raw)))
    bp = normalise_marker_list(strip_ki67(list(bert_pos_raw)))
    bn = normalise_marker_list(strip_ki67(list(bert_neg_raw)))

    # BERT-primary combination:
    #   rules supplement BERT unless BERT explicitly predicted the other polarity
    combined_pos = bp | (rp - bn)
    combined_neg = bn | (rn - bp)

    # Safety: resolve any remaining conflicts (BERT/positive wins)
    conflicts     = combined_pos & combined_neg
    combined_neg -= conflicts

    return combined_pos, combined_neg


# ── Inference-only (no GT evaluation) ────────────────────────────────────────

def run_inference(df: pd.DataFrame, rule_extract, model, tokenizer, predict_fn) -> pd.DataFrame:
    """
    Run the hybrid pipeline on a dataframe that has no ground-truth marker columns.
    Returns a DataFrame with patient_id, diagnosis, ihc_text, pred_positive,
    pred_negative, equivocal, ki67_index.
    """
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Processing case {i}/{len(df)}...")

        text = str(row["ihc_text"])

        rule_output         = rule_extract(text)
        bert_pos_raw, bert_neg_raw = predict_fn(model, tokenizer, text)
        hyb_pos, hyb_neg    = combine(rule_output, bert_pos_raw, bert_neg_raw)
        equivocal           = detect_equivocal(text, hyb_pos)
        ki67_value          = extract_ki67(text)

        rows.append({
            "patient_id":    row["patient_id"],
            "diagnosis":     row.get("diagnosis", ""),
            "ihc_text":      text,
            "pred_positive": ", ".join(sorted(hyb_pos)),
            "pred_negative": ", ".join(sorted(hyb_neg)),
            "equivocal":     ", ".join(sorted(equivocal)),
            "ki67_index":    ki67_value if ki67_value else "",
        })

    return pd.DataFrame(rows)


# ── Metrics ───────────────────────────────────────────────────────────────────

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def print_results(label: str, tp_p, fp_p, fn_p, tp_n, fp_n, fn_n):
    p_pos, r_pos, f1_pos = prf(tp_p, fp_p, fn_p)
    p_neg, r_neg, f1_neg = prf(tp_n, fp_n, fn_n)
    _, _, f1_all          = prf(tp_p + tp_n, fp_p + fp_n, fn_p + fn_n)
    print(f"\n{'=' * 58}")
    print(f"EVALUATION : {label}")
    print(f"{'=' * 58}")
    print(f"{'':22s}  {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'Positive markers':22s}  {p_pos:>10.4f} {r_pos:>10.4f} {f1_pos:>10.4f}")
    print(f"{'Negative markers':22s}  {p_neg:>10.4f} {r_neg:>10.4f} {f1_neg:>10.4f}")
    print(f"{'Overall (micro)':22s}  {'':>10} {'':>10} {f1_all:>10.4f}")
    print(f"TP pos={tp_p}  FP pos={fp_p}  FN pos={fn_p}")
    print(f"TP neg={tp_n}  FP neg={fp_n}  FN neg={fn_n}")
    return f1_all


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(full: bool) -> pd.DataFrame:
    if full:
        df = pd.read_csv(DATA_DIR / "cleaned_ground_truth.csv", dtype=str).fillna("")
        df = df[df["has_anonymous"].str.strip().str.lower() != "true"]
        df = df[df["ihc_text"].str.strip() != ""]
        df = df[df["patient_id"] != "S00157601"].reset_index(drop=True)
        print(f"Full ground truth: {len(df)} cases, {df['diagnosis'].nunique()} diagnoses")
    else:
        df = pd.read_csv(DATA_DIR / "pilot_leiomyosarcoma.csv", dtype=str).fillna("")
        df = df[df["patient_id"] != "S00157601"].reset_index(drop=True)
        print(f"Pilot: {len(df)} cases")
    return df


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_hybrid(df: pd.DataFrame, rule_extract, model, tokenizer, predict_fn,
               compare: bool = False) -> pd.DataFrame:
    """
    Run hybrid pipeline on all cases. Returns per-case results DataFrame.
    If compare=True, also runs and reports rule-only and BERT-only metrics.
    """
    rows = []

    # Accumulators for each pipeline
    acc = {
        "hybrid": dict(tp_p=0, fp_p=0, fn_p=0, tp_n=0, fp_n=0, fn_n=0),
        "rules":  dict(tp_p=0, fp_p=0, fn_p=0, tp_n=0, fp_n=0, fn_n=0),
        "bert":   dict(tp_p=0, fp_p=0, fn_p=0, tp_n=0, fp_n=0, fn_n=0),
    }

    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Processing case {i}/{len(df)}...")

        text   = str(row["ihc_text"])
        gt_pos = parse_and_normalise(row.get("positive_ihc", ""))
        gt_neg = parse_and_normalise(row.get("negative_ihc", ""))

        # Run rule-based
        rule_output  = rule_extract(text)
        rule_pos_raw = {r["marker"] for r in rule_output
                        if r["outcome"] in ("positive", "focal_positive")}
        rule_neg_raw = {r["marker"] for r in rule_output
                        if r["outcome"] == "negative"}
        rule_pos     = normalise_marker_list(strip_ki67(list(rule_pos_raw)))
        rule_neg     = normalise_marker_list(strip_ki67(list(rule_neg_raw)))

        # Run ClinicalBERT
        bert_pos_raw, bert_neg_raw = predict_fn(model, tokenizer, text)
        bert_pos = normalise_marker_list(strip_ki67(list(bert_pos_raw)))
        bert_neg = normalise_marker_list(strip_ki67(list(bert_neg_raw)))

        # Combine
        hyb_pos, hyb_neg = combine(rule_output, bert_pos_raw, bert_neg_raw)

        # Equivocal detection on hybrid positive set
        equivocal = detect_equivocal(text, hyb_pos)

        # Ki-67 extraction
        ki67_value = extract_ki67(text)

        # Accumulate metrics
        for pipe, pp, pn in [
            ("hybrid", hyb_pos, hyb_neg),
            ("rules",  rule_pos, rule_neg),
            ("bert",   bert_pos, bert_neg),
        ]:
            a = acc[pipe]
            a["tp_p"] += len(pp & gt_pos); a["fp_p"] += len(pp - gt_pos)
            a["fn_p"] += len(gt_pos - pp); a["tp_n"] += len(pn & gt_neg)
            a["fp_n"] += len(pn - gt_neg); a["fn_n"] += len(gt_neg - pn)

        rows.append({
            "patient_id":      row["patient_id"],
            "diagnosis":       row.get("diagnosis", ""),
            "ihc_text":        text,
            "gt_positive":     ", ".join(sorted(gt_pos)),
            "gt_negative":     ", ".join(sorted(gt_neg)),
            "rule_positive":   ", ".join(sorted(rule_pos)),
            "rule_negative":   ", ".join(sorted(rule_neg)),
            "bert_positive":   ", ".join(sorted(bert_pos)),
            "bert_negative":   ", ".join(sorted(bert_neg)),
            "pred_positive":   ", ".join(sorted(hyb_pos)),
            "pred_negative":   ", ".join(sorted(hyb_neg)),
            "equivocal":       ", ".join(sorted(equivocal)),
            "ki67_index":      ki67_value if ki67_value else "",
            "fp_pos": len(hyb_pos - gt_pos), "fn_pos": len(gt_pos - hyb_pos),
            "fp_neg": len(hyb_neg - gt_neg), "fn_neg": len(gt_neg - hyb_neg),
        })

    results = pd.DataFrame(rows)

    if compare:
        f1_rules  = print_results("Rule-based (normalised)",  **acc["rules"])
        f1_bert   = print_results("ClinicalBERT (normalised)", **acc["bert"])
        f1_hybrid = print_results("Hybrid (normalised)",       **acc["hybrid"])

        print(f"\n{'=' * 58}")
        print("SUMMARY")
        print(f"{'=' * 58}")
        print(f"  Rule-based F1 (normalised):   {f1_rules:.4f}")
        print(f"  ClinicalBERT F1 (normalised): {f1_bert:.4f}")
        print(f"  Hybrid F1 (normalised):       {f1_hybrid:.4f}")
    else:
        print_results("Hybrid pipeline", **acc["hybrid"])

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid IHC extraction pipeline"
    )
    parser.add_argument("--pilot",     action="store_true",
                        help="Run on leiomyosarcoma HG pilot")
    parser.add_argument("--full",      action="store_true",
                        help="Run on full ground truth dataset")
    parser.add_argument("--compare",   action="store_true",
                        help="Report rule / BERT / hybrid metrics side by side")
    parser.add_argument("--no-immuno", action="store_true",
                        help="Run inference on cleaned_no_immuno.csv (no GT evaluation)")
    args = parser.parse_args()

    if not (args.pilot or args.full or args.no_immuno):
        parser.print_help()
        return

    print(f"Device: {DEVICE}")

    rule_extract              = load_rule_pipeline()
    model, tokenizer, predict = load_bert_pipeline()

    if args.no_immuno:
        # Run pipeline on no-GT cases only; no metrics computed
        ni_path = DATA_DIR / "cleaned_no_immuno.csv"
        if not ni_path.exists():
            print(f"ERROR: {ni_path} not found.")
            return
        df_ni = pd.read_csv(ni_path, dtype=str).fillna("")
        df_ni = df_ni[df_ni["ihc_text"].str.strip() != ""].reset_index(drop=True)
        print(f"No-immuno dataset: {len(df_ni)} cases, "
              f"{df_ni['diagnosis'].nunique()} diagnoses")
        results_ni = run_inference(df_ni, rule_extract, model, tokenizer, predict)
        results_ni.to_csv(DATA_DIR / "no_immuno_predictions.csv", index=False)
        ki67_ni = (results_ni["ki67_index"] != "").sum()
        equiv_ni = (results_ni["equivocal"] != "").sum()
        print(f"\nSaved: data/no_immuno_predictions.csv  ({len(results_ni)} rows)")
        print(f"Ki-67 extracted:   {ki67_ni} cases")
        print(f"Equivocal flags:   {equiv_ni} cases")
        return

    df = load_data(full=args.full)

    results = run_hybrid(df, rule_extract, model, tokenizer, predict,
                         compare=args.compare)

    errors = results[
        (results["fp_pos"] + results["fn_pos"] +
         results["fp_neg"] + results["fn_neg"]) > 0
    ].copy()

    results.to_csv(DATA_DIR / "hybrid_predictions.csv", index=False)
    errors.to_csv(DATA_DIR  / "hybrid_errors.csv",      index=False)

    ki67_found = results[results["ki67_index"] != ""].shape[0]
    equiv_found = results[results["equivocal"] != ""].shape[0]

    print(f"\nSaved: data/hybrid_predictions.csv  ({len(results)} rows)")
    print(f"Saved: data/hybrid_errors.csv        ({len(errors)} rows)")
    print(f"Ki-67 index extracted:   {ki67_found} cases")
    print(f"Equivocal flags raised:  {equiv_found} cases")


if __name__ == "__main__":
    main()
