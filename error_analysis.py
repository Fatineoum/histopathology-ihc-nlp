"""
error_analysis.py
IHC NLP Project : Error Analysis from Evaluation Results
---------------------------------------------------------
Reads per-case prediction output (produced by 03_rule_based_pipeline.py or
the hybrid pipeline) alongside the ground truth and reports:

  - Most commonly missed markers (false negatives) by polarity
  - Most commonly hallucinated markers (false positives) by polarity
  - Per-marker precision / recall / F1

Usage:
    # Step 1: generate per-case predictions CSV
    python evaluate.py --output results/eval_cases.csv

    # Step 2: run this script against the predictions
    python error_analysis.py --input results/eval_cases.csv

    # Or generate predictions and analyse in one step (rule-based pipeline):
    python error_analysis.py --run-rule

Alternatively, pass --run-rule to generate predictions automatically.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import openpyxl
import pandas as pd

DATA_DIR = Path("data")
GT_FILE  = DATA_DIR / "reports_ground_truth.xlsx"

COL_PID      = 0
COL_DX       = 4
COL_IHC_TEXT = 17
COL_POS_GT   = 18
COL_NEG_GT   = 19


# ── Ground truth loader ───────────────────────────────────────────────────────

def load_ground_truth():
    wb = openpyxl.load_workbook(GT_FILE, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    data = []
    for row in rows[2:]:
        pid      = row[COL_PID]
        ihc_text = row[COL_IHC_TEXT]
        pos_gt   = row[COL_POS_GT]
        neg_gt   = row[COL_NEG_GT]
        dx       = row[COL_DX]
        if not pid or (not ihc_text and not pos_gt and not neg_gt):
            continue
        data.append({
            "patient_id": str(pid).strip(),
            "diagnosis":  str(dx).strip().lower() if dx else "unknown",
            "ihc_text":   str(ihc_text).strip() if ihc_text else "",
            "pos_gt":     str(pos_gt).strip()   if pos_gt   else "",
            "neg_gt":     str(neg_gt).strip()   if neg_gt   else "",
        })
    wb.close()
    df = pd.DataFrame(data)
    df = df[(df["pos_gt"].str.lower().isin({"", "nil", "na", "n/a"}) == False)
            | (df["neg_gt"].str.lower().isin({"", "nil", "na", "n/a"}) == False)]
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rule_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rule_pipeline", Path("03_rule_based_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canonical(name, mod):
    """Normalise a marker name using the pipeline's own canonical function."""
    try:
        return mod.canonical(name)
    except AttributeError:
        return name.lower().strip()


def parse_list(s, mod):
    try:
        return set(mod.parse_gt_list(s))
    except Exception:
        if not s or str(s).lower() in {"", "nil", "na", "n/a"}:
            return set()
        return {canonical(t.strip(), mod) for t in str(s).split(",") if t.strip()}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_predictions(df_gt, mod, predict_fn):
    """
    Run predict_fn(text) -> list[{"marker": str, "outcome": "positive"|"negative"}]
    against every case in df_gt and accumulate per-marker error counts.
    """
    per_marker = defaultdict(lambda: defaultdict(int))
    # Counters: per_marker[marker][polarity][tp/fp/fn]

    false_pos = defaultdict(int)   # marker -> count of FP occurrences
    false_neg = defaultdict(int)   # marker -> count of FN occurrences

    for _, row in df_gt.iterrows():
        preds = predict_fn(row["ihc_text"])
        # Canonicalise predicted markers to match ground-truth normalisation
        pred_pos = {canonical(p["marker"], mod) for p in preds if p["outcome"] == "positive"}
        pred_neg = {canonical(p["marker"], mod) for p in preds if p["outcome"] == "negative"}

        gt_pos = parse_list(row["pos_gt"], mod)
        gt_neg = parse_list(row["neg_gt"], mod)

        # Positive polarity
        for m in gt_pos:
            if m in pred_pos:
                per_marker[m]["pos_tp"] += 1
            else:
                per_marker[m]["pos_fn"] += 1
                false_neg[f"(+) {m}"] += 1
        for m in pred_pos:
            if m not in gt_pos:
                per_marker[m]["pos_fp"] += 1
                false_pos[f"(+) {m}"] += 1

        # Negative polarity
        for m in gt_neg:
            if m in pred_neg:
                per_marker[m]["neg_tp"] += 1
            else:
                per_marker[m]["neg_fn"] += 1
                false_neg[f"(-) {m}"] += 1
        for m in pred_neg:
            if m not in gt_neg:
                per_marker[m]["neg_fp"] += 1
                false_pos[f"(-) {m}"] += 1

    return per_marker, false_pos, false_neg


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_top_errors(counter, label, n=20):
    print(f"\n  {label} (top {n}):")
    print(f"  {'Marker':<35s}  {'Count':>6}")
    print(f"  {'-' * 45}")
    for name, count in sorted(counter.items(), key=lambda x: -x[1])[:n]:
        print(f"  {name:<35s}  {count:>6}")


def print_per_marker(per_marker, min_support=5):
    print(f"\n{'=' * 75}")
    print(f"  Per-marker precision / recall / F1  (min {min_support} GT occurrences)")
    print(f"{'=' * 75}")
    print(f"  {'Marker':<30s}  {'Pol':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-' * 68}")

    rows = []
    for marker, c in per_marker.items():
        for pol, prefix in [("pos", "(+)"), ("neg", "(-)")]:
            tp = c.get(f"{pol}_tp", 0)
            fp = c.get(f"{pol}_fp", 0)
            fn = c.get(f"{pol}_fn", 0)
            gt_total = tp + fn
            if gt_total < min_support:
                continue
            p, r, f = prf(tp, fp, fn)
            rows.append((marker, prefix, p, r, f, tp, fp, fn))

    rows.sort(key=lambda x: x[4])   # sort by F1 ascending (worst first)
    for marker, prefix, p, r, f, tp, fp, fn in rows:
        label = marker[:28] + ".." if len(marker) > 30 else marker
        print(f"  {label:<30s}  {prefix:>4}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  "
              f"{tp:>4}  {fp:>4}  {fn:>4}")
    print(f"{'=' * 75}\n")


def run_error_analysis(df_gt, mod, predict_fn, top_n=20, min_support=5):
    print("\nRunning error analysis ...", flush=True)
    per_marker, false_pos, false_neg = analyse_predictions(df_gt, mod, predict_fn)

    print(f"\n{'=' * 60}")
    print("  False Positives (markers the pipeline hallucinated)")
    print(f"{'=' * 60}")
    print_top_errors(false_pos, "Most hallucinated markers", n=top_n)

    print(f"\n{'=' * 60}")
    print("  False Negatives (markers the pipeline missed)")
    print(f"{'=' * 60}")
    print_top_errors(false_neg, "Most missed markers", n=top_n)

    print_per_marker(per_marker, min_support=min_support)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Error analysis for IHC extraction pipeline")
    parser.add_argument("--run-rule",    action="store_true",
                        help="Generate rule-based predictions in-memory (no CSV needed)")
    parser.add_argument("--top-n",       type=int, default=20,
                        help="Number of top false positives / negatives to show (default 20)")
    parser.add_argument("--min-support", type=int, default=5,
                        help="Min GT occurrences for per-marker table (default 5)")
    args = parser.parse_args()

    print(f"Loading ground truth from {GT_FILE} ...", flush=True)
    df_gt = load_ground_truth()
    print(f"  {len(df_gt)} cases loaded.\n", flush=True)

    mod = load_rule_module()

    if args.run_rule:
        def predict_fn(text):
            return mod.extract_from_text(text)
        run_error_analysis(df_gt, mod, predict_fn,
                           top_n=args.top_n, min_support=args.min_support)
    else:
        parser.print_help()
        print("\nNo prediction mode specified. Use --run-rule to analyse the rule-based pipeline.")
        sys.exit(0)


if __name__ == "__main__":
    main()
