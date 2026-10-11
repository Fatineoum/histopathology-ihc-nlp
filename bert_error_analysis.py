"""
bert_error_analysis.py
IHC NLP Project : Marker-level Error Analysis for ClinicalBERT (8 epochs)
-------------------------------------------------------------------------
Reads the saved held-out predictions from bert_full_predictions_e8.csv
and produces the same marker-level analysis as error_analysis.py (rule-based),
allowing direct comparison of error patterns between the two pipelines.

Usage:
    python bert_error_analysis.py
    python bert_error_analysis.py --top-n 15 --min-support 3
"""

import argparse
import importlib.util
from pathlib import Path
from collections import defaultdict

import pandas as pd

DATA_DIR  = Path("data")
BERT_PREDS = DATA_DIR / "bert_full_predictions_e8.csv"


def load_rule_module():
    spec = importlib.util.spec_from_file_location(
        "rule_pipeline", Path("03_rule_based_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canonical(name, mod):
    try:
        return mod.canonical(name)
    except AttributeError:
        return name.lower().strip()


def parse_marker_set(s, mod):
    """Parse a comma-separated marker string into a normalised set."""
    if not s or str(s).lower() in {"", "nan", "nil", "na", "n/a"}:
        return set()
    return {canonical(t.strip(), mod) for t in str(s).split(",") if t.strip()}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def analyse(df, mod):
    per_marker = defaultdict(lambda: defaultdict(int))
    false_pos  = defaultdict(int)
    false_neg  = defaultdict(int)

    for _, row in df.iterrows():
        pred_pos = parse_marker_set(row["pred_positive"], mod)
        pred_neg = parse_marker_set(row["pred_negative"], mod)
        gt_pos   = parse_marker_set(row["gt_positive"],   mod)
        gt_neg   = parse_marker_set(row["gt_negative"],   mod)

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


def print_top(counter, label, n):
    print(f"\n  {label} (top {n}):")
    print(f"  {'Marker':<35s}  {'Count':>6}")
    print(f"  {'-' * 45}")
    for name, count in sorted(counter.items(), key=lambda x: -x[1])[:n]:
        print(f"  {name:<35s}  {count:>6}")


def print_per_marker(per_marker, min_support):
    print(f"\n{'=' * 75}")
    print(f"  Per-marker P / R / F1  (min {min_support} GT occurrences, sorted F1 ascending)")
    print(f"{'=' * 75}")
    print(f"  {'Marker':<30s}  {'Pol':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-' * 68}")

    rows = []
    for marker, c in per_marker.items():
        for pol, prefix in [("pos", "(+)"), ("neg", "(-)")]:
            tp = c.get(f"{pol}_tp", 0)
            fp = c.get(f"{pol}_fp", 0)
            fn = c.get(f"{pol}_fn", 0)
            if (tp + fn) < min_support:
                continue
            p, r, f = prf(tp, fp, fn)
            rows.append((marker, prefix, p, r, f, tp, fp, fn))

    rows.sort(key=lambda x: x[4])
    for marker, prefix, p, r, f, tp, fp, fn in rows:
        lbl = marker[:28] + ".." if len(marker) > 30 else marker
        print(f"  {lbl:<30s}  {prefix:>4}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  "
              f"{tp:>4}  {fp:>4}  {fn:>4}")
    print(f"{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Marker-level error analysis for ClinicalBERT (8 epochs)")
    parser.add_argument("--top-n",       type=int, default=20)
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading ClinicalBERT predictions from {BERT_PREDS} ...", flush=True)
    df = pd.read_csv(BERT_PREDS)
    print(f"  {len(df)} held-out cases.\n", flush=True)

    mod = load_rule_module()
    per_marker, false_pos, false_neg = analyse(df, mod)

    print(f"\n{'=' * 60}")
    print("  ClinicalBERT (8 epochs): False Positives (hallucinated)")
    print(f"{'=' * 60}")
    print_top(false_pos, "Most hallucinated markers", args.top_n)

    print(f"\n{'=' * 60}")
    print("  ClinicalBERT (8 epochs): False Negatives (missed)")
    print(f"{'=' * 60}")
    print_top(false_neg, "Most missed markers", args.top_n)

    print_per_marker(per_marker, args.min_support)


if __name__ == "__main__":
    main()
