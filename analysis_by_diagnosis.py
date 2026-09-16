"""
analysis_by_diagnosis.py
IHC NLP Project : Per-Diagnosis Performance Breakdown
------------------------------------------------------
Computes precision, recall, F1 for each diagnosis using the hybrid pipeline
predictions. Only includes diagnoses with at least MIN_CASES cases.

Usage:
    python analysis_by_diagnosis.py
    python analysis_by_diagnosis.py --min-cases 20

Outputs:
    data/diagnosis_performance.csv   : full per-diagnosis metrics
"""

import argparse
from pathlib import Path
import pandas as pd
from marker_utils import parse_and_normalise

DATA_DIR  = Path("data")
MIN_CASES = 10


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cases", type=int, default=MIN_CASES,
                        help="Minimum cases for a diagnosis to be included")
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / "hybrid_predictions.csv", dtype=str).fillna("")
    print(f"Loaded {len(df)} cases across {df['diagnosis'].nunique()} diagnoses\n")

    rows = []
    for diag, group in df.groupby("diagnosis"):
        if len(group) < args.min_cases:
            continue

        tp_p = fp_p = fn_p = 0
        tp_n = fp_n = fn_n = 0

        for _, row in group.iterrows():
            gt_pos  = parse_and_normalise(row["gt_positive"])
            gt_neg  = parse_and_normalise(row["gt_negative"])
            prd_pos = parse_and_normalise(row["pred_positive"])
            prd_neg = parse_and_normalise(row["pred_negative"])

            tp_p += len(prd_pos & gt_pos);  fp_p += len(prd_pos - gt_pos)
            fn_p += len(gt_pos - prd_pos);  tp_n += len(prd_neg & gt_neg)
            fp_n += len(prd_neg - gt_neg);  fn_n += len(gt_neg - prd_neg)

        p_pos, r_pos, f1_pos = prf(tp_p, fp_p, fn_p)
        p_neg, r_neg, f1_neg = prf(tp_n, fp_n, fn_n)
        _, _, f1_all          = prf(tp_p + tp_n, fp_p + fp_n, fn_p + fn_n)

        rows.append({
            "diagnosis":    diag,
            "n_cases":      len(group),
            "pos_precision": p_pos, "pos_recall": r_pos, "pos_f1": f1_pos,
            "neg_precision": p_neg, "neg_recall": r_neg, "neg_f1": f1_neg,
            "overall_f1":   f1_all,
        })

    results = pd.DataFrame(rows).sort_values("overall_f1", ascending=False)
    results.to_csv(DATA_DIR / "diagnosis_performance.csv", index=False)

    # Print summary
    print(f"{'Diagnosis':<40} {'N':>5} {'Pos F1':>8} {'Neg F1':>8} {'Overall F1':>10}")
    print("-" * 75)
    for _, r in results.iterrows():
        print(f"{r['diagnosis']:<40} {int(r['n_cases']):>5} "
              f"{r['pos_f1']:>8.3f} {r['neg_f1']:>8.3f} {r['overall_f1']:>10.3f}")

    print(f"\nSaved: data/diagnosis_performance.csv  ({len(results)} diagnoses)")
    print(f"\nTop 5 best performing diagnoses:")
    for _, r in results.head(5).iterrows():
        print(f"  {r['diagnosis']} (n={int(r['n_cases'])}) F1={r['overall_f1']:.3f}")
    print(f"\nBottom 5 diagnoses (most challenging):")
    for _, r in results.tail(5).iterrows():
        print(f"  {r['diagnosis']} (n={int(r['n_cases'])}) F1={r['overall_f1']:.3f}")


if __name__ == "__main__":
    main()
