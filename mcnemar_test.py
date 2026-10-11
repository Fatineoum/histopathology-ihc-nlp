"""
mcnemar_test.py
IHC NLP Project : McNemar's Test: Rule-based vs ClinicalBERT (8 epochs)
------------------------------------------------------------------------
Compares the two pipelines on the held-out test set of 809 cases.
A case is 'correct' if F1 == 1.0 (exact match on both pos and neg markers).

Usage:
    python mcnemar_test.py
"""

import importlib.util
from pathlib import Path
from collections import defaultdict
import pandas as pd
import math

DATA_DIR = Path("data")
BERT_PREDS = DATA_DIR / "bert_full_predictions_e8.csv"

def load_rule_module():
    spec = importlib.util.spec_from_file_location(
        "rule_pipeline", Path("03_rule_based_pipeline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f

def case_f1(tp_pos, fp_pos, fn_pos, tp_neg, fp_neg, fn_neg):
    """Micro-averaged F1 across pos and neg for a single case."""
    tp = tp_pos + tp_neg
    fp = fp_pos + fp_neg
    fn = fn_pos + fn_neg
    return prf(tp, fp, fn)

def mcnemar(b, c):
    """McNemar's test with continuity correction. b = A wrong B right, c = A right B wrong."""
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    # p-value from chi-squared distribution with 1 df
    import math as chi2_dist
    p_value = math.erfc(math.sqrt(chi2_stat / 2))
    return chi2_stat, p_value

def main():
    print(f"Loading ClinicalBERT predictions from {BERT_PREDS} ...", flush=True)
    bert_df = pd.read_csv(BERT_PREDS)
    print(f"  {len(bert_df)} held-out cases loaded.\n", flush=True)

    mod = load_rule_module()
    print("Running rule-based pipeline on held-out cases ...", flush=True)

    # 2x2 table: rows = rule-based, cols = ClinicalBERT
    # both_correct, rule_only_correct, bert_only_correct, both_wrong
    both_correct = 0
    rule_only    = 0
    bert_only    = 0
    both_wrong   = 0

    rule_f1s = []
    bert_f1s = []

    for _, row in bert_df.iterrows():
        # ClinicalBERT correctness
        bert_f = case_f1(row["tp_pos"], row["fp_pos"], row["fn_pos"],
                         row["tp_neg"], row["fp_neg"], row["fn_neg"])
        bert_correct = (bert_f == 1.0)
        bert_f1s.append(bert_f)

        # Rule-based on same text
        preds = mod.extract_from_text(str(row["ihc_text"]) if row["ihc_text"] else "")
        gt_pos = mod.parse_gt_list(str(row["gt_positive"]) if str(row["gt_positive"]) != "nan" else "")
        gt_neg = mod.parse_gt_list(str(row["gt_negative"]) if str(row["gt_negative"]) != "nan" else "")
        stats = mod.evaluate_case(preds, gt_pos, gt_neg)
        rule_f = case_f1(stats["tp_pos"], stats["fp_pos"], stats["fn_pos"],
                         stats["tp_neg"], stats["fp_neg"], stats["fn_neg"])
        rule_correct = (rule_f == 1.0)
        rule_f1s.append(rule_f)

        if rule_correct and bert_correct:
            both_correct += 1
        elif rule_correct and not bert_correct:
            rule_only += 1
        elif not rule_correct and bert_correct:
            bert_only += 1
        else:
            both_wrong += 1

    n = len(bert_df)
    chi2_stat, p_val = mcnemar(rule_only, bert_only)

    print(f"\n{'=' * 60}")
    print(f"  McNemar's Test: Rule-based vs ClinicalBERT (8 epochs)")
    print(f"  Held-out test set: {n} cases")
    print(f"{'=' * 60}")
    print(f"  Exact-match F1=1.0 per case:")
    print(f"    Both correct:           {both_correct:>4}  ({both_correct/n:.1%})")
    print(f"    Rule correct, BERT wrong:{rule_only:>4}  ({rule_only/n:.1%})")
    print(f"    BERT correct, Rule wrong:{bert_only:>4}  ({bert_only/n:.1%})")
    print(f"    Both wrong:             {both_wrong:>4}  ({both_wrong/n:.1%})")
    print(f"\n  Mean case-level F1:")
    print(f"    Rule-based:    {sum(rule_f1s)/len(rule_f1s):.4f}")
    print(f"    ClinicalBERT:  {sum(bert_f1s)/len(bert_f1s):.4f}")
    print(f"\n  McNemar's test (with continuity correction):")
    print(f"    chi2 = {chi2_stat:.4f},  p = {p_val:.6f}")
    sig = "statistically significant (p < 0.05)" if p_val < 0.05 else "not statistically significant"
    print(f"    Result: {sig}")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
