"""
01_data_exploration.py
IHC NLP Project : Data Loading and Exploration
------------------------------------------------
Run this first to sanity-check both spreadsheets before any NLP work.

Usage:
    python 01_data_exploration.py

Update the two file paths below to match your actual file names.
"""

import pandas as pd
from collections import Counter

# ── File paths ────────────────────────────────────────────────────────────────
MAIN_FILE       = "data/main_reports.xlsx"       # 5,000+ reports
GROUND_TRUTH_FILE = "data/ground_truth.xlsx"     # manually annotated cases
# ─────────────────────────────────────────────────────────────────────────────


def load_main(path):
    df = pd.read_excel(path, header=0)
    df.columns = ["case_id", "diagnosis", "report_text"]
    return df


def load_ground_truth(path):
    df = pd.read_excel(path, header=0)
    df.columns = ["case_id", "diagnosis", "report_text", "positive_immuno", "negative_immuno"]

    # Parse comma-separated marker lists into Python lists
    def parse_markers(cell):
        if pd.isna(cell) or str(cell).strip() == "":
            return []
        return [m.strip() for m in str(cell).split(",") if m.strip()]

    df["positive_list"] = df["positive_immuno"].apply(parse_markers)
    df["negative_list"] = df["negative_immuno"].apply(parse_markers)
    return df


def explore_main(df):
    print("=" * 60)
    print("MAIN SPREADSHEET")
    print("=" * 60)
    print(f"Total reports:          {len(df):,}")
    print(f"Unique diagnoses:       {df['diagnosis'].nunique():,}")
    print(f"Missing report text:    {df['report_text'].isna().sum()}")
    print(f"Missing diagnosis:      {df['diagnosis'].isna().sum()}")
    print()

    print("Top 10 diagnoses:")
    for diag, count in df["diagnosis"].value_counts().head(10).items():
        print(f"  {count:>5}  {diag}")
    print()

    # Report text length stats
    df["text_length"] = df["report_text"].astype(str).apply(len)
    print(f"Report text length (characters):")
    print(f"  Min:    {df['text_length'].min():,}")
    print(f"  Median: {int(df['text_length'].median()):,}")
    print(f"  Max:    {df['text_length'].max():,}")
    print()

    print("Sample row (first report):")
    row = df.iloc[0]
    print(f"  Case ID:   {row['case_id']}")
    print(f"  Diagnosis: {row['diagnosis']}")
    print(f"  Text:      {str(row['report_text'])[:300]}...")
    print()


def explore_ground_truth(df):
    print("=" * 60)
    print("GROUND TRUTH SPREADSHEET")
    print("=" * 60)
    print(f"Total annotated cases:  {len(df):,}")
    print(f"Unique diagnoses:       {df['diagnosis'].nunique():,}")
    print(f"Cases with no positives: {df['positive_list'].apply(len).eq(0).sum()}")
    print(f"Cases with no negatives: {df['negative_list'].apply(len).eq(0).sum()}")
    print()

    # Marker counts per case
    pos_counts = df["positive_list"].apply(len)
    neg_counts = df["negative_list"].apply(len)
    print(f"Positive markers per case:  mean={pos_counts.mean():.1f}, max={pos_counts.max()}")
    print(f"Negative markers per case:  mean={neg_counts.mean():.1f}, max={neg_counts.max()}")
    print()

    # Most common markers across the dataset
    all_positive = [m for markers in df["positive_list"] for m in markers]
    all_negative = [m for markers in df["negative_list"] for m in markers]
    all_markers  = set(all_positive + all_negative)

    print(f"Unique markers in ground truth: {len(all_markers)}")
    print()

    print("Top 15 most common positive markers:")
    for marker, count in Counter(all_positive).most_common(15):
        print(f"  {count:>4}  {marker}")
    print()

    print("Top 15 most common negative markers:")
    for marker, count in Counter(all_negative).most_common(15):
        print(f"  {count:>4}  {marker}")
    print()

    print("Sample row (first ground truth case):")
    row = df.iloc[0]
    print(f"  Case ID:   {row['case_id']}")
    print(f"  Diagnosis: {row['diagnosis']}")
    print(f"  Positive:  {row['positive_list']}")
    print(f"  Negative:  {row['negative_list']}")
    print(f"  Text:      {str(row['report_text'])[:300]}...")
    print()


def check_overlap(main_df, gt_df):
    print("=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)
    main_ids = set(main_df["case_id"].astype(str))
    gt_ids   = set(gt_df["case_id"].astype(str))
    overlap  = main_ids & gt_ids
    print(f"Ground truth cases also in main spreadsheet: {len(overlap):,}")
    print(f"Ground truth cases NOT in main spreadsheet:  {len(gt_ids - main_ids):,}")
    print()


if __name__ == "__main__":
    print("\nLoading data...\n")

    main_df = load_main(MAIN_FILE)
    gt_df   = load_ground_truth(GROUND_TRUTH_FILE)

    explore_main(main_df)
    explore_ground_truth(gt_df)
    check_overlap(main_df, gt_df)

    print("Exploration complete. Review the output above before proceeding.")
