"""
02_data_cleaning.py
IHC NLP Project : Data Cleaning and Extraction
------------------------------------------------
Extracts the columns we care about from both raw spreadsheets,
deduplicates by patient ID, normalises nil/NA values, flags
anonymous marker placeholders (x1, x2, x3...), and produces
a clean pilot subset for leiomyosarcoma HG.

Outputs (all saved to data/):
    cleaned_ground_truth.csv   : full cleaned ground truth
    cleaned_no_immuno.csv      : full cleaned no-immuno reports
    flagged_anonymous.csv      : GT cases with x1/x2/x3 markers (for review)
    pilot_leiomyosarcoma.csv   : leiomyosarcoma HG pilot subset

Usage:
    python 02_data_cleaning.py
"""

import pandas as pd
import re
from pathlib import Path

DATA_DIR   = Path("data")
GT_FILE    = DATA_DIR / "reports_ground_truth.xlsx"
NI_FILE    = DATA_DIR / "reports_no_immuno.xlsx"

PILOT_DIAGNOSIS = "leiomyosarcoma hg"

# ── Helpers ────────────────────────────────────────────────────────────────────

NIL_VALUES = {"nil", "na", "n/a", "n/a", "-", "none", ""}

def load_file(path, has_immuno_cols=True):
    """Load raw Excel, skip the merged top header row, rename key columns."""
    df = pd.read_excel(path, header=1, skiprows=[0], dtype=str)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # Rename by position (columns are consistent across both files)
    col_map = {
        df.columns[0]:  "patient_id",   # A
        df.columns[3]:  "slide_id",     # D
        df.columns[4]:  "diagnosis",    # E
        df.columns[17]: "ihc_text",     # R
    }
    if has_immuno_cols:
        col_map[df.columns[18]] = "positive_ihc"   # S
        col_map[df.columns[19]] = "negative_ihc"   # T

    df = df.rename(columns=col_map)

    # Keep only the columns we renamed
    keep = list(col_map.values())
    df = df[keep].copy()

    return df


def normalise_nil(value):
    """Return empty string if value is a nil-equivalent, else return stripped value."""
    if pd.isna(value):
        return ""
    cleaned = str(value).strip().lower()
    if cleaned in NIL_VALUES:
        return ""
    return str(value).strip()


def parse_markers(cell):
    """Parse a comma-separated marker cell into a sorted list of strings."""
    val = normalise_nil(cell)
    if not val:
        return []
    return sorted([m.strip() for m in val.split(",") if m.strip()])


def has_anonymous(cell):
    """Return True if cell contains x1, x2, x3 ... placeholder markers."""
    val = normalise_nil(cell)
    if not val:
        return False
    return bool(re.search(r'\bx\d+\b', val, re.IGNORECASE))


def completeness_score(row, has_immuno=True):
    """
    Score a row by how complete its key fields are.
    Higher = more complete. Used to pick the best record when deduplicating.
    """
    score = 0
    if pd.notna(row.get("ihc_text", "")) and str(row.get("ihc_text", "")).strip():
        score += 2
    if has_immuno:
        pos = normalise_nil(row.get("positive_ihc", ""))
        neg = normalise_nil(row.get("negative_ihc", ""))
        if pos:
            score += len(pos.split(","))
        if neg:
            score += len(neg.split(","))
    return score


def deduplicate(df, has_immuno=True):
    """
    For each patient_id, keep the single most complete record.
    Ties are broken by keeping the first occurrence.
    """
    df = df.copy()
    df["_score"] = df.apply(lambda r: completeness_score(r, has_immuno), axis=1)
    df = df.sort_values("_score", ascending=False)
    df = df.drop_duplicates(subset="patient_id", keep="first")
    df = df.drop(columns=["_score"])
    return df.reset_index(drop=True)


# ── Ground truth cleaning ──────────────────────────────────────────────────────

def clean_ground_truth(path):
    df = load_file(path, has_immuno_cols=True)

    print(f"GT raw rows:              {len(df):,}")

    # Normalise nil/NA in IHC columns
    df["positive_ihc"] = df["positive_ihc"].apply(normalise_nil)
    df["negative_ihc"] = df["negative_ihc"].apply(normalise_nil)

    # Parse marker lists (for stats : stored as pipe-joined string for CSV compatibility)
    df["positive_list"] = df["positive_ihc"].apply(parse_markers)
    df["negative_list"] = df["negative_ihc"].apply(parse_markers)

    # Flag anonymous markers
    df["has_anonymous"] = (
        df["positive_ihc"].apply(has_anonymous) |
        df["negative_ihc"].apply(has_anonymous)
    )

    anon_count = df["has_anonymous"].sum()
    print(f"GT rows with x1/x2/x3:   {anon_count:,}  (flagged, not removed)")

    # Deduplicate by patient ID
    before = len(df)
    df = deduplicate(df, has_immuno=True)
    print(f"GT after deduplication:   {len(df):,}  (removed {before - len(df)} duplicates)")

    # Separate flagged cases for review
    flagged = df[df["has_anonymous"]].copy()
    clean   = df.copy()  # keep all in main file, flag column tells you which to exclude

    print(f"GT clean (no x1/x2/x3):  {(~clean['has_anonymous']).sum():,}")
    print(f"GT flagged for review:    {len(flagged):,}")

    return clean, flagged


# ── No-immuno cleaning ─────────────────────────────────────────────────────────

def clean_no_immuno(path):
    df = load_file(path, has_immuno_cols=False)

    print(f"\nNI raw rows:              {len(df):,}")

    before = len(df)
    df = deduplicate(df, has_immuno=False)
    print(f"NI after deduplication:   {len(df):,}  (removed {before - len(df)} duplicates)")

    return df


# ── Pilot subset ───────────────────────────────────────────────────────────────

def make_pilot(df, diagnosis=PILOT_DIAGNOSIS):
    """
    Extract cases matching the pilot diagnosis.
    Excludes rows flagged as having anonymous markers.
    """
    mask = (
        (df["diagnosis"].str.strip().str.lower() == diagnosis.lower()) &
        (~df["has_anonymous"])
    )
    pilot = df[mask].copy()
    print(f"\nPilot ({diagnosis}):")
    print(f"  Total matching cases:   {len(pilot):,}")
    print(f"  With IHC text:          {pilot['ihc_text'].notna().sum():,}")
    print(f"  With positives listed:  {(pilot['positive_ihc'] != '').sum():,}")
    print(f"  With negatives listed:  {(pilot['negative_ihc'] != '').sum():,}")
    return pilot


# ── Save outputs ───────────────────────────────────────────────────────────────

def save(df, filename, drop_cols=None):
    out = df.copy()
    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    path = DATA_DIR / filename
    out.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(out):,} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CLEANING GROUND TRUTH")
    print("=" * 55)
    gt_clean, gt_flagged = clean_ground_truth(GT_FILE)

    print("\n" + "=" * 55)
    print("CLEANING NO-IMMUNO REPORTS")
    print("=" * 55)
    ni_clean = clean_no_immuno(NI_FILE)

    print("\n" + "=" * 55)
    print("PILOT SUBSET")
    print("=" * 55)
    pilot = make_pilot(gt_clean, PILOT_DIAGNOSIS)

    print("\n" + "=" * 55)
    print("SAVING OUTPUTS")
    print("=" * 55)
    # Drop the parsed list columns (redundant with the string columns in CSV)
    list_cols = ["positive_list", "negative_list"]

    save(gt_clean,    "cleaned_ground_truth.csv", drop_cols=list_cols)
    save(gt_flagged,  "flagged_anonymous.csv",    drop_cols=list_cols)
    save(ni_clean,    "cleaned_no_immuno.csv")
    save(pilot,       "pilot_leiomyosarcoma.csv", drop_cols=list_cols)

    print("\nDone. Review flagged_anonymous.csv before deciding whether")
    print("to include or exclude those cases from training.")
