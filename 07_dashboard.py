"""
07_dashboard.py
IHC NLP Project -- Streamlit Decision-Support Dashboard
--------------------------------------------------------
Interactive prototype dashboard for exploring IHC marker patterns
extracted by the hybrid NLP pipeline.

Four views:
  1. Diagnosis Panel       -- marker positivity/negativity rates per diagnosis
  2. Literature Comparison -- institutional rates vs published expected profiles
  3. LMS Explorer          -- subclassification breakdown for leiomyosarcoma groups
  4. Case Browser          -- searchable table of individual cases

Usage:
    streamlit run 07_dashboard.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from marker_utils import CANONICAL_MARKERS

DB_PATH = Path("data/ihc_database.db")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IHC Knowledge Base Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Metric cards ─────────────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #e2e8f0;
    border-top: 4px solid #0f766e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    height: 100%;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f766e;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
    font-weight: 500;
}

/* ── Section banner ───────────────────────────────────────────────────── */
.section-banner {
    background: linear-gradient(135deg, #0f766e 0%, #0e7490 100%);
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(15,118,110,0.25);
}
.section-banner-title {
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.section-banner-sub {
    color: #99f6e4;
    font-size: 0.85rem;
    margin: 0;
}

/* ── Alert/info boxes ─────────────────────────────────────────────────── */
.warning-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #78350f;
}
.info-box {
    background: #f0fdf4;
    border-left: 4px solid #0f766e;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.85rem;
    color: #14532d;
}

/* ── Marker badges ────────────────────────────────────────────────────── */
.badge-pos {
    display: inline-block;
    background: #dcfce7; color: #166534;
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.8rem; font-weight: 600; margin: 2px;
}
.badge-neg {
    display: inline-block;
    background: #fee2e2; color: #991b1b;
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.8rem; font-weight: 600; margin: 2px;
}
.badge-eq {
    display: inline-block;
    background: #fef9c3; color: #713f12;
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.8rem; font-weight: 600; margin: 2px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebarContent"] { padding-top: 1rem; }

/* ── Dataframe zebra rows ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] tr:nth-child(even) td {
    background: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


# ── Feature flags ─────────────────────────────────────────────────────────────
# Set to True to re-enable Ki-67 histogram and box plots in the dashboard.
# Data is preserved in the ki67_results table; load_ki67() still works.
SHOW_KI67 = True


# ── Database helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}. Run `python 06_database.py --build` first.")
        st.stop()
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=MEMORY")
    return con


@st.cache_data(ttl=300)
def load_diagnoses() -> pd.DataFrame:
    con = get_connection()
    return pd.read_sql(
        "SELECT canonical_diagnosis, display_name, n_gt_cases, include_in_analysis, notes "
        "FROM diagnoses ORDER BY n_gt_cases DESC",
        con
    )


@st.cache_data(ttl=300)
def load_marker_frequencies(diagnosis: str, source: str = "pipeline") -> pd.DataFrame:
    """Positive and negative marker frequencies for a diagnosis."""
    con = get_connection()
    df = pd.read_sql("""
        SELECT
            m.marker,
            m.outcome,
            COUNT(DISTINCT m.patient_id) AS n_cases,
            ROUND(100.0 * COUNT(DISTINCT m.patient_id) /
                  (SELECT COUNT(*) FROM cases WHERE canonical_diagnosis = :diag), 1) AS pct
        FROM ihc_markers m
        JOIN cases c ON m.patient_id = c.patient_id
        WHERE c.canonical_diagnosis = :diag
          AND m.source = :source
          AND m.outcome IN ('positive', 'negative', 'equivocal')
        GROUP BY m.marker, m.outcome
        ORDER BY n_cases DESC
    """, con, params={"diag": diagnosis, "source": source})
    return df


@st.cache_data(ttl=300)
def load_literature_comparison(diagnosis: str) -> pd.DataFrame:
    con = get_connection()
    total = pd.read_sql(
        "SELECT COUNT(*) AS n FROM cases WHERE canonical_diagnosis = ?",
        con, params=(diagnosis,)
    ).iloc[0]["n"]

    df = pd.read_sql("""
        SELECT
            l.marker,
            l.expected_outcome,
            l.typical_frequency_pct  AS lit_pct,
            l.citation,
            COUNT(DISTINCT CASE WHEN m.outcome = l.expected_outcome
                                THEN m.patient_id END) AS our_n,
            COUNT(DISTINCT CASE WHEN m.outcome IS NOT NULL
                                THEN m.patient_id END) AS tested_n
        FROM literature l
        LEFT JOIN cases c ON c.canonical_diagnosis = l.canonical_diagnosis
        LEFT JOIN ihc_markers m
               ON m.patient_id = c.patient_id
              AND m.marker = l.marker
              AND m.source = 'pipeline'
        WHERE l.canonical_diagnosis = ?
        GROUP BY l.marker, l.expected_outcome, l.typical_frequency_pct, l.citation
        ORDER BY l.typical_frequency_pct DESC
    """, con, params=(diagnosis,))
    # our_pct: % of ALL cases with the expected outcome (includes untested cases as denominator)
    df["our_pct"] = (df["our_n"] / total * 100).round(1).fillna(0) if total > 0 else 0.0
    # tested_pct: % of cases WHERE THE MARKER WAS TESTED that showed the expected outcome
    df["tested_pct"] = df.apply(
        lambda r: round(r["our_n"] / r["tested_n"] * 100, 1) if r["tested_n"] > 0 else None,
        axis=1,
    )
    # coverage_pct: % of all cases where this marker was tested at all
    df["coverage_pct"] = (df["tested_n"] / total * 100).round(1).fillna(0) if total > 0 else 0.0
    return df, total


@st.cache_data(ttl=300)
def load_lms_subclass_markers(diagnosis: str, subclass: str | None) -> pd.DataFrame:
    con = get_connection()
    if subclass and subclass not in ("(all)", "(none)"):
        params = {"diag": diagnosis, "sub": subclass}
        where_sub = "AND c.subclassification = :sub"
    elif subclass == "(none)":
        params = {"diag": diagnosis}
        where_sub = "AND c.subclassification IS NULL"
    else:
        params = {"diag": diagnosis}
        where_sub = ""

    total_sql = (
        f"SELECT COUNT(*) FROM cases c WHERE c.canonical_diagnosis = :diag "
        f"{where_sub}"
    )
    total = pd.read_sql(total_sql, con, params=params).iloc[0, 0]
    if total == 0:
        return pd.DataFrame(), 0

    df = pd.read_sql(f"""
        SELECT
            m.marker,
            m.outcome,
            COUNT(DISTINCT m.patient_id) AS n,
            ROUND(100.0 * COUNT(DISTINCT m.patient_id) / {total}, 1) AS pct
        FROM ihc_markers m
        JOIN cases c ON m.patient_id = c.patient_id
        WHERE c.canonical_diagnosis = :diag
          {where_sub}
          AND m.source = 'pipeline'
          AND m.outcome IN ('positive', 'negative')
        GROUP BY m.marker, m.outcome
        ORDER BY n DESC
    """, con, params=params)
    return df, total


@st.cache_data(ttl=300)
def load_subclass_distribution(diagnosis: str) -> pd.DataFrame:
    con = get_connection()
    return pd.read_sql("""
        SELECT
            COALESCE(subclassification, '(none)') AS subclassification,
            COUNT(*) AS n_cases
        FROM cases
        WHERE canonical_diagnosis = ?
        GROUP BY subclassification
        ORDER BY n_cases DESC
    """, con, params=(diagnosis,))


@st.cache_data(ttl=300)
def load_ki67(diagnosis: str, subclass: str | None = None) -> pd.DataFrame:
    con = get_connection()
    if subclass and subclass != "(all)":
        params = (diagnosis, subclass)
        sub_where = "AND c.subclassification = ?"
    else:
        params = (diagnosis,)
        sub_where = ""
    return pd.read_sql(f"""
        SELECT k.ki67_pct, COALESCE(c.subclassification, '(none)') AS subclassification
        FROM ki67_results k
        JOIN cases c ON k.patient_id = c.patient_id
        WHERE c.canonical_diagnosis = ? {sub_where}
        ORDER BY k.ki67_pct
    """, con, params=params)


@st.cache_data(ttl=300)
def load_cases(diagnosis: str, subclass: str | None = None,
               marker_filter: str | None = None,
               outcome_filter: str | None = None) -> pd.DataFrame:
    con = get_connection()
    query = """
        SELECT DISTINCT c.patient_id, c.canonical_diagnosis, c.subclassification,
               c.ihc_text
        FROM cases c
        WHERE c.canonical_diagnosis = ?
    """
    params: list = [diagnosis]
    if subclass and subclass != "(all)":
        query += " AND c.subclassification = ?"
        params.append(subclass)
    if marker_filter:
        query += """
            AND c.patient_id IN (
                SELECT patient_id FROM ihc_markers
                WHERE marker = ? AND source = 'pipeline'
        """
        params.append(marker_filter)
        if outcome_filter:
            query += " AND outcome = ?"
            params.append(outcome_filter)
        query += ")"
    query += " ORDER BY c.patient_id LIMIT 200"
    return pd.read_sql(query, con, params=params)


@st.cache_data(ttl=300)
def load_case_markers(patient_id: str) -> pd.DataFrame:
    con = get_connection()
    return pd.read_sql("""
        SELECT marker, outcome, source
        FROM ihc_markers
        WHERE patient_id = ?
        GROUP BY marker, outcome, source
        ORDER BY source, outcome, marker
    """, con, params=(patient_id,))


@st.cache_data(ttl=300)
def load_all_cases_global(
    pid_search: str = "",
    diag_filter: str = "(all)",
    marker_name: str = "",
    marker_outcome: str = "positive",
    fusion_filter: str = "",
) -> pd.DataFrame:
    """Universal case browser -- no diagnosis required."""
    con = get_connection()
    wheres: list[str] = []
    params: dict = {}
    if pid_search:
        wheres.append("c.patient_id LIKE :pid")
        params["pid"] = f"%{pid_search}%"
    if diag_filter and diag_filter != "(all)":
        wheres.append("c.canonical_diagnosis = :diag")
        params["diag"] = diag_filter
    if marker_name:
        wheres.append("""c.patient_id IN (
            SELECT patient_id FROM ihc_markers
            WHERE marker = :mname AND outcome = :mout AND source = 'pipeline'
        )""")
        params["mname"] = marker_name
        params["mout"]  = marker_outcome
    if fusion_filter:
        # fusion_filter is a grouped/normalized name (e.g. "EWSR1 rearrangement")
        # find all patient_ids whose normalized positive_result matches
        _con_fus = get_connection()
        _mol_rows = _con_fus.execute(
            "SELECT DISTINCT patient_id, positive_result FROM molecular_tests "
            "WHERE positive_result IS NOT NULL AND positive_result != ''"
        ).fetchall()
        _match_pids = [pid for pid, pr in _mol_rows if normalize_molecular(pr) == fusion_filter]
        if not _match_pids:
            return pd.DataFrame(columns=["patient_id", "canonical_diagnosis", "has_gt", "ihc_text"])
        # Embed patient IDs directly as a literal IN list (safe: values come from DB, not user input)
        pid_list_sql = ",".join(f"'{p}'" for p in _match_pids)
        wheres.append(f"c.patient_id IN ({pid_list_sql})")
    where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    return pd.read_sql(f"""
        SELECT c.patient_id, c.canonical_diagnosis, c.has_gt, c.ihc_text
        FROM cases c
        {where_sql}
        ORDER BY c.canonical_diagnosis, c.patient_id
        LIMIT 500
    """, con, params=params)


# Right-gene-is-driver fusions: these genes are the diagnostic target regardless of partner
_RIGHT_GENE_DRIVERS = {'ALK', 'TFE3', 'VGLL3', 'NUTM1', 'NUT', 'FOXO1', 'FKHR'}

import re as _re

def normalize_molecular(text: str) -> str:
    """Return the canonical gene group name for a molecular finding.
    All variant forms collapse to just the gene name:
      CTNNB1 (p.Ser45Phe) -> CTNNB1
      CTNNB1 mutation      -> CTNNB1
      CTNNB1 rearrangement -> CTNNB1
      EWSR1::FLI1          -> EWSR1
      CARS1::ALK           -> ALK   (right-gene driver)
      TFE3 rearrangement   -> TFE3
      TFE3                 -> TFE3
      MDM2 amplification   -> MDM2
      BCOR ITD             -> BCOR
    """
    if not text:
        return text
    # Protein-change notation: GENE (p.xxx) -> GENE
    m = _re.match(r'^([\w\d]+)\s*\(p\.', text)
    if m:
        return m.group(1)
    # Fusion: GENE::PARTNER
    if '::' in text:
        left, right = text.split('::', 1)
        right_gene = right.split()[0].rstrip(',.;()').strip()
        if right_gene.upper() in _RIGHT_GENE_DRIVERS:
            return right_gene
        return left.strip()
    # "GENE word..." -> first word is the gene
    return text.split()[0].rstrip(',.;()').strip()


def get_detail_label(text: str) -> str:
    """Detail-mode display label: return the stored value as-is."""
    return text or ""


@st.cache_data(ttl=300)
def load_molecular_positives(canonical_diagnosis: str) -> pd.DataFrame:
    """Fusions/rearrangements confirmed in our cases for a diagnosis (confirmed only, grouped)."""
    con = get_connection()
    df = pd.read_sql(
        """
        SELECT mt.positive_result, mt.patient_id
        FROM molecular_tests mt
        JOIN cases c ON c.patient_id = mt.patient_id
        JOIN diagnoses d ON d.canonical_diagnosis = c.canonical_diagnosis
        WHERE c.canonical_diagnosis = :dx
          AND d.is_confirmed = 1
          AND mt.positive_result IS NOT NULL AND mt.positive_result != ''
        """,
        con, params={"dx": canonical_diagnosis},
    )
    if df.empty:
        return df
    df["grouped"] = df["positive_result"].apply(normalize_molecular)
    out = (df.groupby("grouped")["patient_id"]
             .nunique()
             .reset_index()
             .rename(columns={"grouped": "positive_result", "patient_id": "n_cases"})
             .sort_values("n_cases", ascending=False))
    return out


@st.cache_data(ttl=300)
def load_all_fusions() -> pd.DataFrame:
    """All confirmed fusions/rearrangements across diagnoses (confirmed only, grouped)."""
    con = get_connection()
    df = pd.read_sql(
        """
        SELECT mt.positive_result, c.canonical_diagnosis, mt.patient_id
        FROM molecular_tests mt
        JOIN cases c ON c.patient_id = mt.patient_id
        JOIN diagnoses d ON d.canonical_diagnosis = c.canonical_diagnosis
        WHERE d.is_confirmed = 1
          AND mt.positive_result IS NOT NULL AND mt.positive_result != ''
        """,
        con,
    )
    if df.empty:
        return df
    df["positive_result"] = df["positive_result"].apply(normalize_molecular)
    out = (df.groupby(["positive_result", "canonical_diagnosis"])["patient_id"]
             .nunique()
             .reset_index()
             .rename(columns={"patient_id": "n_cases"})
             .sort_values("n_cases", ascending=False))
    return out


@st.cache_data(ttl=300)
def load_distinct_fusions() -> list[str]:
    """Distinct grouped finding labels for the Case Browser dropdown (confirmed diagnoses only)."""
    con = get_connection()
    rows = con.execute(
        """SELECT DISTINCT mt.positive_result FROM molecular_tests mt
           JOIN cases c ON c.patient_id = mt.patient_id
           JOIN diagnoses d ON d.canonical_diagnosis = c.canonical_diagnosis
           WHERE d.is_confirmed = 1
             AND mt.positive_result IS NOT NULL AND mt.positive_result != ''
           ORDER BY mt.positive_result"""
    ).fetchall()
    grouped = sorted(set(normalize_molecular(r[0]) for r in rows))
    return grouped


@st.cache_data(ttl=300)
def load_literature_all() -> pd.DataFrame:
    con = get_connection()
    return pd.read_sql(
        "SELECT canonical_diagnosis, marker, expected_outcome, typical_frequency_pct "
        "FROM literature",
        con,
    )


@st.cache_data(ttl=300)
def load_data_marker_freq_all() -> pd.DataFrame:
    """
    Per-diagnosis, per-marker outcome frequencies.
    Denominator = cases where this marker was actually tested (positive OR negative),
    not total cases for the diagnosis. This avoids inflating 'missing' markers as negatives.
    Only rows with >= 5 tested cases are included (below that the estimate is unreliable).
    """
    con = get_connection()
    return pd.read_sql("""
        SELECT
            c.canonical_diagnosis,
            m.marker,
            m.outcome,
            COUNT(DISTINCT m.patient_id) AS n,
            (
                SELECT COUNT(DISTINCT m2.patient_id)
                FROM ihc_markers m2
                JOIN cases c2 ON m2.patient_id = c2.patient_id
                WHERE c2.canonical_diagnosis = c.canonical_diagnosis
                  AND m2.marker = m.marker
                  AND m2.source = 'pipeline'
                  AND m2.outcome IN ('positive', 'negative')
            ) AS tested,
            ROUND(100.0 * COUNT(DISTINCT m.patient_id) /
                (
                    SELECT COUNT(DISTINCT m2.patient_id)
                    FROM ihc_markers m2
                    JOIN cases c2 ON m2.patient_id = c2.patient_id
                    WHERE c2.canonical_diagnosis = c.canonical_diagnosis
                      AND m2.marker = m.marker
                      AND m2.source = 'pipeline'
                      AND m2.outcome IN ('positive', 'negative')
                ), 1) AS pct
        FROM ihc_markers m
        JOIN cases c ON m.patient_id = c.patient_id
        WHERE m.source = 'pipeline'
          AND m.outcome IN ('positive', 'negative')
        GROUP BY c.canonical_diagnosis, m.marker, m.outcome
        HAVING tested >= 5
    """, con)


@st.cache_data(ttl=300)
def load_molecular_tests(patient_id: str) -> pd.DataFrame:
    """Return molecular test records for a given patient, or empty df."""
    con = get_connection()
    return pd.read_sql(
        "SELECT test_type, negative_probes, positive_result, raw_finding, source_file "
        "FROM molecular_tests WHERE patient_id=:pid",
        con, params={"pid": patient_id},
    )


@st.cache_data(ttl=300)
def load_molecular_dx_markers(canonical_diagnosis: str) -> pd.DataFrame:
    """Return knowledge-base entries for a diagnosis."""
    con = get_connection()
    return pd.read_sql(
        "SELECT test_type, alteration, role, prevalence_pct, note "
        "FROM molecular_dx_markers WHERE canonical_diagnosis=:dx "
        "ORDER BY role",
        con, params={"dx": canonical_diagnosis},
    )


@st.cache_data(ttl=300)
def load_morphology_note(patient_id: str) -> str | None:
    """Return morphology_notes for a patient if populated."""
    con = get_connection()
    row = con.execute(
        "SELECT morphology_notes FROM cases WHERE patient_id=?", (patient_id,)
    ).fetchone()
    return row[0] if row and row[0] else None


@st.cache_data(ttl=300)
def load_marker_diag_chart(marker: str) -> pd.DataFrame:
    """Positive frequency of a single marker across confirmed diagnoses only."""
    con = get_connection()
    return pd.read_sql("""
        SELECT
            c.canonical_diagnosis,
            m.outcome,
            COUNT(DISTINCT m.patient_id) AS n,
            d.n_gt_cases AS total,
            ROUND(100.0 * COUNT(DISTINCT m.patient_id) / d.n_gt_cases, 1) AS pct
        FROM ihc_markers m
        JOIN cases c ON m.patient_id = c.patient_id
        JOIN diagnoses d ON d.canonical_diagnosis = c.canonical_diagnosis
        WHERE m.marker = :marker AND m.source = 'pipeline'
          AND m.outcome IN ('positive', 'negative')
          AND d.include_in_analysis = 1
        GROUP BY c.canonical_diagnosis, m.outcome
        ORDER BY pct DESC
    """, con, params={"marker": marker})


def _write_to_db(patient_id: str, raw_diag: str, canon_diag: str,
                 ihc_text: str, pred_pos: list, pred_neg: list) -> None:
    """Write a manually-reviewed case to the database (bypasses cache)."""
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=MEMORY")
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO cases
            (patient_id, raw_diagnosis, canonical_diagnosis, subclassification, has_gt, ihc_text)
        VALUES (?, ?, ?, NULL, 0, ?)
    """, (patient_id, raw_diag, canon_diag, ihc_text))
    cur.execute("DELETE FROM ihc_markers WHERE patient_id = ? AND source = 'pipeline'",
                (patient_id,))
    for m in pred_pos:
        cur.execute(
            "INSERT OR REPLACE INTO ihc_markers (patient_id, marker, outcome, source) "
            "VALUES (?, ?, 'positive', 'pipeline')", (patient_id, m))
    for m in pred_neg:
        cur.execute(
            "INSERT OR REPLACE INTO ihc_markers (patient_id, marker, outcome, source) "
            "VALUES (?, ?, 'negative', 'pipeline')", (patient_id, m))
    con.commit()
    con.close()
    st.cache_data.clear()


def _compute_ddx(pos_markers: list, neg_markers: list) -> pd.DataFrame:
    """
    Score every diagnosis from literature + our data; return top matches.

    Literature score: of the provided markers that appear in the literature for this
    diagnosis, what fraction match the expected outcome?  Contradictions (wrong direction)
    reduce the score proportionally.

    Data score: of the provided markers where we have >= 5 tested cases for this diagnosis,
    what fraction are consistent with our institutional frequencies (>= 50% in expected
    direction)? Only computed when at least 2 markers have enough data.
    """
    if not pos_markers and not neg_markers:
        return pd.DataFrame()

    lit       = load_literature_all()
    data_freq = load_data_marker_freq_all()
    diag_names = load_diagnoses().set_index("canonical_diagnosis")["display_name"].to_dict()

    input_map: dict[str, str] = {m: "positive" for m in pos_markers}
    input_map.update({m: "negative" for m in neg_markers})

    rows = []
    for diag in lit["canonical_diagnosis"].unique():
        diag_lit  = lit[lit["canonical_diagnosis"] == diag]
        diag_data = data_freq[data_freq["canonical_diagnosis"] == diag]

        lit_match_w, lit_total_w = 0.0, 0.0
        supporting, contradicting = [], []
        covered = 0

        for marker, outcome in input_map.items():
            lr = diag_lit[diag_lit["marker"] == marker]
            if lr.empty:
                continue
            covered += 1
            expected = lr.iloc[0]["expected_outcome"]
            freq     = lr.iloc[0]["typical_frequency_pct"]
            weight   = freq / 100.0          # higher-frequency markers carry more weight
            lit_total_w += weight
            if expected == outcome:
                lit_match_w += weight
                supporting.append(f"{marker} ({freq:.0f}%)")
            else:
                # Do not add; effectively penalty via reduced numerator
                contradicting.append(f"{marker} ({freq:.0f}%)")

        if covered == 0:
            continue

        lit_score = (lit_match_w / lit_total_w * 100) if lit_total_w > 0 else 0.0

        # Data score (only when we have sufficient tested cases for >= 2 markers)
        data_matches, data_covered = 0, 0
        for marker, outcome in input_map.items():
            dr = diag_data[
                (diag_data["marker"] == marker) & (diag_data["outcome"] == outcome)
            ]
            if dr.empty:
                continue
            data_covered += 1
            if dr.iloc[0]["pct"] >= 50:
                data_matches += 1

        data_score = (data_matches / data_covered * 100) if data_covered >= 2 else None

        # Coverage fraction: how many of the provided markers appear in this
        # diagnosis's literature? A diagnosis matching 1 / 10 markers perfectly
        # should rank below one matching 7 / 10 perfectly.
        n_input = len(input_map)
        coverage_frac = covered / n_input  # 0 < coverage_frac <= 1

        # Base score: average of available scores (lit only, or lit + data)
        base = (lit_score + data_score) / 2.0 if data_score is not None else lit_score

        # Combined ranking score is penalised by how much of the input the
        # literature actually covers. This prevents a diagnosis with 1 matching
        # marker outranking one with 7 matching markers at the same match rate.
        combined = base * coverage_frac

        rows.append({
            "canonical_diagnosis": diag,
            "display_name":        diag_names.get(diag, diag.replace("_", " ").title()),
            "lit_score":           round(lit_score, 1),
            "data_score":          round(data_score, 1) if data_score is not None else None,
            "combined_score":      round(combined, 2),
            "supporting":          supporting,
            "contradicting":       contradicting,
            "covered":             covered,
            "n_input":             n_input,
            "data_covered":        data_covered,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("combined_score", ascending=False).reset_index(drop=True)


def _suggest_markers(
    pos_markers: list, neg_markers: list, top_diags: list, lit: pd.DataFrame
) -> list[dict]:
    """Return markers that best discriminate between the top two differential diagnoses."""
    if len(top_diags) < 2:
        return []
    provided = set(pos_markers) | set(neg_markers)
    d1, d2  = top_diags[0], top_diags[1]
    lit1    = lit[lit["canonical_diagnosis"] == d1]
    lit2    = lit[lit["canonical_diagnosis"] == d2]
    all_m   = set(lit1["marker"]) | set(lit2["marker"])
    suggestions = []
    for m in sorted(all_m - provided):
        r1 = lit1[lit1["marker"] == m]
        r2 = lit2[lit2["marker"] == m]
        if r1.empty or r2.empty:
            continue
        o1, f1 = r1.iloc[0]["expected_outcome"], r1.iloc[0]["typical_frequency_pct"]
        o2, f2 = r2.iloc[0]["expected_outcome"], r2.iloc[0]["typical_frequency_pct"]
        if o1 != o2:
            power = (f1 + f2) / 2
        else:
            power = abs(f1 - f2)
            if power < 30:
                continue
        suggestions.append({
            "marker":          m,
            "expected_in_d1":  f"{o1} ({f1:.0f}%)",
            "expected_in_d2":  f"{o2} ({f2:.0f}%)",
            "power":           power,
        })
    return sorted(suggestions, key=lambda x: x["power"], reverse=True)[:5]


def _render_ddx_results(ddx_df: "pd.DataFrame", pos_snap: list, neg_snap: list,
                        key_prefix: str = "da", patient_id: str | None = None) -> None:
    """Shared DDx results renderer used by both Pipeline Runner and Diagnostic Assistant."""
    diag_names_r = load_diagnoses().set_index("canonical_diagnosis")["display_name"].to_dict()
    top3 = ddx_df.head(3)

    st.subheader("Top 3 differential diagnoses")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        rank_label   = {1: "1st", 2: "2nd", 3: "3rd"}[rank]
        border_color = {1: "#0f766e", 2: "#4f46e5", 3: "#0891b2"}[rank]
        rank_bg      = {1: "#f0fdf4", 2: "#eef2ff", 3: "#f0f9ff"}[rank]
        n_cov  = int(row.get("covered", 0))
        n_tot  = int(row.get("n_input", 0))
        st.markdown(
            f'<div style="border-left:5px solid {border_color};'
            f'padding:14px 18px;border-radius:8px;margin-bottom:10px;'
            f'background:{rank_bg};box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<span style="font-size:0.7rem;font-weight:700;color:{border_color};'
            f'text-transform:uppercase;letter-spacing:0.08em">'
            f'{rank_label} DIFFERENTIAL</span>'
            f'<span style="font-size:0.75rem;color:#9ca3af;margin-left:12px">'
            f'{n_cov} of {n_tot} markers in literature</span><br>'
            f'<span style="font-size:1.2rem;font-weight:700;color:#0f172a">{row["display_name"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        sc1, sc2 = st.columns(2)
        with sc1:
            lit_val = row["lit_score"]
            lit_str = f"{lit_val:.0f}%" if pd.notna(lit_val) else "n/a"
            st.metric(
                "Literature score", lit_str,
                help="Weighted match rate against published marker profiles for this diagnosis. "
                     "High-frequency markers count more. "
                     "Ranking also penalises low coverage (few of your markers appear in literature).",
            )
        with sc2:
            dat_val = row["data_score"]
            dat_str = f"{dat_val:.0f}%" if pd.notna(dat_val) else "n/a"
            st.metric(
                "Our data score", dat_str,
                help="Match rate against our institutional pipeline data. "
                     "Shown only when >= 5 cases with this diagnosis were tested for at least "
                     "2 of your provided markers. 'n/a' means insufficient institutional data.",
            )
        for markers_list, label, icon, css_class in [
            (row["supporting"],    "Literature supports",    "✓", "badge-pos"),
            (row["contradicting"], "Literature contradicts", "✗", "badge-neg"),
        ]:
            if markers_list:
                st.markdown(
                    f'<span style="font-size:0.78rem;color:#64748b">'
                    f'{icon} <em>{label}</em> (% = published frequency for this diagnosis):</span><br>'
                    + " ".join(
                        f'<span class="{css_class}">{m}</span>'
                        for m in markers_list
                    ),
                    unsafe_allow_html=True,
                )
        st.write("")

    with st.expander("All diagnoses scored"):
        all_disp = ddx_df[["display_name", "lit_score", "data_score",
                            "combined_score", "covered", "n_input"]].copy()
        all_disp["coverage"] = all_disp.apply(
            lambda r: f"{int(r['covered'])}/{int(r['n_input'])}", axis=1
        )
        all_disp = all_disp[["display_name", "lit_score", "data_score",
                              "combined_score", "coverage"]]
        all_disp.columns = ["Diagnosis", "Lit score (%)", "Data score (%)",
                             "Combined (rank)", "Lit coverage"]
        st.dataframe(all_disp, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Suggested additional tests")
    top_canon_r = top3["canonical_diagnosis"].tolist()
    lit_all_r   = load_literature_all()
    suggestions = _suggest_markers(pos_snap, neg_snap, top_canon_r, lit_all_r)
    if not suggestions:
        st.info("No discriminating markers found in the literature for the top differentials.")
    else:
        n1 = diag_names_r.get(top_canon_r[0], top_canon_r[0])
        n2 = diag_names_r.get(top_canon_r[1], top_canon_r[1]) if len(top_canon_r) > 1 else ""
        st.caption(f"Markers that best discriminate between {n1} and {n2}, based on literature.")
        sug_df = pd.DataFrame(suggestions)
        sug_df.columns = ["Marker", f"Expected in {n1}", f"Expected in {n2}", "Power"]
        sug_df["Power"] = sug_df["Power"].map(lambda x: f"{x:.0f}")
        st.dataframe(sug_df, use_container_width=True, hide_index=True)

    # ── Molecular pathology guidance ──────────────────────────────────────────
    st.divider()
    st.subheader("Molecular pathology guidance")

    # If a patient_id was passed, show their existing molecular result first
    if patient_id:
        mol_existing = load_molecular_tests(patient_id)
        morph_note   = load_morphology_note(patient_id)
        if not mol_existing.empty or morph_note:
            st.markdown("**Results already on file for this case:**")
            if morph_note:
                st.markdown(
                    f'<span style="background:#fef9c3;padding:3px 8px;border-radius:4px;'
                    f'font-size:0.85rem">Morphology: {morph_note}</span>',
                    unsafe_allow_html=True,
                )
            for _, mr in mol_existing.iterrows():
                pos_html = (
                    f'<span style="background:#dcfce7;padding:2px 8px;border-radius:4px;'
                    f'font-size:0.82rem;color:#166534">Positive: {mr["positive_result"]}</span> '
                    if mr.get("positive_result") else ""
                )
                neg_html = (
                    f'<span style="background:#fee2e2;padding:2px 8px;border-radius:4px;'
                    f'font-size:0.82rem;color:#991b1b">Not detected: {mr["negative_probes"]}</span>'
                    if mr.get("negative_probes") else ""
                )
                st.markdown(
                    f'<span style="font-size:0.82rem;color:#64748b">Test: {mr["test_type"]}</span> '
                    + pos_html + neg_html,
                    unsafe_allow_html=True,
                )
            st.write("")

    # For each top differential, show recommended molecular test from knowledge base
    ROLE_ICON = {"confirmatory": "🔬", "supportive": "🧪", "exclusion": "⊘"}
    ROLE_COLOR = {"confirmatory": "#0f766e", "supportive": "#4f46e5", "exclusion": "#dc2626"}

    any_mol_guidance = False
    for _, row in top3.iterrows():
        canon = row["canonical_diagnosis"]
        name  = row["display_name"]
        mol_kb = load_molecular_dx_markers(canon)
        if mol_kb.empty:
            continue
        any_mol_guidance = True
        st.markdown(f"**{name}**")
        for _, kr in mol_kb.iterrows():
            role  = kr["role"]
            icon  = ROLE_ICON.get(role, "")
            color = ROLE_COLOR.get(role, "#374151")
            pct_str = f" ({int(kr['prevalence_pct'])}% of cases)" if pd.notna(kr.get("prevalence_pct")) and kr.get("prevalence_pct") else ""
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:6px 12px;'
                f'margin:4px 0;border-radius:0 6px 6px 0;background:#f8fafc">'
                f'<span style="font-size:0.72rem;font-weight:700;color:{color};'
                f'text-transform:uppercase">{icon} {role}</span><br>'
                f'<span style="font-size:0.88rem;font-weight:600">{kr["alteration"]}</span>'
                f'<span style="font-size:0.8rem;color:#6b7280"> via {kr["test_type"]}{pct_str}</span><br>'
                f'<span style="font-size:0.78rem;color:#6b7280">{kr["note"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.write("")

    if not any_mol_guidance:
        st.info(
            "No molecular testing guidance available in the knowledge base for the top differentials. "
            "This may mean the diagnosis is confirmed on morphology and IHC alone."
        )


# ── Shared UI helpers ──────────────────────────────────────────────────────────

OUTCOME_COLORS = {
    "positive":  "#0f766e",
    "negative":  "#dc2626",
    "equivocal": "#d97706",
}

LMS_GROUPS = [
    "leiomyosarcoma_hg",
    "leiomyosarcoma_asmt",
    "uterine_leiomyosarcoma_hg",
    "uterine_leiomyosarcoma_asmt",
    "uterine_tumour_uncertain_malignant_potential",
]


def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = ""):
    sub_html = f'<p class="section-banner-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-banner">'
        f'<p class="section-banner-title">{title}</p>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def warning(text: str):
    st.markdown(
        f'<div class="warning-box">⚠ {text}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p style="font-size:1.3rem;font-weight:700;color:#0f766e;margin:0;">🔬 IHC Dashboard</p>'
        '<p style="font-size:0.75rem;color:#64748b;margin:2px 0 0;">MSc 2026 · University of London</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    diagnoses_df = load_diagnoses()
    main_diag    = diagnoses_df[diagnoses_df["include_in_analysis"] == 1]
    reserved_diag = diagnoses_df[diagnoses_df["include_in_analysis"] == 0]

    # Tabs that carry their own diagnosis filter inside the view
    _GLOBAL_TABS = {
        "Case Browser", "Fusion Browser", "Marker Browser",
        "Pipeline Runner", "Diagnostic Assistant", "Batch Import",
        "References",
    }

    tab_choice = st.radio(
        "View",
        [
            "Diagnosis Panel",
            "Literature Comparison",
            "Case Browser",
            "Fusion Browser",
            "Marker Browser",
            "Pipeline Runner",
            "Diagnostic Assistant",
            "Batch Import",
            "References",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Diagnosis selector -- only for views that need it
    selected_diag = None
    diag_meta     = None

    if tab_choice not in _GLOBAL_TABS:
        include_reserved = st.checkbox("Include reserved diagnoses", value=False)
        pool = pd.concat([main_diag, reserved_diag]) if include_reserved else main_diag
        selected_diag = st.selectbox(
            "Diagnosis",
            pool["canonical_diagnosis"].tolist(),
            format_func=lambda x: diagnoses_df.set_index("canonical_diagnosis")
                                               .loc[x, "display_name"],
        )
        diag_meta = diagnoses_df.set_index("canonical_diagnosis").loc[selected_diag]
        st.caption(f"n = {diag_meta['n_gt_cases']} cases")
        if diag_meta["notes"]:
            warning(diag_meta["notes"])


# ── View 1: Diagnosis Panel ────────────────────────────────────────────────────

if tab_choice == "Diagnosis Panel":
    disp = diag_meta["display_name"]
    section_header(f"Marker Panel: {disp}",
                   f"Pipeline-extracted IHC marker frequencies · n = {diag_meta['n_gt_cases']} cases")

    col1, col2, col3 = st.columns(3)
    freq = load_marker_frequencies(selected_diag, source="pipeline")

    n_pos_markers = freq[freq["outcome"] == "positive"]["marker"].nunique()
    n_neg_markers = freq[freq["outcome"] == "negative"]["marker"].nunique()
    n_equivocal   = freq[freq["outcome"] == "equivocal"]["n_cases"].sum()

    with col1:
        metric_card("Total cases", str(diag_meta["n_gt_cases"]))
    with col2:
        metric_card("Distinct markers tested (positive)", str(n_pos_markers))
    with col3:
        metric_card("Equivocal flags", str(int(n_equivocal)))

    st.divider()

    # Controls: row 1
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        outcome_filter = st.multiselect(
            "Show outcomes",
            ["positive", "negative", "equivocal"],
            default=["positive", "negative"],
        )
    with c2:
        top_n = st.slider("Top N markers", 5, 40, 20)
    with c3:
        show_gt  = st.checkbox("Overlay ground truth", value=False)
        show_ki67 = st.checkbox("Show Ki-67 index", value=True)

    # Controls: row 2: optional marker filter
    # Use the canonical RNOH whitelist intersected with what actually appears
    # for this diagnosis, so the dropdown only shows valid, relevant markers.
    markers_in_diag = set(freq["marker"].unique())
    canonical_in_diag = sorted(markers_in_diag & CANONICAL_MARKERS)
    marker_filter = st.multiselect(
        "Filter to specific markers (leave empty to show top N)",
        canonical_in_diag,
        placeholder="Type a marker name...",
    )

    # Apply filters
    df_plot = freq[freq["outcome"].isin(outcome_filter)].copy()
    if marker_filter:
        # Show only selected markers regardless of top_n
        df_plot = df_plot[df_plot["marker"].isin(marker_filter)]
        active_markers = set(marker_filter)
    else:
        # Fall back to top N by total n_cases
        active_markers = set(
            df_plot.groupby("marker")["n_cases"].sum()
            .nlargest(top_n).index.tolist()
        )
        df_plot = df_plot[df_plot["marker"].isin(active_markers)]

    # Build combined dataframe: pipeline + optional GT overlay
    OVERLAY_COLORS = {
        "Pipeline positive":  "#0f766e",
        "Pipeline negative":  "#dc2626",
        "GT positive":        "#5eead4",
        "GT negative":        "#fca5a5",
    }

    combined_rows = []
    for _, r in df_plot.iterrows():
        combined_rows.append({
            "marker":       r["marker"],
            "source_label": f"Pipeline {r['outcome']}",
            "pct":          r["pct"],
        })

    all_markers_overlay = set(df_plot["marker"].tolist())

    if show_gt:
        freq_gt = load_marker_frequencies(selected_diag, source="gt")
        df_gt = freq_gt[freq_gt["outcome"].isin(outcome_filter)].copy()
        if marker_filter:
            df_gt = df_gt[df_gt["marker"].isin(marker_filter)]
            all_markers_overlay = all_markers_overlay | set(marker_filter)
        else:
            top_gt_markers = (
                df_gt.groupby("marker")["n_cases"].sum()
                .nlargest(top_n).index.tolist()
            )
            all_markers_overlay = all_markers_overlay | set(top_gt_markers)
        for _, r in df_gt[df_gt["marker"].isin(all_markers_overlay)].iterrows():
            combined_rows.append({
                "marker":       r["marker"],
                "source_label": f"GT {r['outcome']}",
                "pct":          r["pct"],
            })

    combined_df = pd.DataFrame(combined_rows)
    combined_df = combined_df[combined_df["marker"].isin(all_markers_overlay)]

    n_shown = combined_df["marker"].nunique()
    title_suffix = " (pipeline + GT overlay)" if show_gt else " (pipeline predictions)"
    fig = px.bar(
        combined_df,
        x="pct",
        y="marker",
        color="source_label",
        orientation="h",
        barmode="group",
        color_discrete_map=OVERLAY_COLORS,
        labels={"pct": "% of cases", "marker": "Marker", "source_label": "Source"},
        title=f"Marker frequencies{title_suffix}: {disp} (n={diag_meta['n_gt_cases']})",
        height=max(400, n_shown * (40 if show_gt else 26)),
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        legend_title_text="Source",
        margin=dict(l=0, r=20, t=50, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    # Ki-67 section (toggle via checkbox above)
    if show_ki67:
        ki67_df = load_ki67(selected_diag)
        if not ki67_df.empty:
            st.divider()
            st.subheader(f"Ki-67 proliferation index: {disp}")
            st.caption(f"{len(ki67_df)} cases with Ki-67 extracted")
            fig_ki = px.histogram(
                ki67_df, x="ki67_pct", nbins=30,
                labels={"ki67_pct": "Ki-67 %"},
                title="Ki-67 distribution",
                color_discrete_sequence=["#6366f1"],
            )
            fig_ki.update_layout(
                plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                margin=dict(l=0, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_ki, use_container_width=True)

    # Subclassification section (LMS groups only)
    if selected_diag in LMS_GROUPS:
        st.divider()
        st.subheader("Subclassification breakdown")

        sub_dist_lms = load_subclass_distribution(selected_diag)
        sub_dist_clean = sub_dist_lms[sub_dist_lms["subclassification"] != "(none)"]

        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            metric_card("Total cases", str(diag_meta["n_gt_cases"]))
            st.write("")
            metric_card("Cases with subclassification", str(sub_dist_clean["n_cases"].sum()))
        with col_p2:
            if not sub_dist_clean.empty:
                fig_pie = px.pie(
                    sub_dist_clean,
                    names="subclassification",
                    values="n_cases",
                    title="Subclassification distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_pie.update_traces(textinfo="label+percent")
                fig_pie.update_layout(
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False,
                    height=280,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No subclassification data available.")

        st.subheader("Markers by subclassification")
        sub_opts = ["(all)"] + sub_dist_lms["subclassification"].tolist()
        c_sub1, c_sub2 = st.columns([1, 1])
        with c_sub1:
            sel_sub = st.selectbox("Subclassification", sub_opts, key="dp_lms_sub")
        with c_sub2:
            top_n_lms = st.slider("Top N markers", 5, 30, 15, key="dp_lms_n")
        out_lms = st.multiselect(
            "Outcomes", ["positive", "negative"],
            default=["positive", "negative"], key="dp_lms_oc",
        )

        df_sub_lms, n_sub_lms = load_lms_subclass_markers(selected_diag, sel_sub)
        if not df_sub_lms.empty:
            df_sub_lms = df_sub_lms[df_sub_lms["outcome"].isin(out_lms)]
            top_m_lms = df_sub_lms.groupby("marker")["n"].sum().nlargest(top_n_lms).index.tolist()
            df_sub_lms = df_sub_lms[df_sub_lms["marker"].isin(top_m_lms)]
            fig_sub_lms = px.bar(
                df_sub_lms, x="pct", y="marker", color="outcome",
                orientation="h",
                color_discrete_map=OUTCOME_COLORS,
                labels={"pct": "% of cases", "marker": "Marker", "outcome": "Outcome"},
                title=f"Marker frequencies: {disp} | {sel_sub} (n={n_sub_lms})",
                height=max(350, top_n_lms * 26),
            )
            fig_sub_lms.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=0, r=20, t=50, b=20),
                plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            )
            fig_sub_lms.update_xaxes(range=[0, 100], ticksuffix="%")
            st.plotly_chart(fig_sub_lms, use_container_width=True)

        else:
            st.info("No cases for this subclassification.")

    # ── Molecular pathology (fusions / rearrangements) ────────────────────────
    mol_pos_dp = load_molecular_positives(selected_diag)
    if not mol_pos_dp.empty:
        st.divider()
        st.subheader("Molecular findings: confirmed fusions & rearrangements")
        st.caption(
            f"Positive molecular results recorded for {disp} cases in the institutional dataset."
        )
        fig_mol = px.bar(
            mol_pos_dp,
            x="n_cases",
            y="positive_result",
            orientation="h",
            color_discrete_sequence=["#6366f1"],
            labels={"n_cases": "Cases confirmed", "positive_result": "Fusion / alteration"},
            title=f"Confirmed molecular findings: {disp}",
            height=max(280, len(mol_pos_dp) * 46),
        )
        fig_mol.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=0, r=20, t=50, b=20),
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
        )
        fig_mol.update_xaxes(tickformat="d")
        st.plotly_chart(fig_mol, use_container_width=True)


# ── View 2: Literature Comparison ─────────────────────────────────────────────

elif tab_choice == "Literature Comparison":
    disp = diag_meta["display_name"]
    section_header(f"Literature Comparison: {disp}",
                   "Institutional pipeline rates vs published expected IHC profiles")

    lit_df, total_cases = load_literature_comparison(selected_diag)

    if lit_df.empty:
        st.info("No literature profiles available for this diagnosis yet.")
    else:
        st.caption(
            f"Comparing pipeline predictions across {total_cases} cases "
            f"against published IHC profiles."
        )

        # Delta: compare tested_pct vs literature (fairer: both based on tested cases)
        # Fall back to our_pct if marker was never tested (tested_pct is None)
        lit_df["delta"] = lit_df.apply(
            lambda r: round((r["tested_pct"] if r["tested_pct"] is not None else 0) - r["lit_pct"], 1),
            axis=1,
        )
        lit_df["direction"] = lit_df.apply(
            lambda r: "Match" if r["tested_n"] > 0 and abs(r["delta"]) <= 15
            else ("Not tested" if r["tested_n"] == 0
                  else ("Above" if r["delta"] > 0 else "Below")),
            axis=1,
        )

        # Summary metrics (exclude "Not tested" from match/above/below counts)
        tested_rows = lit_df[lit_df["tested_n"] > 0]
        n_match    = (tested_rows["direction"] == "Match").sum()
        n_above    = (tested_rows["direction"] == "Above").sum()
        n_below    = (tested_rows["direction"] == "Below").sum()
        n_untested = (lit_df["tested_n"] == 0).sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Within 15% of literature", str(n_match))
        with c2:
            metric_card("Above expected", str(n_above))
        with c3:
            metric_card("Below expected", str(n_below))
        with c4:
            metric_card("Not tested in dataset", str(n_untested))

        st.caption(
            "Institutional % (of tested) = % of cases **where that marker was tested** "
            "that showed the expected outcome. Coverage = % of all cases where the marker "
            "appeared in the report. Delta compares institutional (tested) vs literature."
        )
        st.divider()

        # Grouped bar chart: three bars: Literature, Institutional (all cases), Institutional (tested only)
        plot_rows = []
        for _, row in lit_df.iterrows():
            label = f"{row['marker']} ({row['expected_outcome']})"
            plot_rows.append({
                "marker_label": label, "source": "Literature",
                "pct": row["lit_pct"],
                "hover": f"Literature: {row['lit_pct']}% (from {row['citation']})",
            })
            plot_rows.append({
                "marker_label": label, "source": "Institutional (all cases)",
                "pct": row["our_pct"],
                "hover": (f"{row['our_pct']}% of all {total_cases} cases "
                          f"({int(row['our_n'])} cases)"),
            })
            if row["tested_n"] > 0:
                plot_rows.append({
                    "marker_label": label, "source": "Institutional (of tested)",
                    "pct": row["tested_pct"],
                    "hover": (f"{row['tested_pct']}% of {int(row['tested_n'])} tested cases "
                              f"({row['coverage_pct']}% coverage)"),
                })
        plot_df = pd.DataFrame(plot_rows)

        fig = px.bar(
            plot_df,
            x="pct", y="marker_label",
            color="source",
            orientation="h",
            barmode="group",
            hover_data={"hover": True, "pct": False, "source": False, "marker_label": False},
            color_discrete_map={
                "Literature":                  "#94a3b8",
                "Institutional (all cases)":   "#5eead4",
                "Institutional (of tested)":   "#0f766e",
            },
            labels={"pct": "% of cases", "marker_label": "Marker (expected outcome)",
                    "source": "Source", "hover": ""},
            title=f"Institutional vs literature: {disp}",
            height=max(400, len(lit_df) * 55),
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=0, r=20, t=50, b=20),
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

        # Delta chart (vs tested pct)
        delta_df = lit_df[lit_df["tested_n"] > 0].copy()
        fig_delta = px.bar(
            delta_df.sort_values("delta"),
            x="delta", y="marker",
            color="direction",
            orientation="h",
            color_discrete_map={
                "Match": "#0f766e",
                "Above": "#d97706",
                "Below": "#dc2626",
            },
            labels={"delta": "Institutional (tested) minus Literature (%)", "marker": "Marker"},
            title="Deviation from literature: tested cases only (positive = higher than expected)",
            height=max(350, len(delta_df) * 30),
        )
        fig_delta.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_delta.add_vrect(x0=-15, x1=15, fillcolor="green", opacity=0.05,
                            annotation_text="±15% tolerance zone")
        fig_delta.update_layout(
            margin=dict(l=0, r=20, t=50, b=20),
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_delta, use_container_width=True)

        # Detailed table
        st.subheader("Detailed comparison table")
        table_df = lit_df.copy()
        table_df["tested_pct_display"] = table_df["tested_pct"].apply(
            lambda v: f"{v}%" if v is not None else "-"
        )
        table_df["coverage_display"] = table_df.apply(
            lambda r: f"{r['coverage_pct']}% ({int(r['tested_n'])}/{total_cases})", axis=1
        )
        display_cols = {
            "marker":               "Marker",
            "expected_outcome":     "Expected",
            "lit_pct":              "Literature %",
            "our_pct":              "Institutional % (all cases)",
            "tested_pct_display":   "Institutional % (of tested)",
            "coverage_display":     "Coverage",
            "delta":                "Delta",
            "our_n":                "n with expected outcome",
            "direction":            "Agreement",
            "citation":             "Source",
        }
        table = table_df[list(display_cols)].rename(columns=display_cols)

        def colour_agreement(val):
            if val == "Match":
                return "background-color: #dcfce7"
            elif val == "Above":
                return "background-color: #fef9c3"
            elif val == "Not tested":
                return "background-color: #f1f5f9; color: #94a3b8"
            else:
                return "background-color: #fee2e2"

        styled = table.style.applymap(colour_agreement, subset=["Agreement"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.caption(
            "Agreement: within ±15 percentage points of the literature figure (tested cases only). "
            "Literature sources: WHO Classification of Soft Tissue and Bone Tumours "
            "(5th ed. 2020); Fletcher CDM et al.; Hornick JL."
        )



# ── View 3: Case Browser (universal) ──────────────────────────────────────────

if tab_choice == "Case Browser":
    section_header("Case Browser", "Search and explore individual patient IHC reports")

    diag_names_cb = load_diagnoses()[["canonical_diagnosis", "display_name"]]
    diag_fmt_cb   = diag_names_cb.set_index("canonical_diagnosis")["display_name"].to_dict()

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        pid_cb = st.text_input("Search patient ID", placeholder="partial ID...", key="cb_pid")
    with c2:
        diag_opts_cb = ["(all)"] + diag_names_cb["canonical_diagnosis"].tolist()
        diag_cb = st.selectbox(
            "Diagnosis",
            diag_opts_cb,
            format_func=lambda x: diag_fmt_cb.get(x, x),
            key="cb_diag",
        )
    with c3:
        marker_cb = st.selectbox(
            "Marker present", ["(any)"] + sorted(CANONICAL_MARKERS), key="cb_marker"
        )
    with c4:
        outcome_cb = st.selectbox(
            "Outcome", ["positive", "negative"], key="cb_outcome",
            disabled=(marker_cb == "(any)"),
        )

    fusion_opts_cb = ["(any)"] + load_distinct_fusions()
    fusion_cb = st.selectbox(
        "Fusion / rearrangement present",
        fusion_opts_cb,
        key="cb_fusion",
    )

    cb_df = load_all_cases_global(
        pid_search    = pid_cb,
        diag_filter   = diag_cb,
        marker_name   = marker_cb   if marker_cb   != "(any)" else "",
        marker_outcome= outcome_cb,
        fusion_filter = fusion_cb   if fusion_cb   != "(any)" else "",
    )

    st.caption(f"{len(cb_df)} cases (max 500)")

    if cb_df.empty:
        st.info("No cases match the current filters.")
    else:
        display_cb = cb_df[["patient_id", "canonical_diagnosis", "has_gt"]].copy()
        display_cb["canonical_diagnosis"] = display_cb["canonical_diagnosis"].map(
            lambda x: diag_fmt_cb.get(x, x)
        )
        display_cb.columns = ["Patient ID", "Diagnosis", "Has GT"]
        st.dataframe(display_cb, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Case detail")
        sel_pid_cb  = st.selectbox("Select patient", cb_df["patient_id"].tolist(), key="cb_sel")
        row_cb      = cb_df[cb_df["patient_id"] == sel_pid_cb].iloc[0]
        markers_cb  = load_case_markers(sel_pid_cb)

        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.markdown(f"**Patient ID:** `{sel_pid_cb}`")
            st.markdown(
                f"**Diagnosis:** {diag_fmt_cb.get(row_cb['canonical_diagnosis'], row_cb['canonical_diagnosis'])}"
            )
            for src_label, src_key in [("Pipeline predictions", "pipeline"), ("Ground truth", "gt")]:
                st.write("")
                st.markdown(f"**{src_label}:**")
                for oc, bg in [("positive", "#dcfce7"), ("negative", "#fee2e2"), ("equivocal", "#fef9c3")]:
                    mm = markers_cb[
                        (markers_cb["source"] == src_key) & (markers_cb["outcome"] == oc)
                    ]["marker"].tolist()
                    if mm:
                        st.markdown(
                            f"{oc.capitalize()}: " +
                            " ".join(
                                f'<span style="background:{bg};padding:2px 6px;'
                                f'border-radius:4px;font-size:0.82rem">{m}</span>'
                                for m in mm
                            ),
                            unsafe_allow_html=True,
                        )

            # Morphology notes (e.g. PAS-D crystals for ASPS)
            morph_note_cb = load_morphology_note(sel_pid_cb)
            if morph_note_cb:
                st.write("")
                st.markdown("**Morphology:**")
                st.markdown(
                    f'<span style="background:#fef9c3;padding:3px 8px;border-radius:4px;'
                    f'font-size:0.85rem">{morph_note_cb}</span>',
                    unsafe_allow_html=True,
                )

            # Molecular pathology results
            mol_cb = load_molecular_tests(sel_pid_cb)
            if not mol_cb.empty:
                st.write("")
                st.markdown("**Molecular pathology:**")
                for _, mrow in mol_cb.iterrows():
                    parts = [f"Test: {mrow['test_type']}"]
                    if mrow.get("positive_result"):
                        pos_label = mrow["positive_result"]
                        # Show raw finding in parentheses if it differs (e.g. was split from a multi-finding cell)
                        raw = mrow.get("raw_finding") or ""
                        if raw and raw != pos_label:
                            pos_label = f"{pos_label} <span style='color:#92400e'>(raw: {raw})</span>"
                        parts.append(
                            f'<span style="background:#dcfce7;padding:2px 8px;border-radius:4px;'
                            f'font-size:0.82rem;color:#166534">Positive: {pos_label}</span>'
                        )
                    if mrow.get("negative_probes"):
                        parts.append(
                            f'<span style="background:#fee2e2;padding:2px 8px;border-radius:4px;'
                            f'font-size:0.82rem;color:#991b1b">Not detected: {mrow["negative_probes"]}</span>'
                        )
                    st.markdown(
                        f'<span style="font-size:0.82rem;color:#64748b">{parts[0]}</span> '
                        + " ".join(parts[1:]),
                        unsafe_allow_html=True,
                    )

            # Send to Pipeline Runner / Diagnostic Assistant
            pred_pos_cb = markers_cb[
                (markers_cb["source"] == "pipeline") & (markers_cb["outcome"] == "positive")
            ]["marker"].tolist()
            pred_neg_cb = markers_cb[
                (markers_cb["source"] == "pipeline") & (markers_cb["outcome"] == "negative")
            ]["marker"].tolist()
            st.write("")
            if st.button("Get differentials for this case", key="cb_ddx_btn"):
                st.session_state["da_pos"] = pred_pos_cb
                st.session_state["da_neg"] = pred_neg_cb
                st.session_state["da_prefill"] = True
                st.session_state["da_patient_id"] = sel_pid_cb
                st.info("Markers loaded into Diagnostic Assistant. Switch to that tab to run.")

        with col_r:
            st.markdown("**IHC report text:**")
            st.text_area(
                "", value=row_cb.get("ihc_text", "") or "(no IHC text recorded)", height=300,
                disabled=True, label_visibility="collapsed", key=f"cb_text_{sel_pid_cb}",
            )


# ── View 4b: Fusion Browser ───────────────────────────────────────────────────

if tab_choice == "Fusion Browser":
    section_header("Fusion Browser",
                   "Molecular findings: confirmed diagnoses only")

    con_fb        = get_connection()
    diag_names_fb = load_diagnoses().set_index("canonical_diagnosis")["display_name"].to_dict()

    # Load all confirmed molecular data once (positives + negatives)
    _mol_pos = pd.read_sql(
        """
        SELECT mt.patient_id, c.canonical_diagnosis, mt.test_type,
               mt.positive_result, mt.raw_finding, 'positive' AS outcome
        FROM molecular_tests mt
        JOIN cases c ON c.patient_id = mt.patient_id
        JOIN diagnoses d ON d.canonical_diagnosis = c.canonical_diagnosis
        WHERE d.is_confirmed = 1
          AND mt.positive_result IS NOT NULL AND mt.positive_result != ''
        """,
        con_fb,
    )
    _mol_neg = pd.read_sql(
        """
        SELECT mt.patient_id, c.canonical_diagnosis, mt.test_type,
               mt.negative_probes AS positive_result, NULL AS raw_finding, 'negative' AS outcome
        FROM molecular_tests mt
        JOIN cases c ON c.patient_id = mt.patient_id
        JOIN diagnoses d ON d.canonical_diagnosis = c.canonical_diagnosis
        WHERE d.is_confirmed = 1
          AND mt.negative_probes IS NOT NULL AND mt.negative_probes != ''
        """,
        con_fb,
    )
    _mol_all = pd.concat([_mol_pos, _mol_neg], ignore_index=True)
    _mol_all["gene"] = _mol_all["positive_result"].apply(normalize_molecular)
    _mol_all["detail_label"] = _mol_all["positive_result"].apply(get_detail_label)

    # For Browse by Gene: expand fusions so each :: partner is also searchable.
    # e.g. PAX3::FOXO1 creates a row with gene="PAX3" AND gene="FOXO1".
    def _expand_genes(df):
        extra = []
        for _, row in df.iterrows():
            pr = str(row["positive_result"]) if row["positive_result"] else ""
            if "::" in pr:
                left = pr.split("::")[0].strip().split()[0].rstrip(",.;()")
                right = pr.split("::", 1)[1].strip().split()[0].rstrip(",.;()")
                both = {left, right}
                # Add a row for whichever gene normalize_molecular didn't pick
                picked = row["gene"]
                for alt in both - {picked}:
                    if alt:
                        r = row.copy()
                        r["gene"] = alt
                        extra.append(r)
        if extra:
            return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        return df

    _mol_expanded = _expand_genes(_mol_all)

    if _mol_all.empty:
        st.info("No confirmed molecular results in the dataset yet.")
    else:
        _subtab_gene, _subtab_diag = st.tabs(["Browse by Gene", "By Diagnosis"])

        # ── Sub-tab 1: Browse by Gene (Marker-Browser style) ──────────────────
        with _subtab_gene:
            _all_genes = sorted(_mol_expanded["gene"].dropna().unique())
            _oc_opts_g = ["positive", "negative", "both"]

            col_g1, col_g2 = st.columns([3, 1])
            with col_g1:
                sel_gene_fb = st.selectbox("Gene / alteration", _all_genes, key="fb_gene")
            with col_g2:
                sel_oc_gene = st.radio("Outcome", _oc_opts_g, horizontal=True, key="fb_gene_oc")

            _gene_df = _mol_expanded[_mol_expanded["gene"] == sel_gene_fb].copy()
            if sel_oc_gene != "both":
                _gene_df = _gene_df[_gene_df["outcome"] == sel_oc_gene]

            if _gene_df.empty:
                st.info(f"No {sel_oc_gene} results for **{sel_gene_fb}** in the dataset.")
            else:
                # Count distinct cases per diagnosis, split by outcome
                _gene_agg = (
                    _gene_df.groupby(["canonical_diagnosis", "outcome"])["patient_id"]
                    .nunique().reset_index()
                    .rename(columns={"patient_id": "n_cases"})
                )
                _gene_agg["display_name"] = _gene_agg["canonical_diagnosis"].map(
                    lambda x: diag_names_fb.get(x, x)
                )
                _gene_agg = _gene_agg.sort_values("n_cases", ascending=False)

                # Metrics
                n_total_gene = int(_gene_agg["n_cases"].sum())
                n_diags_gene = _gene_agg["canonical_diagnosis"].nunique()
                cg1, cg2 = st.columns(2)
                with cg1:
                    metric_card("Total cases", str(n_total_gene))
                with cg2:
                    metric_card("Diagnoses", str(n_diags_gene))

                # Sort by total cases descending; truncate names for y-axis readability
                _gene_sorted = (
                    _gene_agg.groupby(["canonical_diagnosis", "display_name"])["n_cases"].sum()
                    .reset_index().sort_values("n_cases", ascending=False)
                    .reset_index(drop=True)
                )
                _gene_sorted["short_name"] = _gene_sorted["display_name"].str[:30]
                _short_map = dict(zip(_gene_sorted["canonical_diagnosis"], _gene_sorted["short_name"]))
                _gene_agg = _gene_agg.copy()
                _gene_agg["short_name"] = _gene_agg["canonical_diagnosis"].map(_short_map)
                # bottom-to-top order: smallest case count at bottom
                _cat_order = _gene_sorted["short_name"].tolist()[::-1]

                fig_gene = px.bar(
                    _gene_agg.sort_values("n_cases"),
                    x="n_cases",
                    y="short_name",
                    color="short_name" if sel_oc_gene != "both" else "outcome",
                    orientation="h",
                    barmode="group",
                    color_discrete_map={"positive": "#3b82f6", "negative": "#f87171"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"n_cases": "Cases", "short_name": "Diagnosis", "outcome": "Outcome"},
                    title=f"{sel_gene_fb}: cases by diagnosis",
                    height=max(320, n_diags_gene * 52),
                )
                fig_gene.update_layout(
                    yaxis={"categoryorder": "array", "categoryarray": _cat_order},
                    margin=dict(l=0, r=20, t=50, b=20),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                    showlegend=False,
                )
                st.plotly_chart(fig_gene, use_container_width=True)

                # Full name reference table
                st.dataframe(
                    _gene_sorted[["display_name", "n_cases"]].rename(
                        columns={"display_name": "Diagnosis", "n_cases": "Cases"}
                    ),
                    use_container_width=True, hide_index=True,
                )

                # Case table
                st.divider()
                _gene_cases = _gene_df.copy()
                _gene_cases["Diagnosis"] = _gene_cases["canonical_diagnosis"].map(
                    lambda x: diag_names_fb.get(x, x)
                )
                st.caption(f"{_gene_cases['patient_id'].nunique()} case(s) shown:")
                st.dataframe(
                    _gene_cases[["patient_id", "Diagnosis", "test_type", "detail_label", "outcome"]]
                    .rename(columns={
                        "patient_id":   "Patient ID",
                        "test_type":    "Test type",
                        "detail_label": "Specific finding",
                        "outcome":      "Outcome",
                    })
                    .sort_values(["Diagnosis", "Patient ID"]),
                    use_container_width=True, hide_index=True,
                )

        # ── Sub-tab 2: By Diagnosis ────────────────────────────────────────────
        with _subtab_diag:
            _diags_with_mol = sorted(
                _mol_all["canonical_diagnosis"].unique(),
                key=lambda x: diag_names_fb.get(x, x),
            )
            _diag_opts = [diag_names_fb.get(d, d) for d in _diags_with_mol]
            _diag_map  = dict(zip(_diag_opts, _diags_with_mol))

            col_d1, col_d2, col_d3 = st.columns([3, 1, 1])
            with col_d1:
                sel_diag_fb = st.selectbox("Diagnosis", _diag_opts, key="fb_diag")
            with col_d2:
                sel_oc_diag = st.radio("Outcome", _oc_opts_g, horizontal=True, key="fb_diag_oc")
            with col_d3:
                show_detail = st.toggle("Details", value=False, key="fb_detail")

            sel_canon_fb = _diag_map[sel_diag_fb]
            _dx_mol = _mol_all[_mol_all["canonical_diagnosis"] == sel_canon_fb].copy()
            if sel_oc_diag != "both":
                _dx_mol = _dx_mol[_dx_mol["outcome"] == sel_oc_diag]

            if _dx_mol.empty:
                st.info(f"No {sel_oc_diag} molecular results for {sel_diag_fb}.")
            else:
                n_cases_dx  = _dx_mol["patient_id"].nunique()
                n_genes_dx  = _dx_mol["gene"].nunique()
                cd1, cd2 = st.columns(2)
                with cd1:
                    metric_card("Cases with molecular data", str(n_cases_dx))
                with cd2:
                    metric_card("Distinct genes", str(n_genes_dx))

                st.divider()

                if show_detail:
                    # Specific variants, colour-coded by gene
                    _detail_agg = (
                        _dx_mol.groupby(["gene", "detail_label", "outcome"])["patient_id"]
                        .nunique().reset_index()
                        .rename(columns={"patient_id": "n_cases"})
                        .sort_values("n_cases")
                    )
                    _color_col = "outcome" if sel_oc_diag == "both" else "gene"
                    fig_dx = px.bar(
                        _detail_agg,
                        x="n_cases",
                        y="detail_label",
                        color=_color_col,
                        orientation="h",
                        barmode="group",
                        color_discrete_map={"positive": "#3b82f6", "negative": "#f87171"},
                        labels={"n_cases": "Cases", "detail_label": "Specific finding",
                                "gene": "Gene", "outcome": "Outcome"},
                        title=f"Molecular findings (detailed): {sel_diag_fb}",
                        height=max(320, len(_detail_agg) * 36),
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                else:
                    # One bar per specific finding (full fusion label)
                    _grouped_agg = (
                        _dx_mol.groupby(["detail_label", "outcome"])["patient_id"]
                        .nunique().reset_index()
                        .rename(columns={"patient_id": "n_cases"})
                        .sort_values("n_cases")
                    )
                    _single_color = (
                        "#f87171" if sel_oc_diag == "negative" else "#3b82f6"
                    )
                    fig_dx = px.bar(
                        _grouped_agg,
                        x="n_cases",
                        y="detail_label",
                        color="outcome" if sel_oc_diag == "both" else None,
                        orientation="h",
                        barmode="group",
                        color_discrete_map={"positive": "#3b82f6", "negative": "#f87171"},
                        color_discrete_sequence=[_single_color],
                        labels={"n_cases": "Cases", "detail_label": "Finding", "outcome": "Outcome"},
                        title=f"Molecular findings: {sel_diag_fb}",
                        height=max(320, _grouped_agg["detail_label"].nunique() * 40),
                    )

                fig_dx.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(l=0, r=20, t=50, b=20),
                    plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
                    showlegend=(show_detail or sel_oc_diag == "both"),
                )
                fig_dx.update_xaxes(tickformat="d")
                st.plotly_chart(fig_dx, use_container_width=True)

                st.divider()

                # Per-gene drill-down
                st.subheader("Case list")
                _gene_opts_dx = sorted(_dx_mol["gene"].unique())
                sel_gene_dx = st.selectbox("Filter by gene", ["(all)"] + _gene_opts_dx, key="fb_dx_gene")

                _cases_dx = _dx_mol.copy()
                if sel_gene_dx != "(all)":
                    _cases_dx = _cases_dx[_cases_dx["gene"] == sel_gene_dx]

                st.caption(f"{_cases_dx['patient_id'].nunique()} case(s):")
                st.dataframe(
                    _cases_dx[["patient_id", "test_type", "detail_label", "outcome"]]
                    .rename(columns={
                        "patient_id":   "Patient ID",
                        "test_type":    "Test type",
                        "detail_label": "Finding",
                        "outcome":      "Outcome",
                    })
                    .sort_values("Patient ID"),
                    use_container_width=True, hide_index=True,
                )

                # Jump to Diagnostic Assistant
                jump_pid = st.selectbox(
                    "Open case in Diagnostic Assistant",
                    ["(select)"] + sorted(_cases_dx["patient_id"].unique().tolist()),
                    key="fb_jump",
                )
                if jump_pid != "(select)":
                    case_markers_fb = load_case_markers(jump_pid)
                    pred_pos_fb = case_markers_fb[
                        (case_markers_fb["source"] == "pipeline") &
                        (case_markers_fb["outcome"] == "positive")
                    ]["marker"].tolist()
                    pred_neg_fb = case_markers_fb[
                        (case_markers_fb["source"] == "pipeline") &
                        (case_markers_fb["outcome"] == "negative")
                    ]["marker"].tolist()
                    if st.button("Load into Diagnostic Assistant", key="fb_load_btn"):
                        st.session_state["da_pos"]        = pred_pos_fb
                        st.session_state["da_neg"]        = pred_neg_fb
                        st.session_state["da_prefill"]    = True
                        st.session_state["da_patient_id"] = jump_pid
                        st.info(f"Markers for {jump_pid} loaded. Switch to Diagnostic Assistant to run.")


# ── View 5: Marker Browser ─────────────────────────────────────────────────────

if tab_choice == "Marker Browser":
    section_header("Marker Browser", "Cross-diagnosis marker positivity rates at a glance")
    st.caption("Select a marker to see how often it appears as positive or negative across diagnoses.")

    diag_names_mb  = load_diagnoses()[["canonical_diagnosis", "display_name"]]
    diag_fmt_mb    = diag_names_mb.set_index("canonical_diagnosis")["display_name"].to_dict()

    c1, c2 = st.columns([2, 1])
    with c1:
        sel_marker_mb = st.selectbox(
            "Marker", sorted(CANONICAL_MARKERS), key="mb_marker"
        )
    with c2:
        mb_outcome = st.radio(
            "Show outcome", ["positive", "negative", "both"], horizontal=True, key="mb_oc"
        )

    chart_df = load_marker_diag_chart(sel_marker_mb)
    lit_df_mb = load_literature_all()

    if chart_df.empty:
        st.info(f"No pipeline data found for **{sel_marker_mb}**.")
    else:
        chart_df["display_name"] = chart_df["canonical_diagnosis"].map(
            lambda x: diag_fmt_mb.get(x, x)
        )

        if mb_outcome != "both":
            plot_df = chart_df[chart_df["outcome"] == mb_outcome].copy()
            plot_df = plot_df.sort_values("pct", ascending=True)
            bar_color = "#0f766e" if mb_outcome == "positive" else "#dc2626"
            fig_mb = px.bar(
                plot_df,
                x="pct",
                y="display_name",
                orientation="h",
                color_discrete_sequence=[bar_color],
                labels={"pct": "% of cases", "display_name": ""},
                title=f"{sel_marker_mb} ({mb_outcome}) frequency by diagnosis",
            )
            fig_mb.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=max(300, 28 * plot_df["display_name"].nunique()),
                margin=dict(l=0, r=20, t=40, b=20),
                yaxis_title="",
                plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            )
        else:
            # "Both" mode: sort diagnoses by their positive pct (descending)
            # so the most positive diagnoses appear at the top of the chart.
            pos_pcts = (
                chart_df[chart_df["outcome"] == "positive"]
                .set_index("display_name")["pct"]
            )
            all_diags = chart_df["display_name"].unique().tolist()
            # Order: highest positive pct at top (ascending for horizontal bar = bottom)
            y_order = sorted(
                all_diags,
                key=lambda d: pos_pcts.get(d, 0),
                reverse=False,  # ascending so highest is at the top of chart
            )
            plot_df = chart_df.copy()
            color_map = {"positive": "#0f766e", "negative": "#dc2626"}
            fig_mb = px.bar(
                plot_df,
                x="pct",
                y="display_name",
                color="outcome",
                color_discrete_map=color_map,
                orientation="h",
                barmode="group",
                labels={"pct": "% of cases", "display_name": "", "outcome": "Outcome"},
                title=f"{sel_marker_mb}: positive vs negative rate by diagnosis",
            )
            fig_mb.update_layout(
                yaxis={"categoryorder": "array", "categoryarray": y_order},
                height=max(350, 40 * len(all_diags)),
                margin=dict(l=0, r=20, t=50, b=20),
                yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            )
        st.plotly_chart(fig_mb, use_container_width=True)

        # Literature comparison
        lit_row = lit_df_mb[lit_df_mb["marker"] == sel_marker_mb]
        if not lit_row.empty:
            st.divider()
            st.subheader("Literature profile for this marker")
            lit_disp = lit_row[["canonical_diagnosis", "expected_outcome", "typical_frequency_pct"]].copy()
            lit_disp["canonical_diagnosis"] = lit_disp["canonical_diagnosis"].map(
                lambda x: diag_fmt_mb.get(x, x)
            )
            lit_disp.columns = ["Diagnosis", "Expected outcome", "Literature frequency (%)"]
            st.dataframe(lit_disp, use_container_width=True, hide_index=True)

        # Raw numbers table
        st.divider()
        st.subheader("Case counts")
        raw_disp = chart_df.copy()
        raw_disp["display_name"] = raw_disp["canonical_diagnosis"].map(
            lambda x: diag_fmt_mb.get(x, x)
        )
        raw_disp = raw_disp[["display_name", "outcome", "n", "total", "pct"]].copy()
        raw_disp.columns = ["Diagnosis", "Outcome", "Cases (marker)", "Total cases", "%"]
        st.dataframe(raw_disp, use_container_width=True, hide_index=True)


# ── View 7: Pipeline Runner ────────────────────────────────────────────────────

if tab_choice == "Pipeline Runner":
    import importlib.util as _ilu
    from pathlib import Path as _Path

    section_header("Pipeline Runner", "Run the hybrid NLP extraction pipeline on new report text")

    @st.cache_resource(show_spinner=False)
    def _load_rule_module():
        spec = _ilu.spec_from_file_location("rule_pipeline", "03_rule_based_pipeline.py")
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @st.cache_resource(show_spinner=False)
    def _load_bert():
        """Load ClinicalBERT checkpoint. Returns (model, tokenizer, device, model_path) or None."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for candidate in ["models/clinicalbert_ihc_full", "models/clinicalbert_ihc"]:
                p = _Path(candidate)
                if p.exists() and any(p.iterdir()):
                    tok   = AutoTokenizer.from_pretrained(str(p))
                    model = AutoModelForTokenClassification.from_pretrained(str(p)).to(device)
                    model.eval()
                    return model, tok, device, candidate
        except Exception:
            pass
        return None

    # Load both on first render (BERT loading may take 20-30 s, shown as spinner)
    with st.spinner("Loading pipeline (first run only)..."):
        _bert_bundle = _load_bert()
        _rule_mod    = _load_rule_module()

    _bert_available = _bert_bundle is not None
    if _bert_available:
        st.caption(
            f"Hybrid pipeline active (ClinicalBERT + rule-based). "
            f"Model: `{_bert_bundle[3]}`"
        )
    else:
        st.caption(
            "ClinicalBERT checkpoint not found: running rule-based extraction only. "
            "Run `python 04_clinicalbert.py --train --full` to generate the model."
        )

    ihc_input = st.text_area(
        "IHC report text",
        height=200,
        placeholder="Paste the immunohistochemistry section of the report here...",
        key="pr_text",
    )

    run_btn = st.button("Run pipeline", type="primary", key="pr_run")

    if run_btn and ihc_input.strip():
        from marker_utils import normalise_marker

        rule_mod = _rule_mod
        raw_results = rule_mod.extract_from_text(ihc_input)

        if _bert_available:
            # Full hybrid: BERT-primary + rule supplement
            import torch
            bert_mod, tok, dev, _ = _bert_bundle

            # Import predict_text from 04_clinicalbert without running main()
            _bert_spec = _ilu.spec_from_file_location("bert_pipeline", "04_clinicalbert.py")
            _bert_mod  = _ilu.module_from_spec(_bert_spec)
            _bert_spec.loader.exec_module(_bert_mod)

            bert_pos_raw, bert_neg_raw = _bert_mod.predict_text(bert_mod.to(dev), tok, ihc_input)

            # combine() logic from 05_hybrid_pipeline
            from marker_utils import normalise_marker_list, strip_ki67

            rule_pos_raw = {r["marker"] for r in raw_results
                            if r["outcome"] in ("positive", "focal_positive")}
            rule_neg_raw = {r["marker"] for r in raw_results if r["outcome"] == "negative"}

            rp = set(normalise_marker_list(strip_ki67(list(rule_pos_raw))))
            rn = set(normalise_marker_list(strip_ki67(list(rule_neg_raw))))
            bp = set(normalise_marker_list(strip_ki67(list(bert_pos_raw))))
            bn = set(normalise_marker_list(strip_ki67(list(bert_neg_raw))))

            combined_pos = bp | (rp - bn)
            combined_neg = bn | (rn - bp)
            combined_pos -= combined_neg  # safety: resolve any remaining conflicts

            init_pos = sorted(m for m in combined_pos if m)
            init_neg = sorted(m for m in combined_neg if m)
            st.session_state["pr_mode"] = "hybrid"
        else:
            # Rule-based fallback
            pred: dict[str, str] = {}
            for r in raw_results:
                norm = normalise_marker(r["marker"])
                if norm is None:
                    continue
                oc = "positive" if r["outcome"] in ("positive", "focal_positive") else "negative"
                if norm not in pred or oc == "positive":
                    pred[norm] = oc
            init_pos = sorted(m for m, o in pred.items() if o == "positive")
            init_neg = sorted(m for m, o in pred.items() if o == "negative")
            st.session_state["pr_mode"] = "rule"

        st.session_state["pr_init_pos"] = init_pos
        st.session_state["pr_init_neg"] = init_neg
        st.session_state["pr_ran"] = True
        st.session_state["pr_run_id"] = st.session_state.get("pr_run_id", 0) + 1

    if st.session_state.get("pr_ran"):
        st.divider()
        mode_label = "Hybrid (ClinicalBERT + rules)" if st.session_state.get("pr_mode") == "hybrid" else "Rule-based only"
        st.subheader(f"Extracted markers (editable): {mode_label}")

        all_canon = sorted(CANONICAL_MARKERS)
        init_pos  = st.session_state.get("pr_init_pos", [])
        init_neg  = st.session_state.get("pr_init_neg", [])
        _run_id   = st.session_state.get("pr_run_id", 0)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            final_pos = st.multiselect(
                "Positive", all_canon, default=init_pos, key=f"pr_final_pos_{_run_id}"
            )
        with col_e2:
            final_neg = st.multiselect(
                "Negative", all_canon, default=init_neg, key=f"pr_final_neg_{_run_id}"
            )

        st.divider()
        st.subheader("Save to database (optional)")

        diag_names_pr = load_diagnoses()[["canonical_diagnosis", "display_name"]]
        diag_fmt_pr   = diag_names_pr.set_index("canonical_diagnosis")["display_name"].to_dict()

        c1, c2, c3 = st.columns(3)
        with c1:
            save_pid = st.text_input("Patient ID", key="pr_save_pid", placeholder="e.g. 24CH01234")
        with c2:
            save_raw_diag = st.text_input("Raw diagnosis", key="pr_save_raw",
                                          placeholder="as on the report")
        with c3:
            save_canon = st.selectbox(
                "Canonical diagnosis",
                diag_names_pr["canonical_diagnosis"].tolist(),
                format_func=lambda x: diag_fmt_pr.get(x, x),
                key="pr_save_canon",
            )

        save_btn = st.button("Save confirmed output to database", key="pr_save_btn",
                             disabled=not save_pid.strip())
        if save_btn:
            if not save_pid.strip():
                st.error("Patient ID is required before saving.")
            else:
                _write_to_db(
                    patient_id = save_pid.strip(),
                    raw_diag   = save_raw_diag.strip() or save_pid.strip(),
                    canon_diag = save_canon,
                    ihc_text   = st.session_state.get("pr_text", ""),
                    pred_pos   = final_pos,
                    pred_neg   = final_neg,
                )
                st.success(f"Saved {save_pid.strip()} with {len(final_pos)} positive and "
                           f"{len(final_neg)} negative markers.")

        # Inline differential diagnosis
        st.divider()
        st.subheader("Differential diagnosis")
        run_ddx_pr = st.button(
            "Get differentials from these markers",
            key="pr_ddx_btn",
            disabled=(not final_pos and not final_neg),
        )
        if run_ddx_pr:
            with st.spinner("Scoring diagnoses..."):
                ddx_pr = _compute_ddx(final_pos, final_neg)
            st.session_state["pr_ddx"] = ddx_pr
            st.session_state["pr_ddx_pos"] = final_pos[:]
            st.session_state["pr_ddx_neg"] = final_neg[:]

        if "pr_ddx" in st.session_state and not st.session_state["pr_ddx"].empty:
            _render_ddx_results(
                st.session_state["pr_ddx"],
                st.session_state.get("pr_ddx_pos", []),
                st.session_state.get("pr_ddx_neg", []),
                key_prefix="pr",
            )



# ── View 8: Diagnostic Assistant ──────────────────────────────────────────────

if tab_choice == "Diagnostic Assistant":
    section_header("Diagnostic Assistant",
                   "Score a marker panel against all diagnoses and identify the best differential")
    st.caption(
        "Enter the IHC markers for a case. The assistant scores each diagnosis against "
        "published literature and our institutional dataset, then suggests additional "
        "markers to help discriminate between the top differentials. "
        "Markers can also be pre-loaded from the Case Browser."
    )

    all_canon_da = sorted(CANONICAL_MARKERS)

    # Respect markers pre-loaded from Case Browser
    _prefill_pos = st.session_state.pop("da_pos", []) if st.session_state.get("da_prefill") else []
    _prefill_neg = st.session_state.pop("da_neg", []) if st.session_state.get("da_prefill") else []
    if st.session_state.get("da_prefill"):
        st.session_state["da_prefill"] = False

    c1, c2 = st.columns(2)
    with c1:
        da_pos = st.multiselect("Positive markers", all_canon_da,
                                default=_prefill_pos or [], key="da_pos_ms")
    with c2:
        da_neg = st.multiselect("Negative markers", all_canon_da,
                                default=_prefill_neg or [], key="da_neg_ms")

    overlap = set(da_pos) & set(da_neg)
    if overlap:
        st.warning(f"These markers appear in both lists: {', '.join(sorted(overlap))}. "
                   "Remove the conflict before running.")

    run_ddx = st.button(
        "Get differentials",
        type="primary",
        key="da_run",
        disabled=bool(overlap) or (not da_pos and not da_neg),
    )

    if run_ddx:
        with st.spinner("Scoring diagnoses..."):
            ddx_df_da = _compute_ddx(da_pos, da_neg)
        st.session_state["da_results"] = ddx_df_da
        st.session_state["da_pos_snap"] = da_pos[:]
        st.session_state["da_neg_snap"] = da_neg[:]

    if "da_results" in st.session_state and not st.session_state["da_results"].empty:
        st.divider()
        _render_ddx_results(
            st.session_state["da_results"],
            st.session_state.get("da_pos_snap", []),
            st.session_state.get("da_neg_snap", []),
            key_prefix="da",
            patient_id=st.session_state.get("da_patient_id"),
        )


# ── View 9: Batch Import ───────────────────────────────────────────────────────

if tab_choice == "Batch Import":
    import importlib.util as _ilu2

    section_header("Batch Import", "Import multiple IHC reports from an Excel or CSV file")
    st.caption(
        "Upload a CSV with patient reports. The rule-based pipeline runs on each row. "
        "Review the output, accept or skip individual cases, then save and optionally "
        "get differential diagnoses for the accepted cases."
    )

    @st.cache_resource
    def _load_rule_module_bi():
        spec = _ilu2.spec_from_file_location("rule_pipeline_bi", "03_rule_based_pipeline.py")
        mod  = _ilu2.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    st.markdown("**Required CSV columns:** `patient_id`, `ihc_text` (optional: `diagnosis`)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="bi_file")

    if uploaded:
        try:
            raw_csv = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            raw_csv = None

        if raw_csv is not None:
            cols = raw_csv.columns.tolist()

            c1, c2, c3 = st.columns(3)
            with c1:
                pid_col = st.selectbox("Patient ID column", cols, key="bi_pid_col",
                                       index=cols.index("patient_id") if "patient_id" in cols else 0)
            with c2:
                text_col = st.selectbox("IHC text column", cols, key="bi_text_col",
                                        index=cols.index("ihc_text") if "ihc_text" in cols else 0)
            with c3:
                diag_col_opts = ["(none)"] + cols
                diag_col = st.selectbox("Diagnosis column (optional)", diag_col_opts,
                                        key="bi_diag_col",
                                        index=diag_col_opts.index("diagnosis")
                                              if "diagnosis" in diag_col_opts else 0)

            n_rows = len(raw_csv)
            st.caption(f"{n_rows} rows detected.")

            run_batch = st.button("Run pipeline on all rows", type="primary", key="bi_run",
                                  disabled=(n_rows > 2000))
            if n_rows > 2000:
                st.warning("Batch limited to 2000 rows. Please split your file.")

            if run_batch:
                from marker_utils import normalise_marker
                rule_mod_bi = _load_rule_module_bi()

                results_bi = []
                prog = st.progress(0)
                for i, row in raw_csv.iterrows():
                    pid  = str(row[pid_col]).strip()
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    diag = str(row[diag_col]).strip() if diag_col != "(none)" and pd.notna(row.get(diag_col)) else ""

                    raw_res = rule_mod_bi.extract_from_text(text)
                    pred: dict[str, str] = {}
                    for r in raw_res:
                        norm = normalise_marker(r["marker"])
                        if norm is None:
                            continue
                        oc = "positive" if r["outcome"] in ("positive", "focal_positive") else "negative"
                        if norm not in pred or oc == "positive":
                            pred[norm] = oc

                    pos_m = sorted(m for m, o in pred.items() if o == "positive")
                    neg_m = sorted(m for m, o in pred.items() if o == "negative")

                    results_bi.append({
                        "patient_id":    pid,
                        "diagnosis":     diag,
                        "ihc_text":      text,
                        "pred_positive": ", ".join(pos_m) if pos_m else "(none)",
                        "pred_negative": ", ".join(neg_m) if neg_m else "(none)",
                        "pos_list":      pos_m,
                        "neg_list":      neg_m,
                        "accept":        True,
                    })
                    prog.progress((i + 1) / n_rows)

                prog.empty()
                st.session_state["bi_results"] = results_bi
                st.success(f"Processed {n_rows} rows.")

    if "bi_results" in st.session_state:
        bi_res = st.session_state["bi_results"]
        st.divider()
        st.subheader("Review results")
        st.caption("Uncheck rows you want to skip. Then save accepted rows and optionally get differentials.")

        diag_names_bi = load_diagnoses()[["canonical_diagnosis", "display_name"]]
        diag_fmt_bi   = diag_names_bi.set_index("canonical_diagnosis")["display_name"].to_dict()
        diag_list_bi  = diag_names_bi["canonical_diagnosis"].tolist()

        # Display table with accept toggles
        accept_flags = []
        for idx, r in enumerate(bi_res):
            with st.expander(
                f"{r['patient_id']} | +{r['pred_positive']} / -{r['pred_negative']}",
                expanded=False,
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    accept = st.checkbox("Accept this case", value=r["accept"],
                                         key=f"bi_acc_{idx}")
                    canon_bi = st.selectbox(
                        "Canonical diagnosis",
                        diag_list_bi,
                        format_func=lambda x: diag_fmt_bi.get(x, x),
                        key=f"bi_diag_{idx}",
                    )
                with col_b:
                    st.text_area("IHC text", value=r["ihc_text"], height=120,
                                 disabled=True, label_visibility="visible",
                                 key=f"bi_text_{idx}")
                bi_res[idx]["accept"]      = accept
                bi_res[idx]["canon_diag"]  = canon_bi
            accept_flags.append(bi_res[idx]["accept"])

        n_accepted = sum(accept_flags)
        st.caption(f"{n_accepted} of {len(bi_res)} cases accepted.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            save_all_bi = st.button(
                f"Save {n_accepted} accepted cases to database",
                key="bi_save",
                disabled=(n_accepted == 0),
            )
        with col_s2:
            ddx_all_bi = st.button(
                f"Get differentials for all {n_accepted} accepted cases",
                key="bi_ddx",
                disabled=(n_accepted == 0),
            )

        if save_all_bi:
            saved = 0
            for r in bi_res:
                if r["accept"]:
                    _write_to_db(
                        patient_id = r["patient_id"],
                        raw_diag   = r.get("diagnosis", r["patient_id"]),
                        canon_diag = r.get("canon_diag", diag_list_bi[0]),
                        ihc_text   = r["ihc_text"],
                        pred_pos   = r["pos_list"],
                        pred_neg   = r["neg_list"],
                    )
                    saved += 1
            st.success(f"Saved {saved} cases to the database.")

        if ddx_all_bi:
            ddx_batch_rows = []
            for r in bi_res:
                if r["accept"]:
                    ddx_r = _compute_ddx(r["pos_list"], r["neg_list"])
                    if not ddx_r.empty:
                        top_row = ddx_r.iloc[0]
                        ddx_batch_rows.append({
                            "Patient ID":     r["patient_id"],
                            "1st differential": top_row["display_name"],
                            "Lit score (%)":  top_row["lit_score"],
                            "Data score (%)": top_row["data_score"],
                            "Positive markers": r["pred_positive"],
                        })

            if ddx_batch_rows:
                st.subheader("Differential diagnosis summary")
                st.dataframe(pd.DataFrame(ddx_batch_rows), use_container_width=True,
                             hide_index=True)
            else:
                st.info("No literature coverage found for any accepted case.")

        st.write("")
        if st.button("Clear batch results", key="bi_clear"):
            del st.session_state["bi_results"]
            st.rerun()


# ── View 10: References ────────────────────────────────────────────────────────

if tab_choice == "References":
    section_header("Literature References",
                   "All sources underpinning the expected IHC profiles in the Literature Comparison view (IEEE format)")

    # ── Textbooks ──────────────────────────────────────────────────────────────
    st.subheader("Textbooks")

    refs_books = [
        (
            "[1]",
            "WHO Classification of Tumours Editorial Board",
            "*Soft Tissue and Bone Tumours*, 5th ed.",
            "Lyon: International Agency for Research on Cancer (IARC), 2020.",
            "WHO 5th ed. 2020",
        ),
        (
            "[2]",
            "C. D. M. Fletcher, Ed.",
            "*Diagnostic Histopathology of Tumours*, 5th ed.",
            "Philadelphia: Elsevier, 2020.",
            "Fletcher 2020",
        ),
        (
            "[3]",
            "J. L. Hornick",
            "*Practical Soft Tissue Pathology: A Diagnostic Approach*, 2nd ed.",
            "Philadelphia: Elsevier, 2019.",
            "Hornick 2019",
        ),
        (
            "[13]",
            "C. P. Crum, M. R. Nucci, B. E. Howitt, S. R. Granter, M. M. Parast, and T. K. Boyd, Eds.",
            "*Diagnostic Gynecologic and Obstetric Pathology*, 3rd ed.",
            "Philadelphia: Elsevier, 2016.",
            "Crum 2016",
        ),
    ]

    for num, authors, title, publisher, tag in refs_books:
        st.markdown(
            f"{num} &nbsp; {authors}, {title} {publisher} "
            f"<span style='color:#6b7280;font-size:0.8rem;'>({tag})</span>",
            unsafe_allow_html=True,
        )

    # ── Journal articles ───────────────────────────────────────────────────────
    st.subheader("Journal Articles")

    refs_papers = [
        (
            "[4]",
            "L. A. Doyle, M. Vivero, C. D. M. Fletcher, F. Mertens, and J. L. Hornick,",
            '"Nuclear expression of STAT6 distinguishes solitary fibrous tumor from histologic mimics,"',
            "*Mod. Pathol.*, vol. 27, no. 3, pp. 390-395, 2014.",
            "Doyle 2014",
        ),
        (
            "[5]",
            "J. Terry, T. Saito, S. Subramanian, C. Ruttan, and J. R. Goldblum,",
            '"TLE1 as a diagnostic immunohistochemical marker for synovial sarcoma emerging from gene expression profiling studies,"',
            "*Am. J. Surg. Pathol.*, vol. 31, no. 2, pp. 240-246, 2007.",
            "Terry 2007",
        ),
        (
            "[6]",
            "R. B. West, C. L. Corless, X. Chen, B. P. Rubin, S. Subramanian, K. Montgomery, et al.,",
            '"The novel marker, DOG1, is expressed ubiquitously in gastrointestinal stromal tumors irrespective of receptor status,"',
            "*Am. J. Pathol.*, vol. 165, no. 1, pp. 107-113, 2004.",
            "West 2004",
        ),
        (
            "[7]",
            "M. Ladanyi, M. Y. Lui, C. R. Antonescu, A. Krause-Boehm, A. Meindl, P. Argani, et al.,",
            '"The der(17)t(X;17)(p11;q25) of human alveolar soft part sarcoma fuses the TFE3 transcription factor gene to ASPL,"',
            "*Oncogene*, vol. 20, no. 1, pp. 48-57, 2001.",
            "Ladanyi 2001",
        ),
        (
            "[8]",
            "P. Dias, B. Chen, S. M. Bhatt, R. B. Houghton, J. Houghton, and P. J. Bhattacharya,",
            '"Strong immunostaining for myogenin in rhabdomyosarcoma is significantly associated with tumors of the alveolar subclass,"',
            "*Am. J. Pathol.*, vol. 156, no. 2, pp. 399-408, Feb. 2000.",
            "Dias 2000",
        ),
        (
            "[9]",
            "B. Bhattacharya, H. P. Dilworth, C. Iacobuzio-Donahue, F. Ricci, K. Weber, D. E. Furlong, et al.,",
            '"Nuclear beta-catenin expression distinguishes deep fibromatosis from other benign and malignant fibroblastic and myofibroblastic lesions,"',
            "*Am. J. Surg. Pathol.*, vol. 29, no. 5, pp. 653-659, 2005.",
            "Bhattacharya 2005",
        ),
        (
            "[10]",
            "A. L. Folpe, J. C. Fanburg-Smith, M. Miettinen, and S. W. Weiss,",
            '"Atypical and malignant glomus tumors: analysis of 52 cases, with a proposal for the reclassification of glomus tumors,"',
            "*Am. J. Surg. Pathol.*, vol. 25, no. 1, pp. 1-12, 2001.",
            "Folpe 2001",
        ),
        (
            "[11]",
            "M. B. Binh, X. Sastre-Garau, L. Guillou, G. de Pinieux, P. Terrier, R. Lagace, et al.,",
            '"MDM2 and CDK4 immunostainings are useful adjuncts in diagnosing well-differentiated and dedifferentiated liposarcoma subtypes: a comparative analysis of 559 soft tissue neoplasms with genetic data,"',
            "*Am. J. Surg. Pathol.*, vol. 29, no. 10, pp. 1340-1347, 2005.",
            "Binh 2005",
        ),
        (
            "[12]",
            "L. A. Doyle, W. Wang, O. Dal Cin, S. Bhatt, V. Tao, J. L. Hornick, and F. Mertens,",
            '"MUC4 is a highly sensitive and specific marker for low-grade fibromyxoid sarcoma,"',
            "*Am. J. Surg. Pathol.*, vol. 35, no. 5, pp. 733-741, 2011.",
            "Doyle 2011",
        ),
        (
            "[14]",
            "T. W. Kelley, E. C. Borden, and J. R. Goldblum,",
            '"Estrogen and progesterone receptor expression in uterine and extrauterine leiomyosarcomas: an immunohistochemical study,"',
            "*Am. J. Clin. Pathol.*, vol. 122, no. 5, pp. 688-694, 2004.",
            "Kelley 2004",
        ),
    ]

    for num, authors, title, journal, tag in refs_papers:
        st.markdown(
            f"{num} &nbsp; {authors} {title} {journal} "
            f"<span style='color:#6b7280;font-size:0.8rem;'>({tag})</span>",
            unsafe_allow_html=True,
        )

    # ── Websites ───────────────────────────────────────────────────────────────
    st.subheader("Online Resources")

    st.markdown(
        "[15] &nbsp; PathologyOutlines.com. [Online]. Available: "
        "https://www.pathologyoutlines.com. Accessed: Aug. 2026. "
        "<span style='color:#6b7280;font-size:0.8rem;'>(PathologyOutlines)</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "These references underpin the expected IHC frequencies shown in the "
        "Literature Comparison view. The literature profiles are not derived from "
        "Institutional data and represent published consensus values only."
    )
