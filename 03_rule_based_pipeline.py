"""
03_rule_based_pipeline.py
IHC NLP Project : Rule-Based Extraction Pipeline
--------------------------------------------------
Extracts IHC marker names and outcomes (positive / negative / focal_positive)
from free-text histopathology reports using regex patterns.

Evaluated against the leiomyosarcoma HG pilot ground truth.

Design decisions (see NOTES.md):
- Focal/patchy positivity classified as positive (matches ground truth convention)
- Referred and in-house results both extracted; conflicts flagged
- Marker names normalised to lowercase for evaluation
- No hard marker whitelist : patterns alone drive extraction

Usage:
    python 03_rule_based_pipeline.py

Outputs:
    data/pilot_predictions.csv   : per-case extraction results
    data/pilot_errors.csv        : cases where pipeline made errors (for analysis)
"""

import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR   = Path("data")
PILOT_FILE = DATA_DIR / "pilot_leiomyosarcoma.csv"

# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text):
    """Split on '. ' or '.\n', keeping partial sentences."""
    parts = re.split(r'\.\s+|\.\n', str(text).strip())
    return [p.strip() for p in parts if p.strip()]


# ── Source classification (referred vs in-house) ──────────────────────────────

REFERRED_RE = re.compile(
    r'referred\s+immuno|by\s+referred|external\s+immuno', re.IGNORECASE)
INHOUSE_RE  = re.compile(
    r'performed\s+at\s+RNOH|done\s+in[\s\-]house|immuno(?:histochemistry)?\s+at\s+RNOH'
    r'|we\s+have\s+done\s+here|in[\s\-]house\s+immuno', re.IGNORECASE)

def sentence_source(sentence):
    if REFERRED_RE.search(sentence):
        return 'referred'
    if INHOUSE_RE.search(sentence):
        return 'inhouse'
    return 'unknown'


# ── Marker list parser ────────────────────────────────────────────────────────

# Words that appear in marker context but are NOT marker names
NON_MARKER = {
    'the', 'other', 'remaining', 'following', 'these', 'those', 'a', 'an',
    'all', 'both', 'each', 'some', 'any', 'no', 'not', 'only', 'also',
    'and', 'or', 'but', 'with', 'for', 'of', 'in', 'on', 'at', 'by', 'to',
    'are', 'is', 'were', 'was', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may', 'might',
    'immunostains', 'immunostain', 'markers', 'marker', 'stains', 'stain',
    'tested', 'performed', 'done', 'shown', 'noted', 'seen', 'observed',
    'positive', 'negative', 'focal', 'focally', 'patchy', 'weak', 'weakly',
    'strong', 'strongly', 'diffuse', 'diffusely', 'variable', 'variably',
    'scattered', 'isolated', 'occasional', 'rare', 'extensive', 'intense',
    'cytoplasmic', 'nuclear', 'membranous', 'expression', 'staining',
    'reactivity', 'reaction', 'highlighted', 'expressed', 'retained',
    'tumour', 'tumor', 'cells', 'cell', 'lesional', 'atypical', 'spindle',
    'further', 'additional', 'subsequent',
}

def parse_marker_list(raw):
    """
    Parse a raw text fragment into individual marker tokens.
    Handles comma/and-separated lists and parenthetical synonyms.
    """
    if not raw or not raw.strip():
        return []

    # Remove trailing sentence punctuation
    raw = raw.strip(' .;:')

    # Expand 'and' to comma (but not inside parentheses)
    raw = re.sub(r'\band\b', ',', raw, flags=re.IGNORECASE)

    # Handle parenthetical synonyms: keep main name, drop parens content
    # e.g. "pancytokeratin (MNF116)" → "pancytokeratin"
    raw = re.sub(r'\([^)]*\)', '', raw)

    tokens = [t.strip() for t in raw.split(',')]
    markers = []

    for token in tokens:
        token = token.strip(' .')
        if not token:
            continue

        # Strip leading adjectives common in pathology prose
        token = re.sub(
            r'^(?:diffuse(?:ly)?|strong(?:ly)?|focal(?:ly)?|patchy|weak(?:ly)?'
            r'|variable|scattered|isolated|extensive|intense|diffuse\s+and\s+strong(?:ly)?'
            r'|strong\s+and\s+diffuse(?:ly)?)\s+',
            '', token, flags=re.IGNORECASE)
        token = token.strip()

        if not token:
            continue

        # Filter: all words are non-marker noise words
        words = set(token.lower().split())
        if words.issubset(NON_MARKER):
            continue

        # Filter: token is a pure number or single char
        if re.match(r'^\d+$', token) or len(token) < 2:
            continue

        # Filter: too long to be a marker name (probably a phrase)
        if len(token) > 35:
            continue

        markers.append(token)

    return markers


# ── Extraction patterns ───────────────────────────────────────────────────────
#
# Each entry: (compiled_regex, outcome)
# The regex must contain ONE capture group for the marker list text.

PATTERNS = [

    # ── NEGATION OF EXPRESSION (must come before positive patterns) ──────────

    # "no expression of X, Y"
    (re.compile(r'no\s+expression\s+of\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'negative'),

    # "there is no X" (e.g. "there is no staining for X")
    (re.compile(r'there\s+is\s+no\s+(?:expression\s+of\s+|staining\s+(?:for|with)\s+)?([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'negative'),

    # ── STRUCTURED FORMAT (e.g. "Positive: X, Y\nNegative: X, Y") ──────────

    # "Positive: X, Y, Z" (colon-delimited label; works mid-sentence too)
    (re.compile(r'\b[Pp]ositive\s*:\s*([\w][\w\s,/\(\)&\-\.]*?)(?=\.(?:\s|$)|;|$|\n)',
                re.IGNORECASE), 'positive'),

    # "Negative: X, Y, Z"
    (re.compile(r'^[Nn]egative\s*:\s*([\w][\w\s,/\(\)&\-\.]*?)(?=\.(?:\s|$)|;|$|\n)',
                re.IGNORECASE | re.MULTILINE), 'negative'),

    # "The following markers are positive: X, Y"
    (re.compile(r'following\s+(?:markers?|immunostains?|stains?)?\s+are\s+positive[:\s]+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'positive'),

    # "The following markers are negative: X, Y"
    (re.compile(r'following\s+(?:markers?|immunostains?|stains?)?\s+are\s+negative[:\s]+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'negative'),

    # ── LIST-DASH FORMAT: "Marker, Marker - [qualifier] positive/negative" ──
    # Handles structured IHC reports where multiple markers share an outcome:
    #   "STAT6, CD34, bcl-2 - widespread, strong positive staining."
    #   "SMA, cytokeratins, calretinin - negative"
    # Capture group restricted to single-line (no \n) to avoid cross-sentence grabs.
    # Noise words (e.g. "widespread", "cytokeratins") are dropped by normalise_marker().

    # "X, Y, Z - [qualifier] positive" (list-dash positive)
    (re.compile(
        r'([\w][\w ,/\(\)&\-\.]+?)\s+-\s+[^\n]*?positive',
        re.IGNORECASE), 'positive'),

    # "X, Y, Z - [qualifier] negative" (list-dash negative)
    (re.compile(
        r'([\w][\w ,/\(\)&\-\.]+?)\s+-\s+[^\n]*?negative',
        re.IGNORECASE), 'negative'),

    # ── FOCAL / PATCHY (check before positive to avoid misclassification) ──

    # "focal positivity for X, Y" -- stop before "and negative" or "negative for" boundary
    (re.compile(r'focal(?:ly)?\s+positiv\w+\s+for\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.(?:\s|$)|;|$|\band\s+(?:\w+\s+)*negative\b|\bnegative\s+for\b)',
                re.IGNORECASE), 'focal_positive'),

    # "patchy positivity/staining for/with X"
    (re.compile(r'patchy\s+(?:positivity|staining)\s+(?:for|with)\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.(?:\s|$)|;|$)',
                re.IGNORECASE), 'focal_positive'),

    # "X shows patchy/diffuse/etc positivity" (inverted form: marker before qualifier)
    (re.compile(r'([\w/\-\.]{2,20})\s+shows?\s+(?:patchy|diffuse|focal|scattered|widespread|strong|weak|variable)\s+positivity\b',
                re.IGNORECASE), 'focal_positive'),

    # "weak positivity for X"
    (re.compile(r'weak(?:ly)?\s+positiv\w+\s+(?:for\s+)?([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'focal_positive'),

    # "variable positivity for X"
    (re.compile(r'variable\s+positivity\s+for\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'focal_positive'),

    # "focal X" (e.g. "focal desmin")
    (re.compile(r'\bfocal\s+((?!positiv)\w[\w\s/\-\.]*?)(?=\.|;|,|$|and\b)',
                re.IGNORECASE), 'focal_positive'),

    # ── POSITIVE ────────────────────────────────────────────────────────────

    # "positive for X, Y, Z"
    (re.compile(r'\bpositive\s+for\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$|negative\b)',
                re.IGNORECASE), 'positive'),

    # "positive with X, Y" (alternate phrasing: "positive with Smooth Muscle Actin")
    (re.compile(r'\bpositive\s+with\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$|negative\b)',
                re.IGNORECASE), 'positive'),

    # "stains with X, Y" / "stain with X" (without requiring strong/diffuse qualifier)
    (re.compile(r'\bstains?\s+with\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$|and\s+are\b|negative\b)',
                re.IGNORECASE), 'positive'),

    # "X staining is retained" / "X shows ... retention" (INI1, SMARCA4 etc.)
    (re.compile(r'([\w][\w\s/\-\.]*?)\s+(?:staining\s+is|is)\s+retained\b',
                re.IGNORECASE), 'positive'),

    # "immunoreactive for X, Y"
    (re.compile(r'immunoreactive\s+for\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$|\band\b\s+are)',
                re.IGNORECASE), 'positive'),

    # "diffuse expression of X, Y" (not preceded by "no")
    (re.compile(r'(?<!no\s)(?:diffuse\s+)?expression\s+of\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'positive'),

    # "staining with X, Y" (positive context : used after positive intro)
    (re.compile(r'(?:strong|diffuse|strongly|diffusely)\s+staining\s+with\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'positive'),

    # "highlighted by X, Y"
    (re.compile(r'highlight(?:ed|s)?\s+by\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.|;|$)',
                re.IGNORECASE), 'positive'),

    # "express/expresses X, Y" (capped to avoid over-capture)
    (re.compile(r'\bexpress(?:es)?\s+([\w][\w\s,/\(\)&\-\.]{1,60}?)(?=\.|;|$|\bwhereas\b|\bwhile\b|\band\s+are\b)',
                re.IGNORECASE), 'positive'),

    # "Positive tumour/cells: X, Y, Z"
    (re.compile(r'[Pp]ositive\s+(?:tumour\s+)?cells?[:\s]+\s*([\w][\w\s,/\(\)&\-\.]+?)(?=\.|;|$|[Nn]egative)',
                re.IGNORECASE), 'positive'),

    # "X, Y and Z are/is/were positive" (list before keyword)
    (re.compile(r'([\w][\w\s,/\(\)&\-\.]+?)\s+(?:is|are|were|was)\s+(?:diffusely\s+|strongly\s+|diffuse(?:ly)?\s+and\s+strongly\s+)?positive',
                re.IGNORECASE), 'positive'),

    # "X-positive [cells/tumour/...]" adjective form (e.g. "CD68-positive osteoclast-like cells")
    # normalise_marker() will reject anything not in CANONICAL_MARKERS
    (re.compile(r'\b([\w]{2,12}(?:/[\w]{1,6})?)-positive\b',
                re.IGNORECASE), 'positive'),

    # "X positive" : short list only, to avoid greedy over-capture
    (re.compile(r'([\w/\-\.]+(?:\s*,\s*[\w/\-\.]+){0,4})\s+positive\b',
                re.IGNORECASE), 'positive'),

    # ── NEGATIVE ────────────────────────────────────────────────────────────

    # "negative for X, Y, Z"
    # Lookahead uses \.(?:\s|$) so periods inside marker names (CAM5.2) don't stop match early
    (re.compile(r'\bnegative\s+for\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.(?:\s|$)|;|$)',
                re.IGNORECASE), 'negative'),

    # "negative with X, Y"
    (re.compile(r'\bnegative\s+with\s+([\w][\w\s,/\(\)&\-\.]*?)(?=\.(?:\s|$)|;|$)',
                re.IGNORECASE), 'negative'),

    # "are/is negative (X, Y, Z)"
    (re.compile(r'(?:is|are|were|was)\s+negative\s*\(\s*([\w][\w\s,/\(\)&\-\.,]+?)\s*\)',
                re.IGNORECASE), 'negative'),

    # "the other markers/immunostains are negative (X, Y)"
    (re.compile(r'(?:other\s+)?(?:markers?|immunostains?|stains?)\s+(?:tested\s+)?(?:are|were|is)\s+negative[:\s]*\(?\s*([\w][\w\s,/\(\)&\-\.,]+?)\s*\)?(?=\.(?:\s|$)|;|$)',
                re.IGNORECASE), 'negative'),

    # "Negative on tumour cells: X, Y"
    (re.compile(r'[Nn]egative\s+(?:on\s+(?:tumour\s+)?cells?|results?)[:\s]+\s*([\w][\w\s,/\(\)&\-\.]+?)(?=\.|;|$)',
                re.IGNORECASE), 'negative'),

    # "X, Y are/is/were negative"
    (re.compile(r'([\w][\w\s,/\(\)&\-\.]+?)\s+(?:is|are|were|was)\s+negative\b',
                re.IGNORECASE), 'negative'),

    # "X is negative" (single marker)
    (re.compile(r'([\w/\-\.]+)\s+(?:is|are|were|was)\s+negative',
                re.IGNORECASE), 'negative'),

]


# ── Core extraction function ──────────────────────────────────────────────────

def extract_from_text(text):
    """
    Process a full IHC text and return a list of dicts:
      {'marker': str, 'outcome': str, 'source': str}
    outcome: 'positive', 'negative', 'focal_positive'
    source:  'referred', 'inhouse', 'unknown'
    """
    results = []
    seen = set()   # (marker_lower, outcome, source) : deduplication

    # Pre-process: normalise < used as a comma typo (e.g. "SMA< desmin")
    # but preserve < in numeric comparisons like "Ki-67 < 1%"
    text = re.sub(r'<(?!\s*[\d.%])', ',', str(text).strip())

    sentences = split_sentences(text)

    for sentence in sentences:
        source = sentence_source(sentence)

        for pattern, outcome in PATTERNS:
            for match in pattern.finditer(sentence):
                raw = match.group(1)
                markers = parse_marker_list(raw)
                for marker in markers:
                    key = (marker.lower(), outcome, source)
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            'marker': marker,
                            'outcome': outcome,
                            'source': source,
                        })

    return results


# ── Marker normalisation (for evaluation) ─────────────────────────────────────

def normalise(marker):
    """Lowercase, strip spaces and hyphens for fuzzy comparison."""
    return re.sub(r'[\s\-_]', '', marker.lower())


SYNONYMS = {
    # Map alternate forms → canonical form (lowercase, no hyphens)
    'ae1ae3': 'ae1/3', 'ae1/ae3': 'ae1/3', 'ae13': 'ae1/3',
    'ae1ae3': 'ae1/3', 'aae13': 'ae1/3',
    'mnf116': 'mnf116',
    'hcaldesmon': 'caldesmon', 'h-caldesmon': 'caldesmon', 'caldesmon': 'caldesmon',
    'sma': 'sma', 'smoothmuscleactin': 'sma',
    'ki67': 'ki67', 'ki-67': 'ki67',
    'cam52': 'cam5.2', 'cam5.2': 'cam5.2',
    'panck': 'panck', 'pancytokeratin': 'panck', 'pan-ck': 'panck',
    'melanA': 'melan-a', 'melana': 'melan-a',
    'hmb45': 'hmb45', 'hmb-45': 'hmb45',
}

def canonical(marker):
    n = normalise(marker)
    return SYNONYMS.get(n, n)


# ── Evaluation ────────────────────────────────────────────────────────────────

def parse_gt_list(cell):
    """Parse ground truth comma-separated cell into a set of canonical markers."""
    if pd.isna(cell) or str(cell).strip().lower() in {'nil', 'na', 'n/a', ''}:
        return set()
    return {canonical(m.strip()) for m in str(cell).split(',') if m.strip()}


def evaluate_case(predictions, gt_positive, gt_negative):
    """
    Compare pipeline predictions against ground truth for one case.
    focal_positive is treated as positive for evaluation.
    Returns dict of TP, FP, FN counts for positive and negative separately.
    """
    # Predicted positive = all non-negative outcomes
    pred_pos = {canonical(r['marker']) for r in predictions
                if r['outcome'] in ('positive', 'focal_positive')}
    pred_neg = {canonical(r['marker']) for r in predictions
                if r['outcome'] == 'negative'}

    tp_pos = len(pred_pos & gt_positive)
    fp_pos = len(pred_pos - gt_positive)
    fn_pos = len(gt_positive - pred_pos)

    tp_neg = len(pred_neg & gt_negative)
    fp_neg = len(pred_neg - gt_negative)
    fn_neg = len(gt_negative - pred_neg)

    return {
        'tp_pos': tp_pos, 'fp_pos': fp_pos, 'fn_pos': fn_pos,
        'tp_neg': tp_neg, 'fp_neg': fp_neg, 'fn_neg': fn_neg,
    }


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = pd.read_csv(PILOT_FILE)

    # Only evaluate on cases that have ground truth in both columns
    df_eval = df[
        df['positive_ihc'].notna() | df['negative_ihc'].notna()
    ].copy()

    print(f"Pilot cases:           {len(df)}")
    print(f"Cases with GT labels:  {len(df_eval)}")
    print()

    all_stats   = []
    error_cases = []

    for _, row in df_eval.iterrows():
        predictions  = extract_from_text(row['ihc_text'])
        gt_pos       = parse_gt_list(row['positive_ihc'])
        gt_neg       = parse_gt_list(row['negative_ihc'])

        stats = evaluate_case(predictions, gt_pos, gt_neg)
        stats['patient_id'] = row['patient_id']
        all_stats.append(stats)

        # Flag cases with errors for review
        if stats['fp_pos'] + stats['fn_pos'] + stats['fp_neg'] + stats['fn_neg'] > 0:
            pred_pos = {canonical(r['marker']) for r in predictions
                        if r['outcome'] in ('positive', 'focal_positive')}
            pred_neg = {canonical(r['marker']) for r in predictions
                        if r['outcome'] == 'negative'}
            error_cases.append({
                'patient_id':    row['patient_id'],
                'ihc_text':      str(row['ihc_text'])[:300],
                'gt_positive':   row['positive_ihc'],
                'gt_negative':   row['negative_ihc'],
                'pred_positive': ', '.join(sorted(pred_pos)),
                'pred_negative': ', '.join(sorted(pred_neg)),
                'fp_pos': stats['fp_pos'], 'fn_pos': stats['fn_pos'],
                'fp_neg': stats['fp_neg'], 'fn_neg': stats['fn_neg'],
            })

    # Aggregate
    tp_pos = sum(s['tp_pos'] for s in all_stats)
    fp_pos = sum(s['fp_pos'] for s in all_stats)
    fn_pos = sum(s['fn_pos'] for s in all_stats)
    tp_neg = sum(s['tp_neg'] for s in all_stats)
    fp_neg = sum(s['fp_neg'] for s in all_stats)
    fn_neg = sum(s['fn_neg'] for s in all_stats)

    p_pos, r_pos, f1_pos = prf(tp_pos, fp_pos, fn_pos)
    p_neg, r_neg, f1_neg = prf(tp_neg, fp_neg, fn_neg)

    # Micro-averaged overall F1
    tp_all = tp_pos + tp_neg
    fp_all = fp_pos + fp_neg
    fn_all = fn_pos + fn_neg
    p_all, r_all, f1_all = prf(tp_all, fp_all, fn_all)

    print("=" * 50)
    print("EVALUATION RESULTS : leiomyosarcoma HG pilot")
    print("=" * 50)
    print(f"{'':20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'Positive markers':20} {p_pos:>10.4f} {r_pos:>10.4f} {f1_pos:>10.4f}")
    print(f"{'Negative markers':20} {p_neg:>10.4f} {r_neg:>10.4f} {f1_neg:>10.4f}")
    print(f"{'Overall (micro)':20} {p_all:>10.4f} {r_all:>10.4f} {f1_all:>10.4f}")
    print()
    print(f"TP pos={tp_pos}  FP pos={fp_pos}  FN pos={fn_pos}")
    print(f"TP neg={tp_neg}  FP neg={fp_neg}  FN neg={fn_neg}")
    print()

    # Save error cases for analysis
    pd.DataFrame(error_cases).to_csv(DATA_DIR / 'pilot_errors.csv', index=False)
    pd.DataFrame(all_stats).to_csv(DATA_DIR / 'pilot_predictions.csv', index=False)
    print(f"Error cases saved: data/pilot_errors.csv ({len(error_cases)} cases)")
    print(f"All stats saved:   data/pilot_predictions.csv")
