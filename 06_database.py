"""
06_database.py
IHC NLP Project -- SQLite Database Construction
-------------------------------------------------
Builds a structured SQLite database from:
  - data/hybrid_predictions.csv  : pipeline predictions for all GT cases
  - data/cleaned_ground_truth.csv: case metadata and GT markers
  - data/lms_subclassification.csv: canonical LMS labels + subclassifications
                                     derived from RNOH masterfile spreadsheets
  - Built-in literature profiles  : expected IHC panels for common diagnoses

Diagnosis normalisation decisions:
  - Capitalisation variants merged to a single canonical form (e.g. all
    leiomyosarcoma HG variants -> leiomyosarcoma_hg).
  - PDS / AFX variants merged: pleomorphic dermal sarcoma, atypical
    fibroxanthoma, and the combined label all map to pleomorphic_dermal_sarcoma.
  - LMS cases reassigned using the masterfile AD column (Final Diagnosis
    for the AI), which provides reviewed clinical labels and distinguishes
    uterine from soft tissue variants.
  - Subclassification (masterfile AG column) stored as a separate field for
    LMS cases (e.g. cutaneous, leiomyoma, STUMP, myxoid, epithelioid).
  - Unconfirmed diagnoses (prefix unconfirmed_) flagged as reserved;
    retained in database but excluded from main analysis.
  - Diagnoses with n < MIN_CASES_ANALYSIS after merging flagged as low_count;
    retained but excluded from comparative statistics.

Schema:
  diagnoses       -- canonical labels, n_cases, include_in_analysis flag
  cases           -- per-patient metadata
  ihc_markers     -- per-patient per-marker outcomes (predicted + GT)
  ki67_results    -- per-patient Ki-67 proliferation index
  literature      -- expected IHC profiles from published references

Usage:
    python 06_database.py --build      # build (or rebuild if exists)
    python 06_database.py --stats      # print summary statistics
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd

from marker_utils import normalise_marker, parse_and_normalise

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "ihc_database.db"
MIN_CASES_ANALYSIS = 30   # minimum cases for a diagnosis to appear in main analysis


# ── Diagnosis normalisation map ────────────────────────────────────────────────
# (raw GT label) -> (canonical_name, is_confirmed)
# LMS cases are further re-labelled at ingestion time using lms_subclassification.csv.

DIAG_MAP: dict[str, tuple[str, bool]] = {
    # LMS family (re-labelled by masterfile; entries here cover any GT cases not in masterfile)
    "leiomyosarcoma hg":             ("leiomyosarcoma_hg",              True),
    "leiomyosarcoma HG":             ("leiomyosarcoma_hg",              True),
    "Leiomyosarcoma HG":             ("leiomyosarcoma_hg",              True),
    "leiomyosarcoma ASMT":           ("leiomyosarcoma_asmt",            True),
    "uterine leiomyosarcoma hg":     ("uterine_leiomyosarcoma_hg",      True),
    "Uterine leiomyosarcoma":        ("uterine_leiomyosarcoma",         True),
    # Confirmed diagnoses
    "ewing sarcoma":                 ("ewing_sarcoma",                  True),
    "fibromatosis desmoid":          ("fibromatosis_desmoid",           True),
    "intramuscular myxoma":          ("intramuscular_myxoma",           True),
    "solitary fibrous tumour":       ("solitary_fibrous_tumour",        True),
    "pleomorphic dermal sarcoma/atypical fibroxanthoma": ("pleomorphic_dermal_sarcoma", True),
    "pleomorphic dermal sarcoma":    ("pleomorphic_dermal_sarcoma",     True),
    "Atypical fibroxanthoma":        ("pleomorphic_dermal_sarcoma",     True),
    "synovial sarcoma":              ("synovial_sarcoma",               True),
    "Synovial sarcoma":              ("synovial_sarcoma",               True),
    "embryonal rhabdomyosarcoma":    ("embryonal_rhabdomyosarcoma",     True),
    "Dermatofibrosarcoma protuberans": ("dermatofibrosarcoma_protuberans", True),
    "dedifferentiated liposarcoma":  ("dedifferentiated_liposarcoma",   True),
    "nodular fasciitis":             ("nodular_fasciitis",              True),
    "neurofibroma":                  ("neurofibroma",                   True),
    "Neurofibroma":                  ("neurofibroma",                   True),
    "angiosarcoma":                  ("angiosarcoma",                   True),
    "alveolar rhabdomyosarcoma":     ("alveolar_rhabdomyosarcoma",      True),
    "low grade fibromyxoid sarcoma": ("low_grade_fibromyxoid_sarcoma",  True),
    "granular cell tumour ":         ("granular_cell_tumour",           True),
    "granular cell tumour":          ("granular_cell_tumour",           True),
    "clear cell sarcoma of soft parts": ("clear_cell_sarcoma",         True),
    "malignant peripheral nerve sheath tumour": ("mpnst",              True),
    "Malignant peripheral nerve sheath tumour": ("mpnst",              True),
    "Epithelioid sarcoma":           ("epithelioid_sarcoma",            True),
    "extraskeletal myxoid chondrosarcoma": ("extraskeletal_myxoid_chondrosarcoma", True),
    "giant cell tumour of bone":     ("giant_cell_tumour_of_bone",      True),
    "hybrid nerve sheath tumour":    ("hybrid_nerve_sheath_tumour",     True),
    "Spindle cell/pleomorphic lipoma": ("spindle_cell_pleomorphic_lipoma", True),
    "pleomorphic liposarcoma":       ("pleomorphic_liposarcoma",        True),
    "glomus tumour":                 ("glomus_tumour",                  True),
    "myopericytoma":                 ("myopericytoma",                  True),
    "pleomorphic rhabdomyosarcoma":  ("pleomorphic_rhabdomyosarcoma",   True),
    "dermatofibroma/benign fibrous histiocytoma": ("dermatofibroma",   True),
    "gastrointestinal stromal tumour": ("gastrointestinal_stromal_tumour", True),
    "alveolar soft part sarcoma":    ("alveolar_soft_part_sarcoma",     True),
    "myxoid liposarcoma":            ("myxoid_liposarcoma",             True),
    "Myxoid liposarcoma":            ("myxoid_liposarcoma",             True),
    "Chondrosarcoma":                ("chondrosarcoma",                 True),
    "chondrosarcoma":                ("chondrosarcoma",                 True),
    "merkel cell carcinoma":         ("merkel_cell_carcinoma",          True),
    "Undifferentiated sarcoma":      ("undifferentiated_sarcoma",       True),
    "perineurioma":                  ("perineurioma",                   True),
    "neuroblastoma":                 ("neuroblastoma",                  True),
    "Neuroblastoma":                 ("neuroblastoma",                  True),
    "tenosynovial giant cell tumour": ("tenosynovial_giant_cell_tumour", True),
    "Malignant melanoma":            ("malignant_melanoma",             True),
    "Inflammatory myofibroblastic tumour": ("inflammatory_myofibroblastic_tumour", True),
    "Schwannoma":                    ("schwannoma",                     True),
    "chordoma":                      ("chordoma",                       True),
    "CIC-rearranged sarcoma":        ("cic_rearranged_sarcoma",         True),
    "ALK-rearranged tumour":         ("alk_rearranged_tumour",          True),
    "Chondroblastoma":               ("chondroblastoma",                True),
    "fibromatosis superficial":      ("fibromatosis_superficial",       True),
    "myxofibrosarcoma":              ("myxofibrosarcoma",               True),
    "Myxofibrosarcoma":              ("myxofibrosarcoma",               True),
    "Epithelioid haemangioendothelioma":  ("epithelioid_haemangioendothelioma", True),
    "Epithelioid haemangioendothelioma ": ("epithelioid_haemangioendothelioma", True),
    "hibernoma":                     ("hibernoma",                      True),
    "Chondromyxoid fibroma ":        ("chondromyxoid_fibroma",          True),
    "Central chondrosarcoma":        ("central_chondrosarcoma",         True),
    "Dedifferentiated chondrosarcoma": ("dedifferentiated_chondrosarcoma", True),
    "Inflammatory rhabdomyoblastic tumour": ("inflammatory_rhabdomyoblastic_tumour", True),
    "Spindle cell tumour/ inflammatory myofibroblastic tumour\xa0": ("spindle_cell_imt", True),
    # Melanoma variants (all map to malignant_melanoma)
    "Melanoma":                          ("malignant_melanoma",                  True),
    "melanoma":                          ("malignant_melanoma",                  True),
    # Plasma cell tumour
    "plasma cell tumour":                ("plasma_cell_tumour",                  True),
    "Plasma cell tumour":                ("plasma_cell_tumour",                  True),
    "Plasma Cell Tumour":                ("plasma_cell_tumour",                  True),
    # Endometrial stromal sarcoma (confirmed, low-grade)
    "endometrial stromal sarcoma low grade":        ("endometrial_stromal_sarcoma_lg",  True),
    "uterine endometrial stromal sarcoma low grade": ("endometrial_stromal_sarcoma_lg", True),
    # CIC rearranged (without hyphen)
    "CIC rearranged sarcoma":            ("cic_rearranged_sarcoma",              True),
    # BCOR rearranged sarcoma
    "BCOR rearranged sarcoma":           ("bcor_rearranged_sarcoma",             True),
    # IMT alias for ALK-rearranged
    "IMT ALK-rearranged tumour":         ("alk_rearranged_tumour",               True),
    # Unconfirmed (reserved)
    "unconfirmed_chordoma":              ("unconfirmed_chordoma",                False),
    "unconfirmed_dermatofibrosarcoma_protuberans":       ("unconfirmed_dfsp",                    False),
    "unconfirmed_synovial_sarcoma":                      ("unconfirmed_synovial_sarcoma",         False),
    "unconfirmed_embryonal_rhabdomyosarcoma":            ("unconfirmed_embryonal_rhabdomyosarcoma", False),
    "unconfirmed_fibromatosis_desmoid":                  ("unconfirmed_fibromatosis_desmoid",     False),
    "unconfirmed_alveolar_rhabdomyosarcoma":             ("unconfirmed_alveolar_rhabdomyosarcoma", False),
    "unconfirmed_myxoid_liposarcoma":                    ("unconfirmed_myxoid_liposarcoma",       False),
    "unconfirmed_dedifferentiated_liposarcoma":          ("unconfirmed_dedifferentiated_liposarcoma", False),
    "unconfirmed_nodular_fasciitis":                     ("unconfirmed_nodular_fasciitis",        False),
    "unconfirmed_low_grade_fibromyxoid_sarcoma":         ("unconfirmed_low_grade_fibromyxoid_sarcoma", False),
    "unconfirmed_epithelioid_sarcoma":                   ("unconfirmed_epithelioid_sarcoma",      False),
    "unconfirmed_granular_cell_tumour ":                 ("unconfirmed_granular_cell_tumour",     False),
    "unconfirmed_alveolar_soft_part_sarcoma":            ("unconfirmed_alveolar_soft_part_sarcoma", False),
    "unconfirmed_solitary_fibrous_tumour":               ("unconfirmed_solitary_fibrous_tumour",  False),
    "unconfirmed_Spindle cell/pleomorphic lipoma":       ("unconfirmed_spindle_cell_pleomorphic_lipoma", False),
    "unconfirmed_pleomorphic_liposarcoma":               ("unconfirmed_pleomorphic_liposarcoma",  False),
    "unconfirmed_extraskeletal_myxoid_chondrosarcoma":   ("unconfirmed_extraskeletal_myxoid_chondrosarcoma", False),
    "unconfirmed_pleomorphic rhabdomyosarcoma":          ("unconfirmed_pleomorphic_rhabdomyosarcoma", False),
    "unconfirmed_intramuscular_myxoma":                  ("unconfirmed_intramuscular_myxoma",     False),
    "unconfirmed_malignant_peripheral_nerve_sheath_tumour": ("unconfirmed_mpnst",                 False),
    "unconfirmed_Giant cell tumour of bone":             ("unconfirmed_giant_cell_tumour_of_bone", False),
    "unconfirmed_ewing_sarcoma":                         ("unconfirmed_ewing_sarcoma",            False),
    "unconfirmed_endometrial_stromal_sarcoma_low_grade": ("unconfirmed_endometrial_stromal_sarcoma_lg", False),
    "unconfirmed_angiosarcoma":                          ("unconfirmed_angiosarcoma",             False),
    "unconfirmed_leiomyosarcoma_ASMT":                   ("unconfirmed_leiomyosarcoma_asmt",      False),
    "unconmfirmed_clear_cell_sarcoma_of_soft_parts":     ("unconfirmed_clear_cell_sarcoma",       False),
}

# Human-readable display names for the dashboard
DISPLAY_NAMES: dict[str, str] = {
    "leiomyosarcoma_hg":                    "Leiomyosarcoma HG",
    "leiomyosarcoma_asmt":                  "Leiomyosarcoma ASMT",
    "uterine_leiomyosarcoma_hg":            "Uterine Leiomyosarcoma HG",
    "uterine_leiomyosarcoma_asmt":          "Uterine Leiomyosarcoma ASMT",
    "uterine_leiomyosarcoma":               "Uterine Leiomyosarcoma (unspecified)",
    "uterine_tumour_uncertain_malignant_potential": "UTUMP",
    "ewing_sarcoma":                        "Ewing Sarcoma",
    "fibromatosis_desmoid":                 "Fibromatosis (Desmoid)",
    "intramuscular_myxoma":                 "Intramuscular Myxoma",
    "solitary_fibrous_tumour":              "Solitary Fibrous Tumour",
    "pleomorphic_dermal_sarcoma":           "Pleomorphic Dermal Sarcoma / AFX",
    "synovial_sarcoma":                     "Synovial Sarcoma",
    "embryonal_rhabdomyosarcoma":           "Embryonal Rhabdomyosarcoma",
    "dermatofibrosarcoma_protuberans":      "Dermatofibrosarcoma Protuberans",
    "dedifferentiated_liposarcoma":         "Dedifferentiated Liposarcoma",
    "nodular_fasciitis":                    "Nodular Fasciitis",
    "neurofibroma":                         "Neurofibroma",
    "angiosarcoma":                         "Angiosarcoma",
    "alveolar_rhabdomyosarcoma":            "Alveolar Rhabdomyosarcoma",
    "low_grade_fibromyxoid_sarcoma":        "Low Grade Fibromyxoid Sarcoma",
    "granular_cell_tumour":                 "Granular Cell Tumour",
    "clear_cell_sarcoma":                   "Clear Cell Sarcoma of Soft Parts",
    "mpnst":                                "Malignant Peripheral Nerve Sheath Tumour",
    "epithelioid_sarcoma":                  "Epithelioid Sarcoma",
    "extraskeletal_myxoid_chondrosarcoma":  "Extraskeletal Myxoid Chondrosarcoma",
    "giant_cell_tumour_of_bone":            "Giant Cell Tumour of Bone",
    "hybrid_nerve_sheath_tumour":           "Hybrid Nerve Sheath Tumour",
    "spindle_cell_pleomorphic_lipoma":      "Spindle Cell / Pleomorphic Lipoma",
    "pleomorphic_liposarcoma":              "Pleomorphic Liposarcoma",
    "glomus_tumour":                        "Glomus Tumour",
    "myopericytoma":                        "Myopericytoma",
    "pleomorphic_rhabdomyosarcoma":         "Pleomorphic Rhabdomyosarcoma",
    "dermatofibroma":                       "Dermatofibroma / Benign Fibrous Histiocytoma",
    "gastrointestinal_stromal_tumour":      "Gastrointestinal Stromal Tumour",
    "alveolar_soft_part_sarcoma":           "Alveolar Soft Part Sarcoma",
    "myxoid_liposarcoma":                   "Myxoid Liposarcoma",
    "chondrosarcoma":                       "Chondrosarcoma",
    "merkel_cell_carcinoma":                "Merkel Cell Carcinoma",
    "undifferentiated_sarcoma":             "Undifferentiated Sarcoma",
    "perineurioma":                         "Perineurioma",
    "neuroblastoma":                        "Neuroblastoma",
    "tenosynovial_giant_cell_tumour":       "Tenosynovial Giant Cell Tumour",
    "malignant_melanoma":                   "Malignant Melanoma",
    "inflammatory_myofibroblastic_tumour":  "Inflammatory Myofibroblastic Tumour",
    "schwannoma":                           "Schwannoma",
    "chordoma":                             "Chordoma",
    "cic_rearranged_sarcoma":               "CIC-Rearranged Sarcoma",
    "alk_rearranged_tumour":                "ALK-Rearranged Tumour",
    "chondroblastoma":                      "Chondroblastoma",
    "fibromatosis_superficial":             "Superficial Fibromatosis",
    "myxofibrosarcoma":                     "Myxofibrosarcoma",
    "epithelioid_haemangioendothelioma":    "Epithelioid Haemangioendothelioma",
    "hibernoma":                            "Hibernoma",
    "chondromyxoid_fibroma":                "Chondromyxoid Fibroma",
    "central_chondrosarcoma":               "Central Chondrosarcoma",
    "dedifferentiated_chondrosarcoma":      "Dedifferentiated Chondrosarcoma",
    "inflammatory_rhabdomyoblastic_tumour": "Inflammatory Rhabdomyoblastic Tumour",
    "spindle_cell_imt":                     "Spindle Cell Tumour / IMT",
    "plasma_cell_tumour":                   "Plasma Cell Tumour",
    "endometrial_stromal_sarcoma_lg":       "Endometrial Stromal Sarcoma (Low Grade)",
    "bcor_rearranged_sarcoma":              "BCOR-Rearranged Sarcoma",
}


def get_canonical(raw_label: str) -> tuple[str, bool]:
    """Return (canonical_name, is_confirmed) for a raw GT diagnosis label."""
    entry = DIAG_MAP.get(raw_label.strip())
    if entry:
        return entry
    # Fallback: lowercase + underscore normalisation
    canon = raw_label.strip().lower()
    canon = re.sub(r"[^a-z0-9]+", "_", canon).strip("_")
    confirmed = not canon.startswith("unconfirmed")
    return canon, confirmed


# ── Literature profiles ────────────────────────────────────────────────────────
# Format: (canonical_diagnosis, marker_canonical, expected_outcome,
#          approx_frequency_pct, citation)
# Sources: WHO Classification of Soft Tissue and Bone Tumours (5th ed., 2020);
#          Fletcher CDM et al., Diagnostic Histopathology of Tumors (5th ed.);
#          Hornick JL, Practical Soft Tissue Pathology (2nd ed., 2019).
# Frequencies are approximate population-level estimates.

LITERATURE: list[tuple] = [
    # Leiomyosarcoma HG / ASMT
    ("leiomyosarcoma_hg",  "sma",       "positive",  95, "WHO 5th ed. 2020"),
    ("leiomyosarcoma_hg",  "desmin",    "positive",  85, "WHO 5th ed. 2020"),
    ("leiomyosarcoma_hg",  "caldesmon", "positive",  80, "Hornick 2019"),
    ("leiomyosarcoma_hg",  "s100",      "negative",  90, "WHO 5th ed. 2020"),
    ("leiomyosarcoma_hg",  "cd34",      "negative",  85, "WHO 5th ed. 2020"),
    ("leiomyosarcoma_hg",  "mnf116",    "negative",  90, "Fletcher 2020"),
    ("leiomyosarcoma_asmt","sma",       "positive",  95, "WHO 5th ed. 2020"),
    ("leiomyosarcoma_asmt","desmin",    "positive",  85, "WHO 5th ed. 2020"),
    ("leiomyosarcoma_asmt","caldesmon", "positive",  80, "Hornick 2019"),
    ("leiomyosarcoma_asmt","s100",      "negative",  90, "WHO 5th ed. 2020"),
    # Uterine leiomyosarcoma HG
    ("uterine_leiomyosarcoma_hg","sma",      "positive",  95, "WHO 5th ed. 2020"),
    ("uterine_leiomyosarcoma_hg","desmin",   "positive",  85, "WHO 5th ed. 2020"),
    ("uterine_leiomyosarcoma_hg","caldesmon","positive",  80, "WHO 5th ed. 2020"),
    ("uterine_leiomyosarcoma_hg","er",       "positive",  40, "Guntupalli 2009"),
    ("uterine_leiomyosarcoma_hg","pr",       "positive",  35, "Guntupalli 2009"),
    ("uterine_leiomyosarcoma_hg","p16",      "positive",  75, "Crum 2016"),
    ("uterine_leiomyosarcoma_hg","p53",      "positive",  60, "Crum 2016"),
    # Solitary fibrous tumour
    ("solitary_fibrous_tumour","cd34",   "positive",  95, "WHO 5th ed. 2020"),
    ("solitary_fibrous_tumour","stat6",  "positive",  95, "Doyle 2012"),
    ("solitary_fibrous_tumour","cd99",   "positive",  70, "Fletcher 2020"),
    ("solitary_fibrous_tumour","bcl2",   "positive",  65, "Fletcher 2020"),
    ("solitary_fibrous_tumour","sma",    "negative",  80, "WHO 5th ed. 2020"),
    ("solitary_fibrous_tumour","desmin", "negative",  90, "WHO 5th ed. 2020"),
    ("solitary_fibrous_tumour","s100",   "negative",  80, "WHO 5th ed. 2020"),
    # Synovial sarcoma
    ("synovial_sarcoma","tle1",   "positive",  90, "Terry 2007"),
    ("synovial_sarcoma","ema",    "positive",  70, "WHO 5th ed. 2020"),
    ("synovial_sarcoma","panck",  "positive",  60, "WHO 5th ed. 2020"),
    ("synovial_sarcoma","bcl2",   "positive",  75, "Fletcher 2020"),
    ("synovial_sarcoma","cd99",   "positive",  65, "Fletcher 2020"),
    ("synovial_sarcoma","s100",   "negative",  80, "WHO 5th ed. 2020"),
    ("synovial_sarcoma","cd34",   "negative",  85, "WHO 5th ed. 2020"),
    # Dermatofibrosarcoma protuberans
    ("dermatofibrosarcoma_protuberans","cd34",   "positive",  95, "WHO 5th ed. 2020"),
    ("dermatofibrosarcoma_protuberans","s100",   "negative",  90, "WHO 5th ed. 2020"),
    ("dermatofibrosarcoma_protuberans","desmin", "negative",  95, "Fletcher 2020"),
    ("dermatofibrosarcoma_protuberans","ema",    "negative",  90, "Fletcher 2020"),
    # Ewing sarcoma
    ("ewing_sarcoma","cd99",   "positive",  95, "WHO 5th ed. 2020"),
    ("ewing_sarcoma","fli1",   "positive",  70, "WHO 5th ed. 2020"),
    ("ewing_sarcoma","erg",    "positive",  25, "WHO 5th ed. 2020"),
    ("ewing_sarcoma","desmin", "negative",  90, "WHO 5th ed. 2020"),
    ("ewing_sarcoma","sma",    "negative",  90, "WHO 5th ed. 2020"),
    # Fibromatosis (desmoid)
    ("fibromatosis_desmoid","beta-catenin","positive",  80, "Bhattacharya 2005"),
    ("fibromatosis_desmoid","sma",          "positive",  60, "Fletcher 2020"),
    ("fibromatosis_desmoid","cd34",         "negative",  75, "WHO 5th ed. 2020"),
    ("fibromatosis_desmoid","s100",         "negative",  95, "WHO 5th ed. 2020"),
    ("fibromatosis_desmoid","desmin",       "negative",  85, "WHO 5th ed. 2020"),
    # Granular cell tumour
    ("granular_cell_tumour","s100",   "positive", 100, "WHO 5th ed. 2020"),
    ("granular_cell_tumour","sox10",  "positive",  95, "Hornick 2019"),
    ("granular_cell_tumour","cd68",   "positive",  85, "Fletcher 2020"),
    ("granular_cell_tumour","hmb45",  "negative",  90, "WHO 5th ed. 2020"),
    ("granular_cell_tumour","desmin", "negative",  95, "WHO 5th ed. 2020"),
    # Angiosarcoma
    ("angiosarcoma","cd31",  "positive",  95, "WHO 5th ed. 2020"),
    ("angiosarcoma","erg",   "positive",  85, "Hornick 2019"),
    ("angiosarcoma","cd34",  "positive",  75, "WHO 5th ed. 2020"),
    ("angiosarcoma","fli1",  "positive",  75, "WHO 5th ed. 2020"),
    ("angiosarcoma","s100",  "negative",  85, "WHO 5th ed. 2020"),
    # Alveolar rhabdomyosarcoma
    ("alveolar_rhabdomyosarcoma","desmin",  "positive",  95, "WHO 5th ed. 2020"),
    ("alveolar_rhabdomyosarcoma","myod1",   "positive",  95, "WHO 5th ed. 2020"),
    ("alveolar_rhabdomyosarcoma","myogenin","positive",  90, "Dias 2000"),
    ("alveolar_rhabdomyosarcoma","sma",     "positive",  50, "WHO 5th ed. 2020"),
    ("alveolar_rhabdomyosarcoma","s100",    "negative",  90, "WHO 5th ed. 2020"),
    # Embryonal rhabdomyosarcoma
    ("embryonal_rhabdomyosarcoma","desmin",  "positive",  90, "WHO 5th ed. 2020"),
    ("embryonal_rhabdomyosarcoma","myod1",   "positive",  85, "WHO 5th ed. 2020"),
    ("embryonal_rhabdomyosarcoma","myogenin","positive",  55, "Dias 2000"),
    ("embryonal_rhabdomyosarcoma","sma",     "positive",  45, "WHO 5th ed. 2020"),
    # Clear cell sarcoma
    ("clear_cell_sarcoma","s100",    "positive", 100, "WHO 5th ed. 2020"),
    ("clear_cell_sarcoma","sox10",   "positive",  95, "Hornick 2019"),
    ("clear_cell_sarcoma","hmb45",   "positive",  75, "WHO 5th ed. 2020"),
    ("clear_cell_sarcoma","melan-a", "positive",  70, "WHO 5th ed. 2020"),
    ("clear_cell_sarcoma","ae1/ae3", "negative",  95, "WHO 5th ed. 2020"),
    ("clear_cell_sarcoma","desmin",  "negative",  95, "WHO 5th ed. 2020"),
    # MPNST
    ("mpnst","sox10",  "positive",  50, "WHO 5th ed. 2020"),
    ("mpnst","s100",   "positive",  50, "WHO 5th ed. 2020"),
    ("mpnst","hmb45",  "negative",  90, "WHO 5th ed. 2020"),
    ("mpnst","melan-a","negative",  90, "WHO 5th ed. 2020"),
    ("mpnst","cd34",   "negative",  70, "Fletcher 2020"),
    # Gastrointestinal stromal tumour
    ("gastrointestinal_stromal_tumour","cd117","positive",  95, "WHO 5th ed. 2020"),
    ("gastrointestinal_stromal_tumour","dog1", "positive",  95, "West 2004"),
    ("gastrointestinal_stromal_tumour","cd34", "positive",  70, "WHO 5th ed. 2020"),
    ("gastrointestinal_stromal_tumour","sma",  "negative",  60, "WHO 5th ed. 2020"),
    ("gastrointestinal_stromal_tumour","desmin","negative", 90, "WHO 5th ed. 2020"),
    ("gastrointestinal_stromal_tumour","s100", "negative",  85, "WHO 5th ed. 2020"),
    # Dedifferentiated liposarcoma
    ("dedifferentiated_liposarcoma","mdm2",  "positive",  95, "WHO 5th ed. 2020"),
    ("dedifferentiated_liposarcoma","cdk4",  "positive",  90, "Tap 2015"),
    ("dedifferentiated_liposarcoma","p16",   "positive",  85, "Tap 2015"),
    ("dedifferentiated_liposarcoma","s100",  "negative",  90, "WHO 5th ed. 2020"),
    # Low grade fibromyxoid sarcoma
    ("low_grade_fibromyxoid_sarcoma","muc4", "positive",  95, "Doyle 2011"),
    ("low_grade_fibromyxoid_sarcoma","ema",  "positive",  30, "WHO 5th ed. 2020"),
    ("low_grade_fibromyxoid_sarcoma","s100", "negative",  90, "WHO 5th ed. 2020"),
    ("low_grade_fibromyxoid_sarcoma","cd34", "negative",  70, "WHO 5th ed. 2020"),
    # Epithelioid sarcoma
    ("epithelioid_sarcoma","ema",   "positive",  95, "WHO 5th ed. 2020"),
    ("epithelioid_sarcoma","panck", "positive",  85, "WHO 5th ed. 2020"),
    ("epithelioid_sarcoma","cd34",  "positive",  50, "WHO 5th ed. 2020"),
    ("epithelioid_sarcoma","ini1",  "negative",  95, "Hornick 2019"),
    ("epithelioid_sarcoma","s100",  "negative",  85, "WHO 5th ed. 2020"),
    # Nodular fasciitis
    ("nodular_fasciitis","sma",   "positive",  95, "WHO 5th ed. 2020"),
    ("nodular_fasciitis","desmin","negative",  80, "Fletcher 2020"),
    ("nodular_fasciitis","s100",  "negative",  90, "Fletcher 2020"),
    ("nodular_fasciitis","cd34",  "negative",  75, "Fletcher 2020"),
    # Neurofibroma
    ("neurofibroma","s100",   "positive", 100, "WHO 5th ed. 2020"),
    ("neurofibroma","sox10",  "positive",  95, "Hornick 2019"),
    ("neurofibroma","hmb45",  "negative",  95, "WHO 5th ed. 2020"),
    ("neurofibroma","desmin", "negative",  95, "WHO 5th ed. 2020"),
    # Pleomorphic liposarcoma
    ("pleomorphic_liposarcoma","mdm2",  "positive",  70, "WHO 5th ed. 2020"),
    ("pleomorphic_liposarcoma","cdk4",  "positive",  65, "WHO 5th ed. 2020"),
    ("pleomorphic_liposarcoma","s100",  "negative",  60, "WHO 5th ed. 2020"),
    # Myxoid liposarcoma
    ("myxoid_liposarcoma","s100",  "positive",  60, "WHO 5th ed. 2020"),
    ("myxoid_liposarcoma","cd34",  "positive",  40, "WHO 5th ed. 2020"),
    ("myxoid_liposarcoma","sma",   "negative",  85, "WHO 5th ed. 2020"),
    ("myxoid_liposarcoma","desmin","negative",  90, "WHO 5th ed. 2020"),
    # Spindle cell / pleomorphic lipoma
    ("spindle_cell_pleomorphic_lipoma","cd34","positive",  95, "WHO 5th ed. 2020"),
    ("spindle_cell_pleomorphic_lipoma","s100","negative",  80, "WHO 5th ed. 2020"),
    ("spindle_cell_pleomorphic_lipoma","sma", "negative",  85, "WHO 5th ed. 2020"),
    # Glomus tumour
    ("glomus_tumour","sma",      "positive",  95, "WHO 5th ed. 2020"),
    ("glomus_tumour","caldesmon","positive",  80, "Folpe 2001"),
    ("glomus_tumour","desmin",   "negative",  85, "Fletcher 2020"),
    ("glomus_tumour","s100",     "negative",  90, "Fletcher 2020"),
    # Myopericytoma
    ("myopericytoma","sma",   "positive",  95, "WHO 5th ed. 2020"),
    ("myopericytoma","desmin","negative",  80, "Hornick 2019"),
    ("myopericytoma","s100",  "negative",  85, "Hornick 2019"),
    # Giant cell tumour of bone
    ("giant_cell_tumour_of_bone","p63",  "positive",  85, "WHO 5th ed. 2020"),
    ("giant_cell_tumour_of_bone","cd68", "positive",  95, "WHO 5th ed. 2020"),
    ("giant_cell_tumour_of_bone","s100", "negative",  90, "WHO 5th ed. 2020"),
    # Intramuscular myxoma
    ("intramuscular_myxoma","sma",  "positive",  40, "WHO 5th ed. 2020"),
    ("intramuscular_myxoma","s100", "negative",  85, "Fletcher 2020"),
    ("intramuscular_myxoma","cd34", "negative",  75, "Fletcher 2020"),
    # Merkel cell carcinoma
    ("merkel_cell_carcinoma","ck20",          "positive",  95, "WHO 5th ed. 2020"),
    ("merkel_cell_carcinoma","cam5.2",        "positive",  90, "WHO 5th ed. 2020"),
    ("merkel_cell_carcinoma","synaptophysin", "positive",  80, "WHO 5th ed. 2020"),
    ("merkel_cell_carcinoma","chromogranin",  "positive",  70, "WHO 5th ed. 2020"),
    ("merkel_cell_carcinoma","ttf1",          "negative",  95, "WHO 5th ed. 2020"),
    # Pleomorphic rhabdomyosarcoma
    ("pleomorphic_rhabdomyosarcoma","desmin",  "positive",  80, "WHO 5th ed. 2020"),
    ("pleomorphic_rhabdomyosarcoma","myod1",   "positive",  60, "WHO 5th ed. 2020"),
    ("pleomorphic_rhabdomyosarcoma","myogenin","positive",  40, "WHO 5th ed. 2020"),
    # Alveolar soft part sarcoma
    ("alveolar_soft_part_sarcoma","tfe3",  "positive",  95, "WHO 5th ed. 2020"),
    ("alveolar_soft_part_sarcoma","myod1", "positive",  85, "Ladanyi 2001"),
    ("alveolar_soft_part_sarcoma","s100",  "negative",  70, "WHO 5th ed. 2020"),
    # Schwannoma
    ("schwannoma","s100",  "positive", 100, "WHO 5th ed. 2020"),
    ("schwannoma","sox10", "positive",  95, "Hornick 2019"),
    ("schwannoma","sma",   "negative",  85, "Fletcher 2020"),
    # Inflammatory myofibroblastic tumour
    ("inflammatory_myofibroblastic_tumour","alk", "positive",  50, "WHO 5th ed. 2020"),
    ("inflammatory_myofibroblastic_tumour","sma", "positive",  70, "WHO 5th ed. 2020"),
    ("inflammatory_myofibroblastic_tumour","desmin","positive",50, "WHO 5th ed. 2020"),
    # Dermatofibroma
    ("dermatofibroma","cd68","positive",  80, "Fletcher 2020"),
    ("dermatofibroma","cd34","negative",  80, "Fletcher 2020"),
    # Tenosynovial giant cell tumour
    ("tenosynovial_giant_cell_tumour","cd68",    "positive",  85, "WHO 5th ed. 2020"),
    ("tenosynovial_giant_cell_tumour","desmin",  "negative",  85, "Fletcher 2020"),
]


# ── Schema ─────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS diagnoses (
    canonical_diagnosis   TEXT PRIMARY KEY,
    display_name          TEXT,
    n_gt_cases            INTEGER,
    is_confirmed          INTEGER DEFAULT 1,
    include_in_analysis   INTEGER DEFAULT 0,
    low_count             INTEGER DEFAULT 0,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS cases (
    patient_id            TEXT PRIMARY KEY,
    raw_diagnosis         TEXT,
    canonical_diagnosis   TEXT REFERENCES diagnoses(canonical_diagnosis),
    subclassification     TEXT,
    has_gt                INTEGER DEFAULT 1,
    ihc_text              TEXT
);

CREATE TABLE IF NOT EXISTS ihc_markers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            TEXT REFERENCES cases(patient_id),
    marker                TEXT,
    outcome               TEXT CHECK(outcome IN ('positive','negative','equivocal')),
    source                TEXT CHECK(source IN ('pipeline','gt'))
);

CREATE INDEX IF NOT EXISTS idx_ihc_markers_pid    ON ihc_markers(patient_id);
CREATE INDEX IF NOT EXISTS idx_ihc_markers_marker ON ihc_markers(marker);
CREATE INDEX IF NOT EXISTS idx_ihc_markers_diag   ON ihc_markers(patient_id, marker, source);

CREATE TABLE IF NOT EXISTS ki67_results (
    patient_id            TEXT REFERENCES cases(patient_id),
    ki67_pct              REAL,
    PRIMARY KEY (patient_id)
);

CREATE TABLE IF NOT EXISTS literature (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_diagnosis   TEXT REFERENCES diagnoses(canonical_diagnosis),
    marker                TEXT,
    expected_outcome      TEXT CHECK(expected_outcome IN ('positive','negative','variable')),
    typical_frequency_pct REAL,
    citation              TEXT
);

CREATE INDEX IF NOT EXISTS idx_literature_diag ON literature(canonical_diagnosis);
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def clean_subclassification(raw: str) -> str | None:
    """Return cleaned, title-cased subclassification or None if not applicable."""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip().replace("\xa0", " ")
    _NULL = re.compile(r"^(not\s+required|not\s+applicable|na|n/a|nil|leiomyosarcoma)$",
                       re.IGNORECASE)
    if _NULL.match(s):
        return None
    # Normalise capitalisation variants: use title case for single-word or
    # known short labels; preserve original casing for molecular fusion strings.
    if re.match(r"^[A-Z]{2,}[\d:]+", s):  # molecular fusion e.g. HMGA2::LSM12
        return s
    return s.title()  # Leiomyoma, Cutaneous Leiomyosarcoma, STUMP -> Stump (acceptable)


# ── Build ──────────────────────────────────────────────────────────────────────

def build_database():
    print("Building IHC database...")

    # Load source data
    preds = pd.read_csv(DATA_DIR / "hybrid_predictions.csv", dtype=str).fillna("")
    gt_df = pd.read_csv(DATA_DIR / "cleaned_ground_truth.csv", dtype=str).fillna("")

    # Load LMS subclassification lookup (from RNOH masterfiles)
    lms_path = DATA_DIR / "lms_subclassification.csv"
    if lms_path.exists():
        lms_raw = pd.read_csv(lms_path, dtype=str).fillna("")
        # Deduplicate: keep first entry per patient_id
        lms_raw = lms_raw.drop_duplicates(subset=["patient_id"], keep="first")
        lms_sub = lms_raw.set_index("patient_id")
    else:
        print("  Warning: lms_subclassification.csv not found; LMS re-labelling skipped.")
        lms_sub = pd.DataFrame(columns=["canonical_diagnosis", "subclassification"])
        lms_sub.index.name = "patient_id"

    preds["patient_id"] = preds["patient_id"].str.strip()
    gt_df["patient_id"] = gt_df["patient_id"].str.strip()

    # Build patient -> (canonical_diag, raw_diag, subclass, ihc_text) table
    gt_lookup = gt_df.set_index("patient_id")[["diagnosis", "ihc_text"]]

    # Connect: build to a temp file, then copy over (avoids journal-lock issues)
    import shutil, tempfile
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    import os; os.close(tmp_fd)
    con = sqlite3.connect(tmp_path)
    cur = con.cursor()
    cur.executescript(DDL)

    # ── 1. Ingest cases + ihc_markers ─────────────────────────────────────────
    case_rows   = []
    marker_rows = []
    ki67_rows   = []
    diag_counts: dict[str, int] = {}
    gt_pids: set[str] = set()

    for _, row in preds.iterrows():
        pid = row["patient_id"]
        raw_diag = row.get("diagnosis", "")
        gt_pids.add(pid)

        # LMS re-labelling: masterfile AD overrides GT label
        if pid in lms_sub.index:
            canon = lms_sub.at[pid, "canonical_diagnosis"]
            subclass = clean_subclassification(lms_sub.at[pid, "subclassification"])
        else:
            canon, _ = get_canonical(raw_diag)
            subclass = None

        diag_counts[canon] = diag_counts.get(canon, 0) + 1
        ihc_text = row.get("ihc_text", "")

        case_rows.append((pid, raw_diag, canon, subclass, 1, ihc_text))

        # Pipeline predictions -> ihc_markers (source='pipeline')
        for marker in parse_and_normalise(row.get("pred_positive", "")):
            if row.get("equivocal", "") and marker in row.get("equivocal", ""):
                marker_rows.append((pid, marker, "equivocal", "pipeline"))
            else:
                marker_rows.append((pid, marker, "positive", "pipeline"))
        for marker in parse_and_normalise(row.get("pred_negative", "")):
            marker_rows.append((pid, marker, "negative", "pipeline"))
        # Equivocal markers not already in pred_positive
        for marker in parse_and_normalise(row.get("equivocal", "")):
            marker_rows.append((pid, marker, "equivocal", "pipeline"))

        # GT markers -> ihc_markers (source='gt')
        for marker in parse_and_normalise(row.get("gt_positive", "")):
            marker_rows.append((pid, marker, "positive", "gt"))
        for marker in parse_and_normalise(row.get("gt_negative", "")):
            marker_rows.append((pid, marker, "negative", "gt"))

        # Ki-67
        ki67 = row.get("ki67_index", "").strip()
        if ki67:
            try:
                ki67_rows.append((pid, float(ki67)))
            except ValueError:
                pass

    cur.executemany(
        "INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?,?)",
        case_rows
    )
    cur.executemany(
        "INSERT INTO ihc_markers (patient_id, marker, outcome, source) VALUES (?,?,?,?)",
        marker_rows
    )
    cur.executemany(
        "INSERT OR IGNORE INTO ki67_results VALUES (?,?)",
        ki67_rows
    )
    print(f"  GT: {len(case_rows)} cases, {len(marker_rows)} marker records, "
          f"{len(ki67_rows)} Ki-67 records")

    # ── 1b. No-immuno predictions (pipeline-only, no GT markers) ─────────────
    ni_path = DATA_DIR / "no_immuno_predictions.csv"
    if ni_path.exists():
        ni_preds = pd.read_csv(ni_path, dtype=str).fillna("")
        ni_preds["patient_id"] = ni_preds["patient_id"].str.strip()

        ni_case_rows   = []
        ni_marker_rows = []
        ni_ki67_rows   = []

        for _, row in ni_preds.iterrows():
            pid = row["patient_id"]
            if pid in gt_pids:
                continue  # already in GT set, skip

            raw_diag = row.get("diagnosis", "")
            canon, _ = get_canonical(raw_diag)
            subclass = None

            diag_counts[canon] = diag_counts.get(canon, 0) + 1
            ihc_text = row.get("ihc_text", "")

            ni_case_rows.append((pid, raw_diag, canon, subclass, 0, ihc_text))

            for marker in parse_and_normalise(row.get("pred_positive", "")):
                if row.get("equivocal", "") and marker in row.get("equivocal", ""):
                    ni_marker_rows.append((pid, marker, "equivocal", "pipeline"))
                else:
                    ni_marker_rows.append((pid, marker, "positive", "pipeline"))
            for marker in parse_and_normalise(row.get("pred_negative", "")):
                ni_marker_rows.append((pid, marker, "negative", "pipeline"))
            for marker in parse_and_normalise(row.get("equivocal", "")):
                ni_marker_rows.append((pid, marker, "equivocal", "pipeline"))

            ki67 = row.get("ki67_index", "").strip()
            if ki67:
                try:
                    ni_ki67_rows.append((pid, float(ki67)))
                except ValueError:
                    pass

        cur.executemany(
            "INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?,?)",
            ni_case_rows
        )
        cur.executemany(
            "INSERT INTO ihc_markers (patient_id, marker, outcome, source) VALUES (?,?,?,?)",
            ni_marker_rows
        )
        cur.executemany(
            "INSERT OR IGNORE INTO ki67_results VALUES (?,?)",
            ni_ki67_rows
        )
        print(f"  No-immuno: {len(ni_case_rows)} cases, {len(ni_marker_rows)} marker records, "
              f"{len(ni_ki67_rows)} Ki-67 records")
    else:
        print("  No-immuno predictions not found; "
              "run `python 05_hybrid_pipeline.py --no-immuno` to generate them")

    # ── 2. Diagnoses table ────────────────────────────────────────────────────
    diag_rows = []
    for canon, n in sorted(diag_counts.items(), key=lambda x: -x[1]):
        confirmed = not canon.startswith("unconfirmed")
        include   = confirmed and (n >= MIN_CASES_ANALYSIS)
        low_cnt   = n < MIN_CASES_ANALYSIS
        display   = DISPLAY_NAMES.get(canon, canon.replace("_", " ").title())
        notes     = None
        if canon in ("solitary_fibrous_tumour",):
            notes = "GT annotation incomplete: only 1 positive marker recorded per case"
        elif canon == "mpnst":
            notes = "GT annotation incomplete: most cases have 0 positives in GT"
        diag_rows.append((canon, display, n, int(confirmed), int(include), int(low_cnt), notes))

    cur.executemany(
        "INSERT OR REPLACE INTO diagnoses VALUES (?,?,?,?,?,?,?)",
        diag_rows
    )
    print(f"  Inserted {len(diag_rows)} diagnoses "
          f"({sum(1 for _,_,n,c,i,_,_ in diag_rows if i)} included in analysis)")

    # ── 3. Literature profiles ────────────────────────────────────────────────
    lit_rows = []
    for diag, marker, outcome, freq, citation in LITERATURE:
        canon_marker = normalise_marker(marker)
        lit_rows.append((diag, canon_marker, outcome, freq, citation))

    cur.executemany(
        "INSERT INTO literature (canonical_diagnosis, marker, expected_outcome, "
        "typical_frequency_pct, citation) VALUES (?,?,?,?,?)",
        lit_rows
    )
    print(f"  Inserted {len(lit_rows)} literature profile entries")

    con.commit()
    con.close()
    # Copy temp db to final path
    shutil.copy2(tmp_path, DB_PATH)
    os.unlink(tmp_path)
    # Remove any leftover SQLite auxiliary files (journal, WAL, shm) that could
    # cause SQLite to roll back the freshly written database on next open.
    for ext in ("-journal", "-wal", "-shm"):
        aux = Path(str(DB_PATH) + ext)
        if aux.exists():
            aux.unlink()
            print(f"  Removed stale auxiliary file: {aux.name}")
    print(f"\nDatabase saved: {DB_PATH}")


# ── Stats ──────────────────────────────────────────────────────────────────────

def print_stats():
    if not DB_PATH.exists():
        print("Database not found. Run with --build first.")
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    print(f"\n{'=' * 60}")
    print("DATABASE SUMMARY")
    print(f"{'=' * 60}")

    for table in ("cases", "ihc_markers", "ki67_results", "diagnoses", "literature"):
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<20}: {n:>6} rows")

    print(f"\nDiagnoses included in analysis (n >= {MIN_CASES_ANALYSIS}, confirmed):")
    rows = cur.execute(
        "SELECT canonical_diagnosis, display_name, n_gt_cases, notes "
        "FROM diagnoses WHERE include_in_analysis = 1 "
        "ORDER BY n_gt_cases DESC"
    ).fetchall()
    print(f"  {'Canonical':<45} {'N':>5}  Notes")
    print("  " + "-" * 60)
    for canon, display, n, notes in rows:
        note_str = f"  [{notes[:40]}...]" if notes else ""
        print(f"  {canon:<45} {n:>5}{note_str}")

    print(f"\nReserved diagnoses (unconfirmed or n < {MIN_CASES_ANALYSIS}):")
    rows = cur.execute(
        "SELECT canonical_diagnosis, n_gt_cases, is_confirmed "
        "FROM diagnoses WHERE include_in_analysis = 0 "
        "ORDER BY n_gt_cases DESC"
    ).fetchall()
    for canon, n, confirmed in rows:
        tag = "(unconfirmed)" if not confirmed else f"(n={n})"
        print(f"  {canon:<55} {tag}")

    print(f"\nTop 10 most frequent markers (pipeline predictions):")
    rows = cur.execute("""
        SELECT m.marker, m.outcome, COUNT(*) as cnt
        FROM ihc_markers m
        JOIN cases c ON m.patient_id = c.patient_id
        JOIN diagnoses d ON c.canonical_diagnosis = d.canonical_diagnosis
        WHERE m.source = 'pipeline' AND d.include_in_analysis = 1
        GROUP BY m.marker, m.outcome
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    for marker, outcome, cnt in rows:
        print(f"  {marker:<20} {outcome:<10} {cnt:>5}")

    print(f"\nLMS subclassification breakdown:")
    rows = cur.execute("""
        SELECT canonical_diagnosis, subclassification, COUNT(*) as n
        FROM cases
        WHERE canonical_diagnosis LIKE '%leiomyosarcoma%'
           OR canonical_diagnosis LIKE '%utump%'
        GROUP BY canonical_diagnosis, subclassification
        ORDER BY canonical_diagnosis, n DESC
    """).fetchall()
    for diag, sub, n in rows:
        sub_str = sub if sub else "(none)"
        print(f"  {diag:<40} {sub_str:<40} n={n}")

    con.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IHC SQLite database builder")
    parser.add_argument("--build",   action="store_true", help="Build (or rebuild) the database")
    parser.add_argument("--stats",   action="store_true", help="Print database summary statistics")
    args = parser.parse_args()

    if not (args.build or args.stats):
        parser.print_help()
        return

    if args.build:
        build_database()
    if args.stats:
        print_stats()


if __name__ == "__main__":
    main()
