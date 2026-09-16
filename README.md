# IHC NLP Pipeline for Soft Tissue Tumour Pathology

MSc Computer Science project — Birkbeck, University of London (UCL-funded)  
Author: Fatine Oumlil

## Overview

An end-to-end NLP pipeline for extracting structured immunohistochemistry (IHC) and molecular findings from free-text histopathology reports, with a prototype decision-support dashboard for soft tissue tumour pathology.

The system has three components:

1. **Hybrid NLP extraction pipeline** (`01` to `05`) — rule-based extraction combined with ClinicalBERT fine-tuning to identify positive and negative IHC marker results from unstructured report text
2. **Structured SQLite database** (`06_database.py`) — normalised schema storing per-case IHC results, molecular findings, and diagnosis metadata
3. **Streamlit dashboard** (`07_dashboard.py`) — interactive prototype for exploring marker positivity rates, molecular alterations, and literature benchmarks by diagnosis

## Structure

```
.
├── 01_data_exploration.py       # Initial dataset analysis
├── 02_data_cleaning.py          # Text preprocessing and normalisation
├── 03_rule_based_pipeline.py    # Rule-based IHC extraction
├── 04_clinicalbert.py           # ClinicalBERT fine-tuning
├── 05_hybrid_pipeline.py        # Combined pipeline
├── 06_database.py               # Database setup and import scripts
├── 07_dashboard.py              # Streamlit dashboard
├── marker_utils.py              # Shared marker normalisation utilities
├── analysis_by_diagnosis.py     # Per-diagnosis analysis scripts
├── literature_references.md     # Literature IHC profiles used for benchmarking
└── NOTES.md                     # Development decisions and methodology notes
```

## Setup

```bash
pip install streamlit plotly pandas openpyxl spacy transformers torch
python -m spacy download en_core_web_sm
```

Run the dashboard:

```bash
streamlit run 07_dashboard.py
```

## Data

Patient data is not included in this repository (NHS Information Governance / GDPR).  
The pipeline expects:
- A SQLite database at `data/ihc_database.db`
- Source report files in `data/`

## Notes

All processing runs on-premise. No patient data is sent to external APIs.
