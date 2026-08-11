# Development Notes — IHC NLP Project

Fatine Oumlil | MSc Computer Science | Birkbeck, University of London
These notes record key decisions and rationale made during development,
to support the methodology section of the final report.

---

## Project overview

Three-component framework:
1. Hybrid NLP extraction pipeline (rule-based + ClinicalBERT fine-tuning)
2. Structured SQLite database of extracted IHC results
3. Streamlit prototype decision-support dashboard

Data: NHS trust repository. Ground truth = manually curated cases with
positive/negative IHC outcomes split into separate columns.

---

## Data

### Source files
- `reports_ground_truth.xlsx` — annotated cases (columns A, D, E, R, S, T)
- `reports_no_immuno.xlsx` — unannotated reports (columns A, D, E, R)

Column mapping:
- A: Patient ID
- D: Brady Slide ID (secondary case identifier)
- E: Final diagnosis
- R: IHC free text (copy of report section)
- S: Positive IHC markers (comma-separated)
- T: Negative IHC markers (comma-separated)

### Data volumes (after cleaning)
- Ground truth: 5,486 unique patients across 93 diagnoses
- No-immuno: 1,934 unique patients across 60 diagnoses

### Known data quality issues and decisions

**Nil/NA/N/A values (S and T columns)**
Meaning: no markers in that category — cell was not left blank by convention.
Decision: treat as empty list, not missing data.

**x1, x2, x3 placeholders (98 cases in negative column)**
Meaning: anonymous count markers added to track number of immunos performed
without recording the actual marker name. Ground truth unreliable for these cases.
Decision: flag and separate into `flagged_anonymous.csv` for review.
Do not use for pipeline training or evaluation without manual review.

**Duplicate patient IDs (162 in GT, 20 in NI)**
Decision: deduplicate by patient ID, keeping the most complete record
(scored by: IHC text present + total marker count in S and T).

**Unconfirmed diagnoses (prefix: unconfirmed_)**
Meaning: diagnosis not confirmed by IHC or molecular testing.
Decision: retain in dataset with prefix intact. Useful for NLP training
(marker extraction does not depend on diagnosis certainty). Exclude or
separate for dashboard analysis where confirmed diagnosis is required.

**Ki-67**
Special case: positive if >1% proliferation index seen. Often not split
into S/T columns or recorded inconsistently.
Decision: TBD — flag during pipeline development and evaluate separately.

**Focal/patchy positivity language**
Examples: "focally positive", "patchy positivity".
Decision: TBD — define classification rule during rule-based pipeline
development. Likely: classify as positive with a focal flag.

**Referred immuno vs RNOH (in-house) immuno**
Reports sometimes contain results from referring institutions alongside
RNOH in-house results. These can duplicate or conflict.
Convention: RNOH in-house results take precedence.
Phrases indicating in-house: "done in house", "immuno performed at RNOH",
"immuno we have done here".
Decision: flag referred immuno sections during text parsing. Prioritise
in-house results for ground truth labelling. Note impact on dashboard
analysis — duplicated results may inflate marker counts.

**Adjectives in S/T columns**
S and T columns should contain strictly comma-separated marker names.
Some entries include adjectives (e.g. "focal CD34") due to data entry errors.
Decision: strip known adjectives during marker list parsing. Log cases
where non-marker text is detected.

**Marker name inconsistencies**
Same marker may appear under different names across cases.
Matters more for dashboard analysis than NLP extraction.
Decision: build a marker synonym/normalisation dictionary during pipeline
development. Apply during database ingestion.

---

## Pilot strategy

**Pilot diagnosis: leiomyosarcoma HG**
Rationale: most common diagnosis in the ground truth (449 cases after
cleaning, 451 before deduplication). Clean, consistent name. All 451
cases have IHC text; 447 have positives listed; 402 have negatives.

Plan: build and validate the rule-based pipeline on this subset first,
then scale to full dataset for ClinicalBERT fine-tuning.

---

## Pipeline approach

**Hybrid: rule-based first pass + ClinicalBERT second pass**
Rationale: rule-based offers interpretability and requires no labelled data
for the first pass. ClinicalBERT handles complex sentence structures
(negation, multi-marker statements, qualified descriptions).
Reference: Alsentzer et al. [12]; Lee et al. [13].

**Fallback plan**
If ClinicalBERT does not improve on rule-based: use rule-based only,
restrict to most frequent markers. Still produces a functional, evaluable system.

**Extension (if time/accuracy permits)**
Clinical ModernBERT (Lee et al. 2025, arXiv:2504.03964) — longer context
window, more recent pre-training. Locally deployed LLMs (MedLlama2,
MedGemma, DeepSeek-V4) as alternative to encoder-based approach.

---

## Infrastructure

- Language: Python 3
- GPU: available at workplace (for ClinicalBERT fine-tuning)
- Database: SQLite (prototype stage — no server infrastructure required)
- Dashboard: Streamlit

---

## Scripts

| File | Purpose |
|------|---------|
| `01_data_exploration.py` | Load and summarise both raw spreadsheets |
| `02_data_cleaning.py` | Extract key columns, deduplicate, flag issues, produce pilot subset |

---

## To decide / revisit

- Ki-67 classification rule
- Focal/patchy positivity classification
- Whether to include flagged anonymous cases after manual review
- Marker synonym dictionary (build incrementally during pipeline dev)
- Which diagnoses have sufficient cases for dashboard analysis
