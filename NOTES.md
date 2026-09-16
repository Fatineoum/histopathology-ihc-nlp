# Development Notes : IHC NLP Project

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
- `reports_ground_truth.xlsx` : annotated cases (columns A, D, E, R, S, T)
- `reports_no_immuno.xlsx` : unannotated reports (columns A, D, E, R)

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
Meaning: no markers in that category : cell was not left blank by convention.
Decision: treat as empty list, not missing data.

**x1, x2, x3 placeholders (97 cases in negative column after deduplication)**
Meaning: anonymous count markers added to track number of immunos performed
without recording the actual marker name. Ground truth unreliable for these cases.
Decision: flagged and separated into `flagged_anonymous.csv`. Excluded from
training and evaluation for now. 5,389 clean cases are sufficient to proceed.
May revisit if more cases are needed at a later stage.

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
Decision: TBD : flag during pipeline development and evaluate separately.

**Focal/patchy positivity language**
Examples: "focally positive", "patchy positivity".
Decision: TBD : define classification rule during rule-based pipeline
development. Likely: classify as positive with a focal flag.

**Referred immuno vs RNOH (in-house) immuno**
Reports sometimes contain results from referring institutions alongside
RNOH in-house results. These can duplicate or conflict.
Convention: RNOH in-house results take precedence.
Phrases indicating in-house: "done in house", "immuno performed at RNOH",
"immuno we have done here".
Decision: flag referred immuno sections during text parsing. Prioritise
in-house results for ground truth labelling. Note impact on dashboard
analysis : duplicated results may inflate marker counts.

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
Clinical ModernBERT (Lee et al. 2025, arXiv:2504.03964) : longer context
window, more recent pre-training. Locally deployed LLMs (MedLlama2,
MedGemma, DeepSeek-V4) as alternative to encoder-based approach.

---

## Infrastructure

- Language: Python 3
- GPU: available at workplace (for ClinicalBERT fine-tuning)
- Database: SQLite (prototype stage : no server infrastructure required)
- Dashboard: Streamlit

---

## Rule-based pipeline results

Evaluated on leiomyosarcoma HG pilot (449 cases, S00157601 excluded).

| | Precision | Recall | F1 |
|--|--|--|--|
| Positive markers | 0.768 | 0.709 | 0.737 |
| Negative markers | 0.845 | 0.864 | 0.854 |
| Overall micro | | | 0.803 |

TP pos=919, FP pos=278, FN pos=378
TP neg=1360, FP neg=250, FN neg=214

Decision: F1 is within the 0.75-0.85 target range. Proceeding to ClinicalBERT
rather than continuing rule refinement (diminishing returns). Key remaining errors
that rules cannot handle:

1. Subordinate clause polarity reversal: "positive for SMA and desmin, whereas
   S100 and caldesmon are negative" (greedy positive pattern captures the
   negative markers in the subordinate clause).
2. Contiguous text without sentence boundaries: "...S100 Caldesmon performed at
   RNOH is positive" (two markers and a source clause merged into one span).
3. Long negative lists following "All of the following are negative:" where the
   structured format parser misses multi-line or non-standard variations.

---

## Results summary

| Pipeline | Positive F1 | Negative F1 | Overall F1 | Notes |
|---|---|---|---|---|
| Rule-based | 0.737 | 0.854 | 0.803 | Pilot, 449 cases, unnormalised |
| Rule-based | 0.568 | 0.767 | 0.699 | Full dataset, 5388 cases, normalised |
| ClinicalBERT | 0.929 | 0.925 | 0.927 | Pilot only, includes training cases |
| ClinicalBERT | 0.867 | 0.872 | **0.870** | Held-out test set, 809 cases, 93 diagnoses |
| ClinicalBERT | 0.912 | 0.904 | 0.907 | Full dataset 5388 cases, normalised (inflated) |
| Hybrid | 0.844 | 0.876 | 0.864 | Full dataset, normalised (inflated) |

**Reportable figures for the final report:**
- Rule-based baseline: F1=0.803 (pilot development metric)
- ClinicalBERT: F1=0.870 (held-out test set, 809 cases, 93 diagnoses; most honest)
- Hybrid vs BERT: hybrid F1 lower than ClinicalBERT alone (0.864 vs 0.907 on full
  dataset). Rule-based supplementation adds false positives that hurt precision
  more than the recall gain compensates. ClinicalBERT is the best extraction engine.
  Hybrid value is in additional clinical outputs: Ki-67 index and equivocal flagging.

**Additional outputs from hybrid pipeline (full dataset):**
- Ki-67 proliferation index extracted: 255 cases
- Equivocal marker flags raised: 117 cases

**Ground truth incompleteness: SFT and MPNST**
Clinical review revealed that the GT annotation for two diagnoses is systematically
incomplete:
- Solitary fibrous tumour (n=302): every case has exactly 1 positive marker in GT,
  296/302 cases have 0 negatives. The full IHC panel was not recorded.
- Malignant peripheral nerve sheath tumour (n=75): 67/75 cases have 0 positives,
  70/75 have only 1 negative. Again, partial annotation only.
This is a data entry convention at the trust, not a pipeline failure. The pipeline
correctly extracts multiple markers from the text, but these appear as FP against
an incomplete GT.

Impact on F1:
- Including SFT + MPNST: hybrid F1 = 0.864 (5388 cases)
- Excluding SFT + MPNST: hybrid F1 = 0.873 (5011 cases)
Both figures should be reported with explanation of the GT limitation.

---

## ClinicalBERT fine-tuning

Script: `04_clinicalbert.py`
Model: Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT, Alsentzer et al. 2019)
Task: token classification with BIO labels (O, B-POS, I-POS, B-NEG, I-NEG)

Labelling strategy (distant supervision):
    Ground truth marker names are searched in the report text using flexible
    regex (allows hyphens/slashes as separators, strips qualifiers such as
    "focal" and "diffuse"). Matched character spans are labelled B-POS/I-POS
    or B-NEG/I-NEG. Positives override negatives on overlap (handles RNOH
    vs referred institution conflicts in the text).

Training setup:
    80/20 train/val split (stratified by random seed 42)
    Batch size: 16 | LR: 2e-5 | Epochs: 5 | Warmup: 10%
    Checkpoint saved when validation token F1 improves.
    Checkpoint excluded from Git (models/ in .gitignore, regeneratable).

Evaluation: case-level set comparison identical to rule-based pipeline,
    so results are directly comparable.

Run:
    python 04_clinicalbert.py --train --eval

---

## Scripts

| File | Purpose |
|------|---------|
| `01_data_exploration.py` | Load and summarise both raw spreadsheets |
| `02_data_cleaning.py` | Extract key columns, deduplicate, flag issues, produce pilot subset |
| `03_rule_based_pipeline.py` | Rule-based IHC extraction; pilot F1=0.803 |
| `04_clinicalbert.py` | ClinicalBERT fine-tuning; full dataset test F1=0.870 |
| `05_hybrid_pipeline.py` | Hybrid: rule-based + ClinicalBERT; adds Ki-67 and equivocal flags |
| `06_database.py` | SQLite database: ingestion, diagnosis normalisation, literature profiles |
| `07_dashboard.py` | Streamlit dashboard: 4 views (panel, literature, LMS, case browser) |
| `marker_utils.py` | Shared normalisation, alias table, Ki-67 extraction |

---

## Excluded cases

**S00157601**: Excluded from pilot evaluation. Report text appears to contain a human data entry error (truncated/incomplete negative list with no explicit keyword), making ground truth unreliable for this case.

---

## Clinical validation of ClinicalBERT errors

150 of 303 error cases reviewed by clinician (Fatine Oumlil).

| Verdict | Count | % |
|---|---|---|
| Pipeline correct | 68 | 45% |
| GT correct, pipeline wrong | 34 | 23% |
| Both partial | 24 | 16% |
| Pipeline correct (normalisation issue only) | 14 | 9% |
| Pipeline partially correct | 5 | 3% |
| Both correct (normalisation issue only) | 3 | 2% |
| GT correct (normalisation issue only) | 2 | 1% |

Key finding: 57% of flagged errors are not genuine model failures (pipeline correct
or normalisation-only issues). True pipeline error rate is lower than F1=0.870 implies.

**Normalisation aliases identified:**

| Variants seen | Canonical form |
|---|---|
| sox-10, sox10 | SOX10 |
| ae1/ae3, ae1/3, ae1-ae3 | AE1/AE3 |
| ini-1, ini1 | INI1 |
| melan-a, melan a | Melan-A |
| h-caldesmon, caldesmon | caldesmon |
| muc-4, muc4 | MUC4 |
| pankeratin mnf116, mnf116 | MNF116 |

**GT data quality issues found:**
- Trailing "." in some GT marker cells (data entry artifact)
- Report-level typos: "CDD99" (should be CD99), "C117" (should be CD117)
  Model correctly extracts what is written; these are source data errors.

**Ki-67 / MIB-1 decision:**
Ki-67 (also known as MIB-1) is a proliferation index reported as a percentage,
not a binary positive/negative marker. Classifying it as positive or negative is
clinically incorrect.
Decision: extract Ki-67/MIB-1 separately as a (marker, percentage) pair.
Strip it from the binary positive/negative classification pipeline.
Store as a dedicated field in the database.

**Focal/patchy positivity:**
Decision: classify as positive (consistent with ground truth convention and
clinical practice for soft tissue tumours).

---

## Design decisions (from clinical review)

**Equivocal markers**
Binary positive/negative classification does not capture equivocal results
(e.g. "equivocal for CD117", "borderline positive", "cannot exclude X").
Model inconsistently classifies these as positive or negative because it was
trained on binary labels only.
Decision: add equivocal as a third outcome in the database schema, detected
by a post-processing rule-based pattern after ClinicalBERT extraction.
Retraining with a third label class is out of scope for this prototype and
would require substantially more annotated equivocal examples.
Noted as a limitation and future work direction in the report.

**RNOH vs referred immuno**
The NLP layer extracts all marker mentions regardless of source. The preference
rule (RNOH in-house takes precedence over referred) is applied at database
ingestion, not during extraction. The rule-based source classifier flags
referred vs in-house; the database layer uses this flag to select the
appropriate result for analysis and dashboard display.

**Marker normalisation**
All marker names are normalised to canonical form at ingestion via marker_utils.py.
This ensures dashboard queries are consistent regardless of how markers were
written in individual reports (sox-10 vs sox10, ae1/3 vs ae1/ae3, etc.).

---

## Database

Script: `06_database.py`
Output: `data/ihc_database.db` (SQLite)

### Schema

| Table | Rows | Description |
|---|---|---|
| `cases` | 7,309 | Patient ID, canonical diagnosis, subclassification, IHC text, morphology_notes |
| `ihc_markers` | ~65k | Per-patient per-marker outcomes (source: pipeline or gt) |
| `ki67_results` | 255 | Ki-67 proliferation index per case |
| `diagnoses` | 83 | Canonical labels, n_cases, include_in_analysis flag |
| `literature` | 146 | Expected IHC profiles from published references |
| `molecular_tests` | 56 | Per-patient molecular test results (test_type, neg_probes, pos_result) |
| `molecular_dx_markers` | 31 | Knowledge base: diagnosis-to-test mappings from literature |

### Diagnosis normalisation decisions

**Capitalisation merges:** all capitalisation variants of the same diagnosis merged
(e.g. neuroblastoma/Neuroblastoma, leiomyosarcoma HG / Leiomyosarcoma HG).

**PDS/AFX merge:** pleomorphic dermal sarcoma/atypical fibroxanthoma (n=271),
pleomorphic dermal sarcoma (n=22), and Atypical fibroxanthoma (n=16) all mapped
to `pleomorphic_dermal_sarcoma` (n=262 after deduplication).

**LMS re-labelling from RNOH masterfiles:**
The five RNOH masterfile spreadsheets provide a pathologist-reviewed canonical
diagnosis (column AD) and subclassification (column AG) for each case. These
override the raw GT label for 542 matched cases. Cross-reference match rate:
96.3% exact agreement between masterfile AT/AU and our GT positive/negative IHC
columns after marker normalisation.

Five canonical LMS groups after re-labelling:

| Canonical diagnosis | n (GT) | Notes |
|---|---|---|
| leiomyosarcoma_hg | 417 | Soft tissue, high grade |
| leiomyosarcoma_asmt | 74 | Atypical smooth muscle tumour |
| synovial_sarcoma | 47 | Uterine, high grade |
| uterine_leiomyosarcoma_asmt | 24 | Reserved (n < 30) |
| uterine_tumour_uncertain_malignant_potential | 4 | Reserved (n < 30) |

**Subclassification (AG column) stored per case** for LMS groups. Key subtypes:
- leiomyosarcoma_asmt: Cutaneous leiomyosarcoma (n=32), Leiomyoma (n=13), STUMP (n=2), Intranodal leiomyoma (n=1)
- uterine_leiomyosarcoma_hg: Myxoid (n=2), Epithelioid (n=1)
- uterine_leiomyosarcoma_asmt: STUMP (n=3), Leiomyoma variants (n=6)
- uterine_tumour_uncertain_malignant_potential: molecular fusion subtypes

**Unconfirmed diagnoses:** retained in database, flagged reserved (include_in_analysis=0).
Not shown in main dashboard analysis. Available for optional extended analysis.

**Minimum case threshold:** n >= 30 confirmed cases required for include_in_analysis=1.
32 diagnoses meet this threshold.

### Literature profiles

146 expected marker-outcome entries across 32 diagnoses. Sources: WHO Classification
of Soft Tissue and Bone Tumours (5th ed., 2020); Fletcher CDM et al., Diagnostic
Histopathology of Tumors (5th ed.); Hornick JL, Practical Soft Tissue Pathology
(2nd ed., 2019). Frequencies are approximate population-level estimates.

Sample validation (leiomyosarcoma_hg):
- SMA: literature 95%, pipeline 90.6%
- Desmin: literature 85%, pipeline 77.0%
- Caldesmon: literature 80%, pipeline 59.2%

### Usage

    python 06_database.py --build    # build/rebuild database
    python 06_database.py --stats    # summary statistics


---

## Dashboard

Script: `07_dashboard.py`
Framework: Streamlit + Plotly

Usage: `streamlit run 07_dashboard.py`

Four views:

1. **Diagnosis Panel** -- marker positivity/negativity rates (pipeline predictions),
   optional ground truth overlay, Ki-67 histogram. Configurable top-N and outcome filter.

2. **Literature Comparison** -- grouped bar chart (institutional vs literature) and
   delta chart highlighting deviations beyond ±15 percentage points. Colour-coded table.

3. **LMS Explorer** -- subclassification distribution (pie chart), per-subclass marker
   frequency bar charts, Ki-67 box plots by subclassification.

4. **Case Browser** -- filterable case table by subclassification and marker/outcome,
   individual case detail view showing pipeline predictions vs GT with colour-coded
   marker chips and full IHC report text.

**8-epoch retraining note:** a retraining run at 8 epochs was initiated on the GPU
server to compare with the 5-epoch model (F1=0.870). The run was interrupted before
completion. Given that F1=0.870 already exceeds the 0.75-0.85 proposal target, and
time constraints required prioritising the database and dashboard, the 8-epoch
experiment was not repeated. This is documented as a planned but incomplete ablation
(noted in the report as a direction for future optimisation).

## Data weaknesses

This section documents known limitations of the dataset that should be
disclosed in the final report methodology and limitations sections.

### Diagnosis-level sample size variability

Case counts vary enormously across diagnoses. The minimum threshold for
inclusion in main dashboard analysis is n >= 30 confirmed cases, but many
included diagnoses sit close to this floor, and supplementary datasets
ingested later have far fewer cases.

| Diagnosis | Cases (GT) | Notes |
|---|---|---|
| leiomyosarcoma_hg | 417 | Main pilot dataset; well-powered |
| pleomorphic_dermal_sarcoma | 262 | Large combined group |
| solitary_fibrous_tumour | 302 | Large but GT annotation incomplete (see above) |
| tenosynovial_giant_cell_tumour | 24 | Small; supplementary Excel only |
| uterine_leiomyosarcoma_asmt | 24 | Below threshold; reserved in analysis |

**Implication:** DDx frequency scores for small-n diagnoses are unreliable.
The dashboard enforces a minimum of 5 tested cases per marker before
displaying a data frequency (shows "n/a" otherwise). For diagnoses with
fewer than 30 cases overall, all data frequencies should be treated as
indicative only.

**TGCT specifically (24 cases):** Added August 2026 as a supplementary
dataset from a separate RNOH Excel spreadsheet (DigPath_TGCT_v5_pseudoanon).
This dataset was not part of the original main data export, so it was not
included in pipeline training or ClinicalBERT fine-tuning. The TGCT subset
is therefore evaluated but not used to train the extraction model. The
small sample size means marker frequencies derived from this subset carry
high uncertainty and should not be generalised.

### Non-contributory IHC reports

Two TGCT cases (S00182460 and S00173400) have IHC text that is explicitly
non-contributory ("Immunohistochemistry is non-contributory." and
"Referred immunohistochemistry is non-contributory."). These cases are
retained in the database (has_gt=0, ihc_text empty) so they count toward
the diagnosis case total, but they contribute no pipeline predictions and
no data frequency signal.

### Markers absent from pipeline vocabulary

The rule-based and ClinicalBERT pipelines were trained on markers present
in the main dataset. Several markers commonly used in TGCT and other
diagnoses are effectively absent from the RNOH pipeline output because they
rarely or never appear in the training corpus:

| Marker | Relevance | Pipeline extraction rate (approx.) |
|---|---|---|
| TLE1 | Synovial sarcoma (strong) | Near zero |
| FLI1 | Ewing sarcoma, synovial sarcoma | Near zero |
| TFE3 | TGCT (occasionally) | Near zero |
| CAM5.2 | Carcinoma, TGCT | Low (some synonyms present) |
| MDM2 | Well-diff. liposarcoma | Near zero |
| CDK4 | Well-diff. liposarcoma | Near zero |

Consequence: the DDx tool will show "n/a" for these markers even when they
are clinically meaningful, because the pipeline rarely extracts them. This
is a coverage gap in the pipeline, not a database design flaw.

### Ground truth incompleteness (SFT and MPNST)

Previously documented above under "Results summary", but restated here for
completeness. For SFT (n=302) and MPNST (n=75), the GT annotation recorded
only the primary confirming marker in most cases. The pipeline correctly
extracts the full panel from the text, but these appear as false positives
against the incomplete GT. This inflates apparent FP counts for these two
diagnoses. Reported separately as F1=0.873 excluding SFT+MPNST.

### Dataset provenance mismatch

The main dataset and the TGCT supplement were exported from different RNOH
systems (main GT spreadsheet vs per-diagnosis DigPath workbooks). Column
layouts, marker naming conventions, and annotation completeness differ
between sources. The TGCT supplement uses columns B, E, U, V, W; the main
GT uses columns A, D, E, R, S, T. Marker normalisation (marker_utils.py)
mitigates naming differences, but any systematic differences in annotation
practice between sources remain.

---

## Molecular pathology data

### New tables (added August 2026)

| Table | Rows | Description |
|---|---|---|
| `molecular_tests` | 56 | Per-patient molecular test results ingested from DigPath workbooks |
| `molecular_dx_markers` | 31 | Knowledge base: diagnosis-to-test mappings from literature |

`molecular_tests` schema: `patient_id, test_type, negative_probes, positive_result, source_file`
`molecular_dx_markers` schema: `canonical_diagnosis, test_type, alteration, role (confirmatory/supportive/exclusion), prevalence_pct, note`

Sources: all DigPath workbooks uploaded (ASPS, Dermatofibroma, TGCT, LMS_HG, LMS_ASMT, Schwannoma, UTUMP, Uterine_ASMT, Uterine_HG).
Molecular data ingested only for patient IDs already present in the `cases` table (56 records across 9 diagnosis groups after cleaning).

### Diagnosis-specific notes: molecular and diagnostic criteria

**Alveolar Soft Part Sarcoma (ASPS)**
Diagnosis requires at least ONE of three criteria:
1. Morphological: PAS-D positive intracytoplasmic crystals on H&E/PAS staining (present in ~40% of cases, from older literature; frequently absent in small biopsies).
2. IHC: nuclear TFE3 positivity (sensitive but not entirely specific; also positive in some Xp11 translocation renal cell carcinomas and PEComas).
3. Molecular: ASPSCR1::TFE3 fusion (PCR or NGS, ~95%) or TFE3 rearrangement by FISH (when fusion partner unknown).

Some older RNOH cases confirmed on morphology and IHC alone (pre-NGS era). Cases S00144045 and S00144066 are examples: PCR/FISH negative for ASPSCR1::TFE3 but confirmed by PAS-D crystal morphology. Case S00144062 has FISH-positive TFE3 rearrangement with unknown fusion partner (PCR for ASPSCR1::TFE3 was negative, consistent with an alternative TFE3 partner).

Morphology data: `morphology_notes` column added to `cases` table. 11 ASPS cases populated with "PAS-D positive crystals" notation extracted from DigPath workbook morphology column.

**Dermatofibroma / Benign Fibrous Histiocytoma (DFB)**
Diagnosis is primarily clinical (skin lesion history, physical presentation), histopathological (spindle cell proliferation in the dermis with characteristic growth pattern), and IHC-based (CD34 variable, CD68 positive in histiocytes, F13a positive, S100 negative). Molecular testing is not routinely performed.

FISH for COL1A1::PDGFB is used selectively to exclude DFSP when morphology is ambiguous (cases PH006966 and PH017101 in this dataset: COL1A1 FISH negative, supporting dermatofibroma over DFSP).

A PDPN::PRKCB fusion has been reported in a subset of dermatofibromas (PMID:24721208). Case 23CH10951 in this dataset carries this fusion on NGS; it is considered an incidental molecular finding consistent with the diagnosis, not a diagnostic marker. The biological significance is not fully established.

**Tenosynovial Giant Cell Tumour (TGCT)**
Diagnosis is primarily clinical, radiological, and histopathological. CSF1 gene rearrangement (leading to CSF-1 overproduction and macrophage recruitment) is present in ~60% of diffuse/aggressive TGCT and can be detected by FISH or NGS. FN1::TEK is an alternative TGCT-associated fusion (~10%). Molecular testing is used selectively: cases S00165119 (USP6 FISH to exclude nodular fasciitis, atypical forearm site) and S00183050 (SS18 FISH to exclude synovial sarcoma) demonstrate exclusion testing for atypical presentations.

**Leiomyosarcoma (LMS, all subtypes)**
Diagnosis is primarily morphological: cytologic atypia, mitotic index (>10/mm2 for high-grade), coagulative tumour necrosis, and cellular architecture (intersecting fascicles). IHC supports the diagnosis (SMA, desmin, caldesmon positivity). Molecular testing is used only for exclusion of mimics, primarily FISH for MDM2 amplification to exclude a dedifferentiated liposarcoma component in pleomorphic variants. NGS in this dataset (Uterine_HG spreadsheet) showed no fusion in confirmed LMS cases, which is expected (LMS is not a fusion-driven tumour). Note: case S00170062 underwent EBV FISH to exclude EBV-associated smooth muscle tumour in the context of tuberculosis infection.

**Schwannoma**
Case S00159173 has a confirmed NF2 nonsense mutation (c.892C>T, p.Gln298Ter) on NGS, consistent with NF2-associated schwannoma. Diagnosis is otherwise morphological and IHC-based (S100 diffuse positive, SOX10 positive).

---

## To decide / revisit

- Whether to include flagged anonymous cases after manual review
- Expand morphology_notes extraction to other diagnoses beyond ASPS (future work)
- Confirm with team which diagnoses routinely undergo molecular testing at RNOH (planned review, week of 25 August 2026)
