"""
marker_utils.py
IHC NLP Project : Marker Normalisation and Ki-67 Extraction
------------------------------------------------------------
Shared utilities used by the hybrid pipeline, database ingestion,
and evaluation scripts.

Key functions:
    normalise_marker(name)      : canonical lowercase form for set comparison
    normalise_marker_list(lst)  : apply to a list of marker names
    extract_ki67(text)          : extract Ki-67/MIB-1 percentage from report text
    strip_ki67(marker_list)     : remove Ki-67/MIB-1 from a marker list
"""

import re


# ── Alias table ────────────────────────────────────────────────────────────────
# Maps any observed variant (lowercased, stripped) to a canonical lowercase form.
# Ordered from most specific (multi-word compound forms) to single-token aliases
# so that exact-match lookup always resolves the longest form first.

ALIAS_TABLE = {

    # ── S100 ──────────────────────────────────────────────────────────────────
    "s-100":                        "s100",
    "s 100":                        "s100",
    "s100 protein":                 "s100",
    "s-100 protein":                "s100",
    "s100protein":                  "s100",

    # ── SOX10 ─────────────────────────────────────────────────────────────────
    "sox-10":                       "sox10",
    "sox 10":                       "sox10",

    # ── AE1/AE3 and all pancytokeratin variants ───────────────────────────────
    # Per RNOH antibody list and user instruction: panck/pancytokeratin variants
    # are clinically merged with AE1/AE3. MNF116 compound forms map to mnf116.
    "ae1/3":                        "ae1/ae3",
    "ae1-ae3":                      "ae1/ae3",
    "ae1-3":                        "ae1/ae3",
    "ae1:ae3":                      "ae1/ae3",
    "ae 1/3":                       "ae1/ae3",
    "ae 1/ae3":                     "ae1/ae3",
    "ae 1-ae3":                     "ae1/ae3",
    "ck ae1/ae3":                   "ae1/ae3",
    "ck ae1-ae3":                   "ae1/ae3",
    "cytokeratin ae1/ae3":          "ae1/ae3",
    "cytokeratin ae1-ae3":          "ae1/ae3",
    "cytokeratin ae1/3":            "ae1/ae3",
    "cytokeratins ae1/ae3":         "ae1/ae3",
    "cytokeratins ae1-ae3":         "ae1/ae3",
    "cytokeratins ae1/3":           "ae1/ae3",
    "keratins ae1/ae3":             "ae1/ae3",
    "keratin ae1/ae3":              "ae1/ae3",
    "keratin ae1/3":                "ae1/ae3",
    "keratin a1/3":                 "ae1/ae3",
    "panck":                        "ae1/ae3",
    "pan-cytokeratin":              "ae1/ae3",
    "pancytokeratin":               "ae1/ae3",
    "pan cytokeratin":              "ae1/ae3",
    "pankeratin":                   "ae1/ae3",
    "pan ck":                       "ae1/ae3",
    "pan-ck":                       "ae1/ae3",
    "pank":                         "ae1/ae3",
    "broad-spectrum cytokeratins":  "ae1/ae3",
    "broad spectrum cytokeratins":  "ae1/ae3",
    "broad-spectrum cytokeratin":   "ae1/ae3",
    "broad spectrum cytokeratin":   "ae1/ae3",
    "hmw ck":                       "ae1/ae3",   # high-molecular-weight CK

    # ── MNF116 (pankeratin qualifier stripped; MNF116 is a specific CK clone) ─
    "pankeratin mnf116":            "mnf116",
    "pan-keratin mnf116":           "mnf116",
    "panck mnf116":                 "mnf116",
    "pank mnf116":                  "mnf116",
    "cytokeratin mnf116":           "mnf116",
    "cytokeratins mnf116":          "mnf116",
    "keratin mnf116":               "mnf116",
    "keratin mnf 116":              "mnf116",
    "pankeratin mnf 116":           "mnf116",
    "pank mnf 116":                 "mnf116",
    "mnf 116":                      "mnf116",
    "mnf-116":                      "mnf116",
    "mnf":                          "mnf116",    # safe in RNOH context (only one MNF clone)
    "ck mnf116":                    "mnf116",

    # ── CAM5.2 ────────────────────────────────────────────────────────────────
    "cam5.2":                       "cam5.2",
    "cam 5.2":                      "cam5.2",
    "cam5":                         "cam5.2",
    "cam 5":                        "cam5.2",
    "cytokeratin cam5":             "cam5.2",
    "cytokeratin cam 5":            "cam5.2",
    "cam":                          "cam5.2",    # safe: only CAM5.2 used at RNOH

    # ── INI1 / BAF47 ──────────────────────────────────────────────────────────
    "ini-1":                        "ini1",
    "ini 1":                        "ini1",
    "ini1-1":                       "ini1",      # typo
    "baf47":                        "ini1",
    "baf-47":                       "ini1",
    "baf 47":                       "ini1",

    # ── Melan-A ───────────────────────────────────────────────────────────────
    "melan a":                      "melan-a",
    "melana":                       "melan-a",
    "mel a":                        "melan-a",
    "mealn a":                      "melan-a",   # typo

    # ── Desmin clones ──────────────────────────────────────────────────────────
    "d33":                          "desmin",    # Dako clone D33 = desmin

    # ── Caldesmon ─────────────────────────────────────────────────────────────
    "h-caldesmon":                  "caldesmon",
    "h caldesmon":                  "caldesmon",

    # ── MUC4 ──────────────────────────────────────────────────────────────────
    "muc-4":                        "muc4",
    "muc 4":                        "muc4",

    # ── HMB45 ─────────────────────────────────────────────────────────────────
    "hmb-45":                       "hmb45",
    "hmb 45":                       "hmb45",

    # ── CD117 / c-kit ─────────────────────────────────────────────────────────
    "c-kit":                        "cd117",
    "ckit":                         "cd117",
    "c kit":                        "cd117",

    # ── SMA ───────────────────────────────────────────────────────────────────
    "smooth muscle actin":          "sma",
    "smooth-muscle actin":          "sma",
    "smooth muscle antigen":        "sma",
    "sm-actin":                     "sma",
    "alpha-sma":                    "sma",
    "alpha sma":                    "sma",
    "a-sma":                        "sma",
    "sm actin":                     "sma",
    "muscle specific actin":        "sma",
    "msa":                          "sma",       # muscle-specific actin
    "actin":                        "sma",       # unqualified 'actin' in RNOH context = SMA
    "sm-actin":                     "sma",
    "sm actin":                     "sma",
    "smooth muscle actin":          "sma",
    "smooth-muscle actin":          "sma",

    # ── Ki-67 / MIB-1 ─────────────────────────────────────────────────────────
    "mib-1":                        "ki67",
    "mib1":                         "ki67",
    "mib 1":                        "ki67",
    "ki-67":                        "ki67",
    "ki 67":                        "ki67",

    # ── Factor XIIIA ──────────────────────────────────────────────────────────
    "factor xiiia":                 "f13a",
    "factor 13a":                   "f13a",
    "factor xiii":                  "f13a",
    "factor x111a":                 "f13a",      # OCR typo (three ones)
    "fxiiia":                       "f13a",
    "fxiii":                        "f13a",
    "xiiia":                        "f13a",
    "factor 13":                    "f13a",
    "f13a":                         "f13a",      # already canonical; keeps lookup fast

    # ── Beta-catenin ──────────────────────────────────────────────────────────
    "beta catenin":                 "beta-catenin",
    "b catenin":                    "beta-catenin",
    "b-catenin":                    "beta-catenin",
    "b catenein":                   "beta-catenin",  # typo

    # ── ER / PR ───────────────────────────────────────────────────────────────
    "oestrogen receptor":           "er",
    "estrogen receptor":            "er",
    "er receptor":                  "er",
    "oestrogen receptors":          "er",
    "estrogen receptors":           "er",
    "progesterone receptor":        "pr",
    "progestogen receptor":         "pr",
    "progesterone receptors":       "pr",

    # ── STAT6 ─────────────────────────────────────────────────────────────────
    "stat-6":                       "stat6",
    "stat 6":                       "stat6",

    # ── TTF1 ──────────────────────────────────────────────────────────────────
    "ttf-1":                        "ttf1",
    "ttf 1":                        "ttf1",

    # ── DOG1 ──────────────────────────────────────────────────────────────────
    "dog-1":                        "dog1",
    "dog 1":                        "dog1",

    # ── BCL2 ──────────────────────────────────────────────────────────────────
    "bcl-2":                        "bcl2",
    "bcl 2":                        "bcl2",

    # ── MyoD1 ─────────────────────────────────────────────────────────────────
    "myo-d1":                       "myod1",
    "myo d1":                       "myod1",

    # ── Myogenin (MYF4 is the gene encoding myogenin) ─────────────────────────
    "myf4":                         "myogenin",
    "myf-4":                        "myogenin",

    # ── CD markers with spaces ─────────────────────────────────────────────────
    "cd 34":                        "cd34",
    "cd 45":                        "cd45",
    "cd 99":                        "cd99",
    "cd 10":                        "cd10",
    "cd 163":                       "cd163",
    "cd 56":                        "cd56",
    "cd 31":                        "cd31",
    "cd 30":                        "cd30",
    "cd 68":                        "cd68",
    "cd 20":                        "cd20",
    "cd 117":                       "cd117",
    "cd 138":                       "cd138",
    "cd 3":                         "cd3",
    "cd 79a":                       "cd79a",
    "cd 1a":                        "cd1a",
    "cd 21":                        "cd21",

    # ── CDK4 ──────────────────────────────────────────────────────────────────
    "cdk 4":                        "cdk4",
    "cdk-4":                        "cdk4",

    # ── LCA / CD45 ────────────────────────────────────────────────────────────
    "lca":                          "cd45",
    "leucocyte common antigen":     "cd45",
    "leucocyte antigen":            "cd45",

    # ── Chromogranin ──────────────────────────────────────────────────────────
    "chromogranin a":               "chromogranin",
    "chromagranin a":               "chromogranin",  # typo
    "chromagranin":                 "chromogranin",  # typo

    # ── MIC2 = CD99 ───────────────────────────────────────────────────────────
    "mic2":                         "cd99",
    "mic-2":                        "cd99",

    # ── ALK ───────────────────────────────────────────────────────────────────
    "alk1":                         "alk",
    "alk-1":                        "alk",
    "alk 1":                        "alk",

    # ── Cyclin D1 ─────────────────────────────────────────────────────────────
    "cyclin d1":                    "cyclin-d1",
    "cyclyn d1":                    "cyclin-d1",    # typo
    "cyclind1":                     "cyclin-d1",

    # ── D2-40 ─────────────────────────────────────────────────────────────────
    "d240":                         "d2-40",
    "d2 40":                        "d2-40",

    # ── FLI1 ──────────────────────────────────────────────────────────────────
    "fli-1":                        "fli1",
    "fli 1":                        "fli1",

    # ── WT1 ───────────────────────────────────────────────────────────────────
    "wt-1":                         "wt1",
    "wt 1":                         "wt1",

    # ── Glut1 ─────────────────────────────────────────────────────────────────
    "glut 1":                       "glut1",
    "glut-1":                       "glut1",

    # ── OCT3/4 ────────────────────────────────────────────────────────────────
    "oct 3/4":                      "oct3/4",
    "oct-3/4":                      "oct3/4",

    # ── Phox2b ────────────────────────────────────────────────────────────────
    "phox-2b":                      "phox2b",
    "phox 2b":                      "phox2b",

    # ── Pan-TRK ───────────────────────────────────────────────────────────────
    "pan-trk":                      "pantrk",
    "pan trk":                      "pantrk",

    # ── Cytokeratin subtypes ───────────────────────────────────────────────────
    "ck 5/6":                       "ck5/6",
    "ck5-6":                        "ck5/6",
    "ck 5":                         "ck5",
    "ck-5":                         "ck5",
    "cytokeratin 5/6":              "ck5/6",
    "cytokeratins 5/6":             "ck5/6",
    "ck 7":                         "ck7",
    "cytokeratin 7":                "ck7",
    "ck-7":                         "ck7",
    "ck 14":                        "ck14",
    "cytokeratin 14":               "ck14",
    "cytokeratin 5":                "ck5",
    "ck 20":                        "ck20",
    "cytokeratin 20":               "ck20",
    "ck 8/18":                      "ck8/18",
    "ck8-18":                       "ck8/18",

    # ── HHV8 / KSHV ───────────────────────────────────────────────────────────
    "hhv-8":                        "hhv8",
    "hhv 8":                        "hhv8",
    "kshv":                         "hhv8",         # KSHV = HHV-8

    # ── EMA synonyms ──────────────────────────────────────────────────────────
    "epithelial membrane antigen":  "ema",

    # ── NSE synonyms ──────────────────────────────────────────────────────────
    "neurone specific enolase":     "nse",
    "neuron specific enolase":      "nse",

    # ── MDM2 ──────────────────────────────────────────────────────────────────
    "mdm-2":                        "mdm2",
    "mdm 2":                        "mdm2",

    # ── H3.3 G34W ─────────────────────────────────────────────────────────────
    "h3.3 g34w":                    "h3-3 g34w",
    "h3 g34w":                      "h3-3 g34w",
    "h3-3g34w":                     "h3-3 g34w",
    "h-3 g34w":                     "h3-3 g34w",
    "h3-3 g34 w":                   "h3-3 g34w",
    "h3f3 g34w":                    "h3-3 g34w",   # H3F3 = gene name for H3.3

    # ── Remaining multi-word compound forms ───────────────────────────────────
    "both ini-1":                   "ini1",
    "pan cytokeratins":             "ae1/ae3",
    "pan-cytokeratin ae1/ae3":      "ae1/ae3",
    "broad spectrum":               "ae1/ae3",
    "type 4 collagen":              "collagen iv",
    "collagen type iv":             "collagen iv",
    "collagen4":                    "collagen iv",

    # ── H3K27M ────────────────────────────────────────────────────────────────
    "mutant h3k27m":                "h3k27m",
    "h3-k27m":                      "h3k27m",

    # ── NB84 ──────────────────────────────────────────────────────────────────
    "nb-84":                        "nb84",
    "nb84a":                        "nb84",

    # ── RNOH official list compound / prefix forms ────────────────────────────
    # "CK - X" format used in the official antibody spreadsheet
    "ck - keratin nos":             "ae1/ae3",
    "ck -  keratin nos":            "ae1/ae3",
    "ck - 20":                      "ck20",
    "ck - 5":                       "ck5",
    "ck - 7":                       "ck7",
    "ck - ae1/ae3":                 "ae1/ae3",
    "ck - cam 5.2":                 "cam5.2",
    "ck - ck19":                    "ck19",
    "ck - mnf116":                  "mnf116",
    "ck-34be12":                    "34be12",
    "ck-ck14":                      "ck14",
    "ck-ck8/18":                    "ck8/18",
    "ck-hmwk":                      "hmwck",
    "34 beta e12":                  "34be12",
    "34 betae12":                   "34be12",
    "ck34be12":                     "34be12",
    "ck 34be12":                    "34be12",
    # CD compounds from RNOH list
    "cd45 / lca":                   "cd45",
    "cd56 / ncam":                  "cd56",
    # SMA / Actin compound label
    "sma / actin":                  "sma",
    "sma/actin":                    "sma",
    # Pancytokeratin compound label from RNOH
    "pancytokeratin / pank / keratin / pck / ck": "ae1/ae3",
    # Factor VIII
    "factor viii":                  "fviii",
    "factor 8":                     "fviii",
    "f8":                           "fviii",
    "fviii":                        "fviii",
    "fviiia":                       "fviii",
    # HepPar-1
    "heppar-1":                     "heppar1",
    "heppar 1":                     "heppar1",
    "hepar-1":                      "heppar1",
    "heppar":                       "heppar1",
    "hep par-1":                    "heppar1",
    # IDH1 R132H
    "idh1 r132h":                   "idh1-r132h",
    "idh1r132h":                    "idh1-r132h",
    "idh1-r132-h":                  "idh1-r132h",
    "idh1":                         "idh1-r132h",
    # Ki67 compound
    "ki67/mib1":                    "ki67",
    "ki67 mib1":                    "ki67",
    # NKX3.1
    "nkx 3.1":                      "nkx3.1",
    "nkx3-1":                       "nkx3.1",
    "nkx-3.1":                      "nkx3.1",
    "nkx3.1":                       "nkx3.1",
    # RCC Ma
    "rcc ma":                       "rcc",
    "rcc marker":                   "rcc",
    # BerEP4
    "ber-ep4":                      "berep4",
    "ber ep4":                      "berep4",
    "berep-4":                      "berep4",
    "ber-ep-4":                     "berep4",
    # H3-3 K36M (H3-e is a variant spelling where 'e' stands for the gene suffix)
    "h3.3 k36m":                    "h3-3 k36m",
    "h3 k36m":                      "h3-3 k36m",
    "k36m":                         "h3-3 k36m",
    "h3k36m":                       "h3-3 k36m",
    "h3-e k36m":                    "h3-3 k36m",
    "h3e k36m":                     "h3-3 k36m",
    "h3e-k36m":                     "h3-3 k36m",
    "h3-ek36m":                     "h3-3 k36m",
    # SMARCB1 = INI1
    "smarcb1":                      "ini1",
    # SMARCA4 / BRG1
    "brg1":                         "smarca4",
    "brg-1":                        "smarca4",
    # SF-1
    "sf1":                          "sf-1",
    "sf 1":                         "sf-1",
    # GATA3 variants
    "gata-3":                       "gata3",
    "gata 3":                       "gata3",
    # FosL-1 variants
    "fosl1":                        "fosl-1",
    "fosl 1":                       "fosl-1",
    "c-fosl1":                      "fosl-1",
    # Claudin-1 variants
    "claudin1":                     "claudin-1",
    "claudin 1":                    "claudin-1",
    "claudin4":                     "claudin-4",
    "claudin 4":                    "claudin-4",
    # c-FOS / c-myc cleanup
    "c fos":                        "c-fos",
    "cfos":                         "c-fos",
    "cmyc":                         "c-myc",
    "c myc":                        "c-myc",
    # PAX8 variants
    "pax-8":                        "pax8",
    "pax 8":                        "pax8",
    # PLAG1 variants
    "plag-1":                       "plag1",
    "plag 1":                       "plag1",
    # SATB2 variants
    "satb-2":                       "satb2",
    "satb 2":                       "satb2",
    # HMGA2 variants
    "hmga-2":                       "hmga2",
    "hmg-a2":                       "hmga2",
    # Pan-TRK variants
    "pantrk":                       "pantrk",
    # D2-40 / podoplanin
    "podoplanin":                   "d2-40",
    # Vimentin typo
    "vimentine":                    "vimentin",
    # MPO / myeloperoxidase
    "myeloperoxidase":              "mpo",
    # CD20 variants
    "cd-20":                        "cd20",
    "cd 20":                        "cd20",
    # AR / androgen receptor
    "androgen receptor":            "ar",
    # MOC31
    "moc-31":                       "moc31",
    # SMM / smooth muscle myosin
    "smooth muscle myosin":         "smm",
    "smooth muscle myosin hc":      "smm",
    "smms-1":                       "smm",
    "smms":                         "smm",
    # CK8 → CK8/18 (RNOH context)
    "ck8":                          "ck8/18",
    # MCK
    "muscle cytokeratin":           "mck",
    # Calponin variants
    "h-calponin":                   "calponin",
    "h calponin":                   "calponin",
    # Lin28A
    "lin28":                        "lin28a",
    "lin-28a":                      "lin28a",
    # Inhibin / alpha-inhibin
    "alpha-inhibin":                "inhibin",
    "alpha inhibin":                "inhibin",
    "inhibin a":                    "inhibin",
    # OCT3/4
    "oct4":                         "oct3/4",
    "oct-3/4":                      "oct3/4",
    "oct 3/4":                      "oct3/4",
    # LEF1
    "lef-1":                        "lef1",
    "lef 1":                        "lef1",
    # NKX2.2 variants
    "nkx 2.2":                      "nkx2.2",
    "nkx-2.2":                      "nkx2.2",
    # PLAP
    "placental alkaline phosphatase": "plap",
    # Neurofilament variants
    "neurofilaments":               "neurofilament",
    "neurofilament protein":        "neurofilament",
    # SM5-1
    "sm 5-1":                       "sm5-1",
    # CAIX variants
    "ca-ix":                        "caix",
    "ca ix":                        "caix",
    # BRAF V600E variants
    "braf v600":                    "braf v600e",
    "brafv600e":                    "braf v600e",
    "braf v600e":                   "braf v600e",
}


# Markers excluded from binary positive/negative classification.
# Ki-67 is a proliferation index (percentage), not a binary marker.
EXCLUDED_FROM_BINARY = {"ki67", "ki-67", "ki 67", "mib-1", "mib1", "mib 1"}


# ── Leading-phrase strip ───────────────────────────────────────────────────────
# Applied iteratively (up to 3 passes) to remove context words that the BERT
# span extractor sometimes includes at the start of a marker span.
# Each pattern must match at the start and leave at least one char remaining.

_LEADING_PATTERN = re.compile(
    r'^(?:'
    r'on\s+immuno(?:histo(?:chemistry|chemical)|stain(?:ing|s)?)\s+'
    r'|immuno(?:histo)?chemistry\s+(?:for|with)\s+'
    r'|immunostain(?:ing|s)?\s+(?:for|with)\s+'
    r'|immunoreactivity\s+(?:for|with)\s+'
    r'|immunoreactive\s+for\s+'
    r'|immunolabelling\s+for\s+'
    r'|stains?\s+(?:for|with)\s+'
    r'|staining\s+(?:for|with|by)\s+'
    r'|(?:positive|negative)\s+for\s+'
    r'|positivity\s+for\s+'
    r'|expression\s+of\s+'
    r'|express(?:es)?\s+'
    r'|whereas\s+'
    r'|whilst?\s+'
    r'|also\s+'
    r'|referred\s+'
    r'|for\s+'
    r'|(?:tumour|neoplastic|lesional|spindle)?\s*cells?\s+'
      r'(?:express(?:es)?|are\s+positive\s+for|co[-\s]?express(?:es)?)\s+'
    r'|both\s+'
    r'|equivocal\s+'
    r'|suspicious\s+for\s+'
    r'|to\s+(?=[a-z])'
    r'|with\s+(?:(?:focal(?:ly)?|multifocal|patchy|weak(?:ly)?|strong(?:ly)?|diffuse(?:ly)?|moderate|intense|extensive|variable|widespread)\s+)+'
    r'|more\s+(?:focally|weakly|strongly|patchily)\s+for\s+'
    r'|very\s+(?:scanty|focal|focally)\s+(?:focal\s+)?'
    r'|a\s+(?=[a-z])'
    r')',
    re.IGNORECASE,
)

# ── Trailing-noise strip ───────────────────────────────────────────────────────
# Applied iteratively (up to 3 passes) to remove status qualifiers, lab
# attribution suffixes, and stray words from the end of extracted spans.

_TRAILING_PATTERN = re.compile(
    r'(?:'
    r'\s+performed\s+at\s+(?:rnoh|rmh|our\s+(?:lab(?:oratory)?)|hsl[-\w\s]*).*$'
    r'|\s+(?:at|from)\s+(?:rnoh|rmh).*$'
    r'|\s+of\s+uncertain\s+significance.*$'
    r'|\s+(?:is|are)\s+(?:'
      r'retained?|lost|partially\s+lost|largely\s+(?:retained?|lost)|'
      r'reatined|ie\s+retained|noted|seen|present|positive|negative|expressed?'
    r').*$'
    r'|\s+(?:expression|staining|positivity|immunostaining|immunostain(?:s)?)$'
    r'|\s+(?:are|were|is|was)$'
    r'|\s+(?:diffusely|focally|patchily|weakly|strongly|multifocally)$'
    r'|\s+but\b.*$'
    r'|\s+whilst?\b.*$'
    r'|\s+whereas\b.*$'
    r'|\s+while\b.*$'
    r'|\s+only$'
    r'|\s+in\s+(?:the\s+)?(?:\w+\s+)?(?:tumou?r|lesional|neoplastic|spindle|atypical)?\s*cells?\b.*$'
    r')',
    re.IGNORECASE,
)

# ── Stop-word check ────────────────────────────────────────────────────────────
# After all stripping, if the result still contains one of these whole words
# it is not a valid marker name.

_STOP_WORD_RE = re.compile(
    r'\b(?:'
    r'cells?|tumour|immunohistochemistry|immunostaining|immunostain|'
    r'positive|negative|expression|referred|performed|tested|'
    r'lesional|neoplastic|spindle|markers?|antibodies|slides?|'
    r'staining|positivity'
    r')\b',
    re.IGNORECASE,
)

# ── Exclusion set ──────────────────────────────────────────────────────────────
# Strings that should return "" regardless of how they look after stripping.

MARKER_EXCLUSIONS = {
    # Non-specific IHC terminology
    "on immunohistochemistry",
    "immunohistochemistry",
    "immunostaining",
    "immunostain",
    "staining",
    "by immunohistochemistry",
    "referred immunohistochemistry",
    "referred immunohistyochemistry",
    "referred immunohistyochemsitry",
    "on ihc",
    "ihc",
    "on immunostaining",
    "referred ihc",
    # Non-marker context fragments
    "at rmh",
    "rnoh",
    "referred",
    "performed at rnoh",
    "referred slides",
    "referred markers",
    # Single-letter or stop-word fragments extracted by BERT
    "they",
    "as",
    "it",
    "me",
    "while",
    "whilst",
    "whereas",
    "also",
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "of",
    "on",
    "in",
    "to",
    "by",
    "with",
    # Anatomy / morphology noise
    "the neoplastic cells",
    "the lesion",
    "lesion",
    "the spindle cell component",
    "the tumour cells",
    "neoplastic cells",
    "spindle cell component",
    # Descriptive states
    "extensively",
    "patchily",
    "shows",
    "positivity",
    # Vague/generic marker-group labels (not analysable as single markers)
    "muscle markers",
    "epithelial markers",
    "vascular markers",
    "lymphoid markers",
    "melanocytic markers",
    "myeloid markers",
    "cytokeratin markers",
    "cytokeratin",          # too vague without a clone/number
    "cytokeratins",         # ditto
    "ck",                   # too vague
    "pan",                  # too vague (fragment of panck)
    "all the other antibodies tested",
    "all three muscle markers",
    "other markers tested",
    "the antibodies tested",
    "the following immunomarkers",
    "immunomarkers",
    "antibodies",
    "markers",
    "the rest of the markers tested",
    "the other markers",
    # Observed BERT noise phrases
    "to a lesser extend",
    "to a lesser extent",
    "whereas the other markers tested",
    "whereas they are",
    "whereas they",
    "whereas the cells",
    "whereas the cells are",
    "while they",
    "lesser extend",
    "lesser extent",
    "to occasionally moderate",
    "s100 with a diffuse cd34 background",
    "there is a diffuse cd34 background",
    "the rest of the immunomarkers",
    "but the majority",
    # Ki67 with embedded percentage (extract_ki67 handles these; exclude from binary)
    "30% ki67",
    "40% ki67",
    "20% ki67",
    "5% ki67",
    "low ki67",
    "high ki67",
}


# ── Canonical marker whitelist ────────────────────────────────────────────────
# Derived from the RNOH official antibody list (IHC_total_antibodies_no_duplicates.xlsx)
# plus a small set of additional valid markers used in soft-tissue pathology reports.
# normalise_marker() rejects any output not in this set.

CANONICAL_MARKERS: frozenset = frozenset({
    # From RNOH official list (canonical lowercase form after normalisation)
    "alk", "bcl2", "bcor", "berep4", "beta-catenin", "brachyury",
    "braf v600e", "caix", "caldesmon", "calponin", "calretinin",
    "camta1", "cathepsin k", "ccnb3",
    "cd3", "cd10", "cd117", "cd138", "cd15", "cd1a", "cd20", "cd21",
    "cd23", "cd30", "cd31", "cd34", "cd35", "cd45", "cd56",
    "cd57", "cd61", "cd68", "cd79a", "cd99", "cd163",
    "cdk4", "chromogranin",
    # Cytokeratin family
    "ae1/ae3",   # panCK / AE1/AE3
    "cam5.2",    # CAM 5.2
    "mnf116",    # MNF116
    "34be12",    # 34bE12
    "ck5", "ck5/6", "ck7", "ck8/18", "ck14", "ck19", "ck20",
    "hmwck", "mck",
    # Other markers
    "ar", "atrx",
    "bap1", "brck",
    "c-fos", "claudin-1", "claudin-4", "cmv", "c-myc",
    "cyclin-d1",
    "d2-40", "ddit3", "desmin", "dog1", "dux4",
    "eber", "ema", "er", "erg", "ersp1",
    "f13a", "fli1", "fosl-1", "fviii",
    "gata3", "gfap", "glut1",
    "h3k27m", "h3k27me3", "h3-3 g34w", "h3-3 k36m",
    "heppar1", "her2", "hhv8", "hmb45", "hmga2",
    "idh1-r132h", "igg4", "inhibin", "ini1", "insm1",
    "langerin", "lef1", "lin28a",
    "mdm2", "melan-a", "mitf", "moc31", "mpo", "muc4",
    "myod1", "myogenin",
    "nb84", "neurofilament", "nkx2.2", "nkx3.1", "nse",
    "oct3/4",
    "p16", "p40", "p53", "p63", "pantrk",
    "pax5", "pax8", "phox2b", "plag1", "plap", "pnl2",
    "ppargamma", "pr", "prame", "prdm10",
    "rb", "rcc",
    "s100", "sact", "satb2", "sf-1", "sm131", "sm5-1",
    "sma", "smarca4", "smm", "sox10", "sox11", "sox9",
    "ss18", "ss18-ssx", "stat6", "synaptophysin",
    "tdt", "tfe3", "thyroglobulin", "tle1", "ttf1", "tyrosinase",
    "ucp1",
    "vimentin",
    "wt1",
    # Ki-67 excluded from binary classification but kept for reference
    # (strip_ki67 / EXCLUDED_FROM_BINARY handles exclusion at the pipeline level)
    "ki67",
})


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise_marker(name: str) -> str:
    """
    Return a canonical lowercase form of a marker name for set comparison.

    Processing pipeline:
      1. Lowercase, strip outer whitespace and trailing punctuation.
      2. Strip trailing parenthetical qualifiers: "(focal)", "(kp1)", etc.
      3. Alias lookup (catches compound forms like "pankeratin mnf116" before
         any phrase stripping).
      4. Leading-phrase strip (up to 3 passes): removes context words such as
         "positive for", "staining for", "whereas", "referred", etc.
      5. Trailing-noise strip (up to 3 passes): removes "is retained",
         "performed at rnoh", "expression", "are", etc.
      6. Re-strip trailing punctuation; collapse whitespace.
      7. Alias lookup again (now the cleaned span may match).
      8. Exclusion check (known non-markers).
      9. Compound check: "X or Y" or "X and Y" spans are ambiguous; discard.
     10. Stop-word check: discard any result still containing words that never
         appear in valid marker names.
    """
    if not name or not str(name).strip():
        return ""

    name = str(name).strip().lower()

    # Step 1: strip trailing punctuation
    name = name.rstrip(".,;:()")

    # Step 2: strip trailing parenthetical qualifiers (with or without closing paren)
    name = re.sub(r'\s*\([^)]*\)?\s*$', '', name).strip()
    name = name.rstrip(".,;:()")

    if not name:
        return ""

    # Collapse internal whitespace before lookup
    name = re.sub(r'\s+', ' ', name)

    # Step 3: alias lookup (catches long compound forms)
    if name in ALIAS_TABLE:
        return ALIAS_TABLE[name]

    # Step 3b: early exclusion check before phrase stripping, so full-phrase
    # entries in MARKER_EXCLUSIONS are caught before "to a lesser extend"
    # becomes "lesser extend" after the leading strip.
    if name in MARKER_EXCLUSIONS:
        return ""

    # Step 4: leading-phrase strip (up to 3 passes)
    for _ in range(3):
        m = _LEADING_PATTERN.match(name)
        if m and m.end() < len(name):
            name = name[m.end():].strip()
        else:
            break

    if not name:
        return ""

    # Step 5: trailing-noise strip (up to 3 passes)
    for _ in range(3):
        new_name = _TRAILING_PATTERN.sub("", name).strip()
        if new_name == name:
            break
        name = new_name

    if not name:
        return ""

    # Step 6: re-strip punctuation and collapse whitespace
    name = name.rstrip(".,;:()")
    name = re.sub(r'\s+', ' ', name).strip()

    # Step 7: alias lookup again (stripped form may now match)
    if name in ALIAS_TABLE:
        return ALIAS_TABLE[name]

    # Step 8: exclusion check
    if name in MARKER_EXCLUSIONS:
        return ""

    # Step 9: compound marker check ("X or Y" / "X and Y" spans are ambiguous)
    if re.search(r'\b(?:or|and)\b', name):
        return ""

    # Step 10: stop-word sanity check
    if _STOP_WORD_RE.search(name):
        return ""

    # Step 11: canonical whitelist — reject anything not on the RNOH antibody list
    if name not in CANONICAL_MARKERS:
        return ""

    return name


def normalise_marker_list(marker_list: list) -> set:
    """
    Normalise a list of marker names and return as a set of canonical forms.
    Filters out empty strings and Ki-67/MIB-1 (handled separately).
    """
    result = set()
    for m in marker_list:
        norm = normalise_marker(m)
        if norm and norm not in EXCLUDED_FROM_BINARY:
            result.add(norm)
    return result


def parse_and_normalise(cell: str) -> set:
    """
    Parse a comma-separated marker cell and return normalised set.
    Convenience wrapper for use in evaluation and database ingestion.
    """
    if not cell or not str(cell).strip():
        return set()
    markers = [m.strip() for m in str(cell).split(",") if m.strip()]
    return normalise_marker_list(markers)


# ── Ki-67 extraction ──────────────────────────────────────────────────────────

# Patterns for Ki-67 / MIB-1 followed (or preceded) by a percentage
_KI67_PATTERN = re.compile(
    r"""
    (?:ki[- ]?67|mib[- ]?1)        # marker name
    [^%\n]{0,60}?                  # any context up to 60 chars
    (\d{1,3}(?:\.\d+)?)\s*%        # percentage value
    |
    (\d{1,3}(?:\.\d+)?)\s*%        # percentage value first
    [^%\n]{0,30}?                  # any context
    (?:ki[- ]?67|mib[- ]?1)        # then marker name
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_ki67(text: str) -> str | None:
    """
    Extract Ki-67/MIB-1 proliferation index from report text.
    Returns the percentage as a string (e.g. '70') or None if not found.
    If multiple values found, returns the last one (RNOH convention: last result
    is typically the in-house result, which takes precedence).
    """
    matches = _KI67_PATTERN.findall(str(text))
    values  = [m[0] or m[1] for m in matches if (m[0] or m[1])]
    if not values:
        return None
    return values[-1]  # last match (in-house takes precedence)


def strip_ki67(marker_list: list) -> list:
    """Remove Ki-67/MIB-1 entries from a marker list (for binary classification)."""
    ki67_re = re.compile(r'^(ki[- ]?67|mib[- ]?1)$', re.IGNORECASE)
    return [m for m in marker_list if not ki67_re.match(m.strip())]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Original tests
        ("sox-10",                   "sox10"),
        ("AE1/3",                    "ae1/ae3"),
        ("H-Caldesmon",              "caldesmon"),
        ("pankeratin MNF116",        "mnf116"),
        ("Melan A",                  "melan-a"),
        ("MIB-1",                    "ki67"),
        ("desmin.",                  "desmin"),
        ("INI-1",                    "ini1"),
        # New: phrase-stripping tests
        ("positive for desmin",      "desmin"),
        ("negative for s100",        "s100"),
        ("staining for s100",        "s100"),
        ("stains for mnf116",        "mnf116"),
        ("whereas s100",             "s100"),
        ("express sma",              "sma"),
        ("for desmin",               "desmin"),
        ("whereas pankeratin mnf116","mnf116"),
        ("referred cd34",            "cd34"),
        ("referred sma",             "sma"),
        # Trailing-noise tests
        ("h3k27me3 is retained",     "h3k27me3"),
        ("ini-1 is lost",            "ini1"),
        ("smarca4 is retained",      "smarca4"),
        ("stat6 performed at rnoh",  "stat6"),
        ("myogenin performed at rnoh","myogenin"),
        ("desmin expression",        "desmin"),
        ("sma positivity",           "sma"),
        ("desmin are",               "desmin"),
        ("s100 are",                 "s100"),
        ("ema (focal)",              "ema"),
        ("dog-1 performed at rnoh",  "dog1"),
        # Alias tests
        ("s100 protein",             "s100"),
        ("s-100",                    "s100"),
        ("oestrogen receptor",       "er"),
        ("progesterone receptor",    "pr"),
        ("factor xiiia",             "f13a"),
        ("factor 13a",               "f13a"),
        ("beta catenin",             "beta-catenin"),
        ("b catenin",                "beta-catenin"),
        ("cyclin d1",                "cyclin-d1"),
        ("panck",                    "ae1/ae3"),
        ("pancytokeratin",           "ae1/ae3"),
        ("broad-spectrum cytokeratins", "ae1/ae3"),
        ("cytokeratins ae1-ae3",     "ae1/ae3"),
        ("mnf 116",                  "mnf116"),
        ("stat 6",                   "stat6"),
        ("dog 1",                    "dog1"),
        ("bcl-2",                    "bcl2"),
        ("myf4",                     "myogenin"),
        ("lca",                      "cd45"),
        ("mic2",                     "cd99"),
        ("alk1",                     "alk"),
        ("chromogranin a",           "chromogranin"),
        ("baf47",                    "ini1"),
        ("kshv",                     "hhv8"),
        ("mel a",                    "melan-a"),
        ("sm-actin",                 "sma"),
        # Exclusion tests
        ("on immunohistochemistry",  ""),
        ("immunohistochemistry",     ""),
        ("the neoplastic cells",     ""),
        ("at rmh",                   ""),
        ("they",                     ""),
        ("referred immunohistochemistry", ""),
        ("to a lesser extend",       ""),
        # Compound tests
        ("sma or desmin",            ""),
        ("desmin or muc4",           ""),
        ("s100 or melan-a",          ""),
    ]

    print("Marker normalisation tests:")
    all_pass = True
    for inp, expected in tests:
        result = normalise_marker(inp)
        status = "PASS" if result == expected else f"FAIL (got {result!r})"
        print(f"  {inp:40s} -> {result:20s}  {status}")
        if result != expected:
            all_pass = False
    print(f"\nAll tests passed: {all_pass}")

    # Ki-67 extraction tests
    ki67_tests = [
        "Ki-67 index of 70%",
        "proliferation index (MIB-1): 45%",
        "Ki67 is approximately 30-40%",
        "no Ki-67 staining",
    ]
    print("\nKi-67 extraction tests:")
    for t in ki67_tests:
        result = extract_ki67(t)
        print(f"  {t!r:45s} -> {result}")
