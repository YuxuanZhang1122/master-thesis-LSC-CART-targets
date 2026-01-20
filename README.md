# AML CAR-T Dual Target Identification Pipeline

> **Thesis Manuscript**: stay tuned ...

Systematic pipeline for identifying combinatorial CAR-T targets in acute myeloid leukemia (AML) using single-cell RNA-seq data. It integrates reference atlas projection, ensemble classification, and differential expression analysis to identify cell surface targets for AML therapy. The pipeline processes single-cell data from multiple AML cohorts to discover malignancy-specific & progenitor-specific biomarkers.


![Graphical Abstract](Graphical%20Abstract.png)

<p align="center"><b>Ensemble-Resolved LSPC Atlas for AND-Gate CAR-T Target Discovery.</b><br>
<sub>We integrated 7 AML scRNA-seq datasets (101 donors) and projected cells onto a healthy bone marrow reference atlas (Zeng 2025) via scANVI. An ensemble of 7 classifiers inferred malignancy labels for primitive HSC/MPP cells, yielding a curated atlas of ~12,700 LSPCs and ~12,700 HSPCs. Differential expression and interaction modeling identified synergistic dual-target pairs—including EMB+CD47, EMB+HCST, and CD9+CD99—as candidate AND-gate CAR-T targets with high LSPC specificity and minimal HSPC toxicity.</sub></p>

## Repository Structure

```
master-thesis-LSC-CART-targets/
│
├── 1_RF_reimplementation/       # Step 1: Baseline Classifier Validation
│   ├── run_pipeline.py          # Main pipeline orchestration
│   ├── step1_prepare_data.py    # Data preprocessing
│   ├── step2_classifier1.py     # Cell type classification
│   ├── step3_prepare_malignant.py
│   ├── step4_classifier2.py     # Malignancy detection
│   ├── balanced_rf.py           # Balanced Random Forest implementation
│   └── Figures/
│
├── 2_Projections/               # Step 2: Reference Atlas Mapping
│   ├── main.py                  # scANVI reference model training
│   ├── process_query.py         # Query dataset projection
│   ├── utils.py                 # Helper functions
│   └── Figures/
│
├── 3_Ensemble/                  # Step 3: Malignancy Classification
│   ├── run_ensemble.py          # Ensemble pipeline execution
│   ├── ensemble.py              # Majority voting framework
│   ├── *_predictor.py           # Individual classifiers (7 models)
│   │   ├── randomforest_predictor.py
│   │   ├── xgboost_predictor.py
│   │   ├── lightgbm_predictor.py
│   │   ├── svm_predictor.py
│   │   ├── mlp_predictor.py
│   │   ├── celltypist_predictor.py
│   │   └── scANVI_predictor.py
│   └── Figures/
│
├── 4_TargetID/                  # Step 4: Target Discovery
│   ├── DEG_DESeq2.py            # Pseudo-bulk differential expression
│   ├── run_gsea.py              # Gene set enrichment analysis
│   ├── pair_search.py         # Combinatorial target identification
│   └── Figures/
│
├── Graphical Abstract.png       # Pipeline overview figure
└── README.md
```

## Pipeline Steps

### Step 1: RF Reimplementation
Reimplementation and validation of the van Galen et al. two-stage Random Forest classifier. Establishes baseline performance for cell type classification and malignancy detection using the original methodology.

### Step 2: Projections
Reference atlas construction using scANVI (single-cell ANnotation using Variational Inference). Projects query AML datasets onto a unified latent space for consistent cell type annotation across multiple cohorts.

### Step 3: Ensemble
Seven-model ensemble classifier for robust malignancy prediction. Combines Random Forest, XGBoost, LightGBM, SVM, MLP, CellTypist, and scANVI through majority voting. Validated via leave-one-patient-out cross-validation and external cohorts.

### Step 4: TargetID
Therapeutic target identification through pseudo-bulk differential expression (DESeq2), gene set enrichment analysis (GSEA), and combinatorial target pair discovery. Identifies cell surface marker pairs with high specificity for leukemic stem cells and high efficacy for CAR-T targeting.