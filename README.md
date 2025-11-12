# AML Therapeutic Target Identification Pipeline

> **⚠️ Work in Progress**: This repository contains research code for an ongoing thesis project. Some analyses are still under development.

Systematic pipeline for identifying therapeutic targets in acute myeloid leukemia (AML) using single-cell RNA-seq data.

## Overview

This project integrates reference atlas projection, ensemble classification, and differential expression analysis to identify cell surface targets for AML therapy. The pipeline processes single-cell data from multiple AML cohorts to discover malignancy-specific biomarkers.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. RF Reimplementation: van Galen Classifier Validation                │
│    Raw Data → Cell Type Classification → Malignancy Detection          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. Projections: Reference Atlas Mapping                                │
│    Reference scANVI Training → Query Projection → Cell Labeling        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Ensemble: Malignancy Prediction (7 ML Models)                       │
│    Individual Classifiers → Majority Voting → Consensus Predictions    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. TargetID: Therapeutic Target Discovery                              │
│    DEG Analysis (Pseudobulk + LMM) → GSEA Enrichment → Candidates      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
Thesis/
├── 1_RF_reimplementation/          # van Galen classifier reimplementation
│   ├── run_pipeline.py              # Two-stage RF classifier
│   └── results/                     # Validation results
│
├── 2_Projections/                   # Reference atlas projection
│   ├── main.py                      # scANVI training
│   ├── process_query.py             # Query projection
│   ├── outputs/                     # Embeddings, models, figures
│   └── utility/                     # Visualization scripts
│
├── 3_Ensemble/                      # Ensemble classification
│   ├── run_ensemble.py              # 7-model ensemble
│   ├── ensemble.py                  # Voting framework
│   ├── individual_predictor/        # Model development
│   ├── pooledLSC/                   # Inference results
│   └── evaluation/                  # Performance metrics
│
├── 4_TargetID/                      # Target identification
│   ├── pseudobulk_deg_DESeq2.py     # Paired pseudo-bulk DEG
│   ├── singlecell_deg_LMM.py        # Single-cell LMM DEG
│   ├── run_gsea.py                  # Pathway enrichment
│   └── DEG_results_*/               # Differential expression
│
├── Reference_raw_hvg.h5ad           # Reference atlas (HVG subset)
├── vanGalen_raw.h5ad                # van Galen AML dataset
└── README.md                        # This file
```

## Step-by-Step Workflow

### Step 1: RF Reimplementation (Validation)

**Purpose**: Validate van Galen's two-stage Random Forest approach

**Key Operations**:
- Classifier 1: 15 normal cell types (feature selection: all genes → 1000)
- Classifier 2: 21 classes (15 normal + 6 malignant-like)
- Balanced sampling for class imbalance

**Output**: Confusion matrices, accuracy metrics, cell-level predictions

**Navigate**: `cd 1_RF_reimplementation && python run_pipeline.py`

---

### Step 2: Projections (Reference Mapping)

**Purpose**: Project query samples onto reference atlas for cell annotation

**Key Operations**:
- Train scANVI on reference (30 latent dims, batch correction)
- Project queries via transfer learning (scANVI surgery)
- KNN label transfer (20 neighbors, distance-weighted)
- UMAP visualization (preserve reference structure)

**Output**: Latent embeddings, cell type predictions with uncertainty, UMAP coordinates

**Navigate**: `cd 2_Projections`
- `python main.py --train` (reference training)
- Edit `process_query.py` and run for each query

---

### Step 3: Ensemble (Malignancy Classification)

**Purpose**: Robust malignancy prediction via 7-model ensemble

**Models**:
1. CellTypist (Logistic Regression)
2. Random Forest (2-stage, 300 trees)
3. SVM (RBF kernel)
4. XGBoost (Gradient boosting)
5. LightGBM (Fast gradient boosting)
6. MLP (3-layer neural network)
7. scANVI (VAE transfer learning)

**Key Operations**:
- HVG selection: 3000 genes (batch-aware)
- Majority voting across models
- Confidence levels: 5+, 6+, 7/7 votes

**Output**: Consensus predictions, per-model predictions, voting breakdown

**Navigate**: `cd 3_Ensemble && python run_ensemble.py`

---

### Step 4: TargetID (Therapeutic Targets)

**Purpose**: Identify malignant-specific therapeutic targets

**DEG Methods**:
1. **Pseudobulk DESeq2**: Paired donor design (`~ Donor + Status`)
2. **Single-cell LMM**: `expression ~ Status + Study + (1|Donor)`

**GSEA Databases**:
- MSigDB Hallmark 2020
- GO Biological Process 2023
- KEGG 2021 Human

**Key Features**:
- Filters: Ribosomal, mitochondrial, housekeeping genes removed
- Surface protein focus: Druggable membrane targets
- Dual validation: Two independent DEG methods

**Output**: DEG lists, volcano plots, enriched pathways, target candidates

**Navigate**: `cd 4_TargetID`
```bash
python singlecell_deg_LMM.py         # Single-cell analysis
python pseudobulk_deg_DESeq2.py      # Pseudobulk analysis
python run_gsea.py                   # Pathway enrichment
```

---

## Data Requirements

> **📁 Data files are NOT included in this repository** due to their large size (~17 GB total). You must obtain them separately.

### Required Files

| File | Size | Description | How to Obtain |
|------|------|-------------|---------------|
| `Reference_raw_hvg.h5ad` | ~773 MB | Reference atlas (HVG-filtered, raw counts)<br>Required columns: `Donor`, `CellType`, `CellType_Broad` | Contact repository owner or use public HSPC atlas |
| `vanGalen_raw.h5ad` | ~120 MB | van Galen AML dataset (raw counts)<br>Required columns: `CellType`, `GroundTruth` | [van Galen et al. 2019](https://doi.org/10.1016/j.cell.2019.01.031)<br>GEO: GSE116256 |
| Query datasets | Various | Multiple AML cohorts for projection<br>Place in `2_Projections/dataset/Queries/` | See publications in citations |

### Expected Directory Structure

After obtaining the data files, your directory should look like:

```
Thesis/
├── Reference_raw_hvg.h5ad          # Place in root directory
├── vanGalen_raw.h5ad               # Place in root directory
│
├── 2_Projections/
│   └── dataset/
│       ├── Queries/                # Query datasets go here
│       │   ├── Henrik_DG.h5ad
│       │   ├── Petti_DG.h5ad
│       │   ├── Ennis.h5ad
│       │   └── ...
│       └── Examples/               # (Optional) Example data
│
└── (code directories as shown above)
```

### Generating Intermediate Files

Most intermediate files (embeddings, models, results) will be generated automatically when you run the pipeline. These are excluded from git via `.gitignore` to keep the repository size manageable.

**Excluded file types**:
- `.h5ad` - Single-cell data objects
- `.pkl` - Saved models
- Large result files (can be regenerated from code)

---

## Key Results

### Validation (Step 1)
- Classifier 1 accuracy: ~85-90% (15 cell types)
- Classifier 2 binary accuracy: ~90-95% (normal vs malignant)
- Reproduced van Galen's hierarchical approach

### Projection (Step 2)
- Successfully projected 11 independent AML datasets
- Cell type annotations with uncertainty quantification
- Batch-corrected embeddings for downstream analysis

### Ensemble (Step 3)
- Internal validation: 90%+ consensus accuracy
- External validation: Generalizes to independent cohorts
- High-confidence predictions: 6+/7 vote threshold

### Target Discovery (Step 4)
- ~50-100 DEGs (FDR<0.05, |log2FC|>1)
- Surface proteins: CD33, IL3RA, CLEC12A, etc.
- Enriched pathways: Immune response, cell cycle ...

---