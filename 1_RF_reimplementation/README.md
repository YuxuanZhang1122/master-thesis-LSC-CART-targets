# 1_RF_reimplementation

Reimplementation of van Galen et al.'s two-stage Random Forest classifier for distinguishing malignant from normal cells in AML.

## Workflow

```
Raw Data → Normalize → Classifier 1 (Cell Type) → Classifier 2 (Malignancy) → Validation
                        (Feature Selection)       (21-class: 15 normal + 6 malignant-like)
```

## Input

| File | Location | Description |
|------|----------|-------------|
| `vanGalen_raw.h5ad` | `../vanGalen_raw.h5ad` | van Galen AML dataset (raw counts) |

**Required metadata columns**:
- `CellType`: Fine-grained cell type annotations
- `GroundTruth`: normal/mutant/unknown labels

## Output

| Directory | Contents |
|-----------|----------|
| `results/` | `classifier1_confusion_matrix_best.png` - Classifier 1 performance<br>`classifier2_cv_best_test_only.png` - Classifier 2 CV performance<br>`classifier2_direct_full_dataset.png` - Direct 21-class results<br>`hierarchical_full_dataset.png` - Two-stage pipeline results<br>`van_galen_classifier_results.csv` - Cell-level predictions |

## Scripts

### Pipeline Execution

**`run_pipeline.py`** - Orchestrates all steps
```bash
# Run complete pipeline
python run_pipeline.py

# Run specific step only (1-4)
python run_pipeline.py 2
```

### Individual Steps

**`step1_prepare_data.py`** - Data preprocessing
- Filters to normal cells (15 cell types)
- Normalizes to Cp10k + log-transform
- Filters genes by mean expression > 0.01

**`step2_classifier1.py`** - Cell type classifier (15 classes)
- Two-stage feature selection:
  - Outer RF (1000 trees) on all genes
  - Inner RF (1500 trees) on top 1000 genes
- 5-fold cross-validation
- Saves final model for downstream use

**`step3_prepare_malignant.py`** - Training data preparation
- Applies Classifier 1 to malignant cells
- Creates 21-class dataset:
  - 15 normal cell types
  - 6 malignant-like types (CellType-like)

**`step4_classifier2.py`** - Malignancy classifier (21 classes)
- Trains on 21-class data (normal + malignant-like)
- 5-fold CV with best model selection
- Tests two approaches:
  - **Direct**: Apply Classifier 2 to full dataset
  - **Hierarchical**: Classifier 1 → Classifier 2 (van Galen approach)

### Utilities

**`balanced_rf.py`** - Balanced Random Forest implementation
- Subsamples majority classes for balance
- Customized for imbalanced cell type distributions

## Configuration

**Classifier 1**:
- Outer RF: 1000 trees, balanced sampling (min 50 cells/class)
- Feature selection: Top 1000 genes by importance
- Inner RF: 1500 trees, balanced sampling

**Classifier 2**:
- RF: 1000 trees, 50 cells/class sampling
- Cross-validation: 5-fold stratified
- Evaluation: 21-class accuracy + binary (normal/malignant) metrics

## Output Format

**Predictions CSV** (`van_galen_classifier_results.csv`):
- `cell_id`: Cell barcode
- `original_celltype`: Ground truth cell type
- `ground_truth`: normal/mutant label
- `clf2_direct_pred`: Direct Classifier 2 prediction
- `clf2_direct_binary`: Direct binary prediction (normal/malignant)
- `clf1_pred`: Classifier 1 prediction
- `clf2_hierarchical_pred`: Hierarchical prediction
- `clf2_hierarchical_binary`: Hierarchical binary prediction

## Performance Metrics

**Confusion Matrices**:
- Classifier 1: 15 normal cell types
- Classifier 2 (21-class): Normal vs malignant-like
- Binary: Normal vs malignant aggregation

**Reported Metrics**:
- Accuracy (21-class and binary)
- Sensitivity (malignant detection)
- Specificity (normal identification)

## Validation Strategy

1. **Classifier 1**: 5-fold CV on normal cells
2. **Classifier 2**: 5-fold CV on 21-class training data
3. **Full dataset**: Compare against van Galen ground truth labels
