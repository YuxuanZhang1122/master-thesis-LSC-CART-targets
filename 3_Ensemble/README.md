# 3_Ensemble

Ensemble classifier combining 7 ML models to predict malignancy in AML cells.

## Workflow

```
Training Data → Individual Models → Majority Vote → Malignancy Predictions → Cell Stratification
```

## Input

| File | Location | Description |
|------|----------|-------------|
| `internal_validation_train.h5ad` | Root | van Galen training set (labeled HSPC/LSPC) |
| `internal_validation_test.h5ad` | Root | van Galen test set (held-out patients) |
| `external_validation_*.h5ad` | Root | Independent datasets (Mende, Petti, Naldini) |
| Projected cells | `../2_Projections/outputs/PooledLSC/` | Query cells for inference |

## Output

| Directory | Contents |
|-----------|----------|
| `evaluation/` | Internal/external validation results<br>Performance metrics per model<br>LOPO-CV results<br>Confusion matrices, ROC curves |
| `pooledLSC/` | Inference results per dataset (e.g., `Ennis.h5ad`, `Petti.h5ad`)<br>`ready_20/`, `ready_30/` - Stratified cells by confidence |
| `pooledLSC/4visualization/` | Malignant distribution plots<br>Summary statistics by dataset/cell type |
| `individual_predictor/` | Model checkpoints and tuning results |

## Scripts

### Core Pipeline

**`run_ensemble.py`** - Main entry point
```python
# Edit paths in script, then:
python run_ensemble.py
```
- Loads reference + query data
- Initializes 7 predictors with tuned hyperparameters
- Runs ensemble voting
- Saves predictions (status='eval' or 'infer')

**`ensemble.py`** - Ensemble framework
- `CellTypeEnsemble` class: coordinates all predictors
- HVG selection (3000 genes)
- Majority voting logic
- Generates consensus predictions (5+ votes, 6+ votes)

### Individual Models (`*_predictor.py`)

| Model | Type | Key Parameters |
|-------|------|----------------|
| **CellTypist** | Logistic Regression | C=0.6, feature selection |
| **RandomForest** | Tree Ensemble | 300 trees, max_depth=20, 2-stage |
| **SVM** | Kernel Method | RBF kernel, C=0.5 |
| **XGBoost** | Gradient Boosting | 100 estimators, lr=0.2 |
| **LightGBM** | Gradient Boosting | 200 estimators, 20 leaves |
| **MLP** | Neural Network | (512, 256, 128), early stopping |
| **scANVI** | VAE Transfer Learning | From Step 2 projections |

### Utilities (`utility/`)

| Script | Purpose |
|--------|---------|
| `lopo_cv.py` | Leave-one-patient-out cross-validation |
| `classify_PooledLSC.py` | Apply ensemble to new datasets |
| `merge_pooled_datasets.py` | Combine datasets for analysis |
| `calculate_lsc17_and_visualize.py` | LSC17 score computation |
| `malignant_distribution_visualize_pooledLSC.py` | Distribution plots |

### Model Development (`individual_predictor/`)

| Script | Purpose |
|--------|---------|
| `main.py` | Train/test individual models |
| `finetune_hyperparameters.py` | Hyperparameter optimization |
| `*_model.py` | Model implementations |

## Configuration

**Ensemble Settings**:
- HVG: 3000 genes (cell_ranger flavor, batch-aware)
- Voting: Majority vote across 7 models
- Confidence levels: 5+ votes, 6+ votes

**Mode**:
- `status='eval'`: Generate metrics for labeled data
- `status='infer'`: Predict unlabeled query cells

## Output Format

**Ensemble results** (`.obs` columns):
- `consensus_prediction`: Majority vote label
- `consensus_label_5votes`: High-confidence (5+/7 votes)
- `consensus_label_6votes`: Very high-confidence (6+/7 votes)
- `{model}_prediction`: Individual model predictions
- `voting_breakdown`: Per-cell vote details
- `true_labels`: Ground truth (eval mode only)