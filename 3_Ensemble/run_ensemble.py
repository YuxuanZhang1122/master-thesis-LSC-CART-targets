from ensemble import CellTypeEnsemble
from scANVI_predictor import ScANVIPredictor
from randomforest_predictor import RandomForestPredictor
from xgboost_predictor import XGBoostPredictor
from svm_predictor import SVMPredictor
from celltypist_predictor import CellTypistPredictor
from mlp_predictor import MLPPredictor
from lightgbm_predictor import LightGBMPredictor
import logging
import ensemble_analysis_binary
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_ensemble(ref_path=None, query_path=None, output_dir=None, status=None):
    """
    Args:
        ref_path: Path to reference data
        query_path: Path to query data
        output_dir: Output directory
    """

    # Create ensemble with HVG selection at ensemble level
    ensemble = CellTypeEnsemble(ref_path=ref_path, query_path=query_path, n_hvg=3000)

    # Add all predictors with error handling
    added_predictors = []

    # Add CellTypist - Logistic regression
    ensemble.add_predictor(CellTypistPredictor(
        use_SGD=False,
        feature_selection=True,
        max_iter=1000,
        mini_batch=False,
        balance_cell_type=True,
        C=0.6
    ))
    added_predictors.append("CellTypist")

    # Add Random Forest - Two-stage ensemble with feature selection
    ensemble.add_predictor(RandomForestPredictor(
        n_estimators=300,
        max_depth=20,
        n_selected_genes=1500,
        outer_n_estimators=1500,
        subset_size=2000,
        random_state=42
    ))
    added_predictors.append("RandomForest")

    # Add SVM - Support Vector Machine with RBF kernel
    ensemble.add_predictor(SVMPredictor(
        kernel='rbf',
        C=0.5,
        gamma='scale',
        random_state=42
    ))
    added_predictors.append("SVM")

    # Add XGBoost - Gradient boosting with regularization
    ensemble.add_predictor(XGBoostPredictor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.2,
        random_state=42,
        alpha=0.001
    ))
    added_predictors.append("XGBoost")

    # Add LightGBM - Faster optimized gradient boosting
    ensemble.add_predictor(LightGBMPredictor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.2,
        random_state=42,
        num_leaves=20,
        reg_alpha=0.001
    ))
    added_predictors.append("LightGBM")

    # Add MLP - Multi-layer perceptron neural network
    ensemble.add_predictor(MLPPredictor(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=1000,
        early_stopping=True,
        random_state=42,
        alpha=0.008
    ))
    added_predictors.append("MLP")

    # Add scANVI - Variational autoencoder with transfer learning
    ensemble.add_predictor(ScANVIPredictor(status=status))
    added_predictors.append("scANVI")

    # Always train from scratch
    print("Training all models...")
    print("="*60)
    ensemble_results = ensemble.run_ensemble(f"{output_dir}/ensemble_results.h5ad")

    return ensemble_results

if __name__ == "__main__":
    folder = '' # the output folder
    path = f'evaluation/{folder}'
    reference_path = 'internal_validation_train.h5ad.h5ad'
    query_path = 'internal_validation_test.h5ad.h5ad'
    status = 'eval' # eval for labelled query, infer for unlabelled
    os.makedirs(path, exist_ok=True)

    run_ensemble(ref_path=reference_path, query_path=query_path, output_dir=path, status=status)

    # Run analysis/visualization, only during evaluation
    if status == 'eval':
        analyzer = ensemble_analysis_binary.BinaryEnsembleAnalyzer(f"{path}/ensemble_results.h5ad")
        results = analyzer.generate_binary_report(output_dir=path)