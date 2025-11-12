import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from ..ensemble import CellTypeEnsemble
from ..scANVI_predictor import ScANVIPredictor
from ..randomforest_predictor import RandomForestPredictor
from ..xgboost_predictor import XGBoostPredictor
from ..svm_predictor import SVMPredictor
from ..celltypist_predictor import CellTypistPredictor
from ..mlp_predictor import MLPPredictor
from ..lightgbm_predictor import LightGBMPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

POOLED_DIR = Path('')
OUTPUT_DIR = Path('')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

REF_PATH = '../../vanGalen_raw_HLSPC.h5ad'

def save_results_unlabeled(ensemble, output_path, raw_adata):
    """Save results for unlabeled query data with full raw counts and predictions only"""
    # Start with raw data (full gene set)
    result_adata = raw_adata.copy()

    # Add only the 3 essential prediction columns
    result_adata.obs['consensus_prediction'] = ensemble.consensus_predictions
    consensus_5votes = np.where(ensemble.max_votes >= 5, ensemble.consensus_predictions, "uncertain")
    consensus_6votes = np.where(ensemble.max_votes >= 6, ensemble.consensus_predictions, "uncertain")
    result_adata.obs['consensus_label_5votes'] = consensus_5votes
    result_adata.obs['consensus_label_6votes'] = consensus_6votes

    result_adata.write_h5ad(output_path)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Output: {result_adata.n_obs} cells, {result_adata.n_vars} genes (full raw counts)")

    return result_adata

def run_ensemble_unlabeled(ref_path, query_path, output_dir, raw_adata):
    """Run ensemble on unlabeled query data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    ensemble = CellTypeEnsemble(ref_path=ref_path, query_path=query_path, n_hvg=3000, status='infer')

    # Add all 7 predictors
    ensemble.add_predictor(CellTypistPredictor(
        use_SGD=False, feature_selection=True, max_iter=1000,
        mini_batch=False, balance_cell_type=True, C=0.6
    ))
    ensemble.add_predictor(RandomForestPredictor(
        n_estimators=300, max_depth=20, n_selected_genes=1500,
        outer_n_estimators=1500, subset_size=2000, random_state=42
    ))
    ensemble.add_predictor(SVMPredictor(
        kernel='rbf', C=0.5, gamma='scale', random_state=42
    ))
    ensemble.add_predictor(XGBoostPredictor(
        n_estimators=100, max_depth=6, learning_rate=0.2,
        random_state=42, alpha=0.001
    ))
    ensemble.add_predictor(LightGBMPredictor(
        n_estimators=200, max_depth=6, learning_rate=0.2,
        random_state=42, num_leaves=20, reg_alpha=0.001
    ))
    ensemble.add_predictor(MLPPredictor(
        hidden_layer_sizes=(512, 256, 128), max_iter=1000,
        early_stopping=True, random_state=42, alpha=0.008
    ))
    ensemble.add_predictor(ScANVIPredictor(status='infer'))

    # Run pipeline
    ensemble.load_data()
    ensemble.run_predictions()
    ensemble.majority_vote()

    # Save results with full raw counts
    output_path = output_dir / "ensemble_results.h5ad"
    results = save_results_unlabeled(ensemble, output_path, raw_adata)

    return results

def classify_dataset(dataset_name, query_path, ref_path):
    """Apply ensemble classifier to predict HSPC/LSPC for pooled dataset"""

    output_subdir = OUTPUT_DIR / dataset_name

    logger.info(f"\n{'='*70}")
    logger.info(f"CLASSIFYING {dataset_name}")
    logger.info(f"{'='*70}")

    raw_adata = sc.read_h5ad(query_path)
    logger.info(f"Query cells: {raw_adata.n_obs}")
    logger.info(f"Query genes: {raw_adata.n_vars}")
    if 'cell_type' in raw_adata.obs.columns:
        logger.info(f"Cell types: {raw_adata.obs['cell_type'].value_counts().to_dict()}")
    if 'time_point' in raw_adata.obs.columns:
        logger.info(f"Time points: {raw_adata.obs['time_point'].value_counts().to_dict()}")

    # Run ensemble
    results = run_ensemble_unlabeled(
        ref_path=ref_path,
        query_path=str(query_path),
        output_dir=str(output_subdir),
        raw_adata=raw_adata
    )

    # Summary
    consensus = results.obs['consensus_prediction'].values
    n_hspc = (consensus == 'HSPC').sum()
    n_lspc = (consensus == 'LSPC').sum()
    pct_lspc = n_lspc / len(consensus) * 100

    logger.info(f"\n{dataset_name} OVERALL:")
    logger.info(f"  Total: {len(consensus)} cells")
    logger.info(f"  HSPC (normal): {n_hspc} ({100-pct_lspc:.1f}%)")
    logger.info(f"  LSPC (malignant): {n_lspc} ({pct_lspc:.1f}%)")

    # Per-cell type breakdown
    logger.info(f"\nPER-CELL TYPE BREAKDOWN:")
    cell_types = results.obs['predicted_CellType_Broad'].values
    for ct in sorted(np.unique(cell_types)):
        mask = cell_types == ct
        n_total = mask.sum()
        n_lspc_ct = ((consensus == 'LSPC') & mask).sum()
        n_hspc_ct = ((consensus == 'HSPC') & mask).sum()
        pct_lspc_ct = n_lspc_ct / n_total * 100
        logger.info(f"  {ct}: {n_lspc_ct}/{n_total} malignant ({pct_lspc_ct:.1f}%), "
                   f"{n_hspc_ct} normal ({100-pct_lspc_ct:.1f}%)")

    # Per-time point breakdown if available
    if 'time_point' in results.obs.columns:
        time_points = results.obs['time_point'].values
        unique_tps = sorted(np.unique(time_points))
        if len(unique_tps) > 1:
            logger.info(f"\nPER-TIME POINT BREAKDOWN:")
            for tp in unique_tps:
                mask = time_points == tp
                n_total = mask.sum()
                n_lspc_tp = ((consensus == 'LSPC') & mask).sum()
                n_hspc_tp = ((consensus == 'HSPC') & mask).sum()
                pct_lspc_tp = n_lspc_tp / n_total * 100
                logger.info(f"  {tp}: {n_lspc_tp}/{n_total} malignant ({pct_lspc_tp:.1f}%), "
                           f"{n_hspc_tp} normal ({100-pct_lspc_tp:.1f}%)")

    return results

def main():
    logger.info("="*70)
    logger.info("POOLED LSC MALIGNANCY CLASSIFICATION BY DATASET")
    logger.info("="*70)

    logger.info(f"\nReference data: {REF_PATH}")
    logger.info(f"Query directory: {POOLED_DIR}/")

    # Get all pooled dataset files
    dataset_files = sorted(POOLED_DIR.glob('*_DG.h5ad'))

    if not dataset_files:
        logger.error(f"No dataset files found in {POOLED_DIR}/")
        logger.error("Run merge_pooled_datasets.py first!")
        return

    logger.info(f"Found {len(dataset_files)} datasets:")
    for f in dataset_files:
        logger.info(f"  {f.name}")

    # Classify each dataset (skip vanGalen - already has predictions)
    results_all = {}
    for dataset_file in dataset_files:
        dataset_name = dataset_file.stem.replace('_DG', '')

        if dataset_name == 'vanGalen':
            logger.info(f"\n{'='*70}")
            logger.info(f"SKIPPING {dataset_name} (already has predictions)")
            logger.info(f"{'='*70}")
            # Load vanGalen results directly
            results_all[dataset_name] = sc.read_h5ad(dataset_file)
        else:
            results_all[dataset_name] = classify_dataset(dataset_name, dataset_file, REF_PATH)

    # Overall summary
    logger.info("\n" + "="*70)
    logger.info("OVERALL SUMMARY")
    logger.info("="*70)

    for dataset_name, results in results_all.items():
        consensus = results.obs['consensus_prediction'].values
        n_lspc = (consensus == 'LSPC').sum()
        pct_lspc = n_lspc / len(consensus) * 100
        logger.info(f"{dataset_name}: {n_lspc}/{len(consensus)} malignant ({pct_lspc:.1f}%)")

    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Results saved to: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
