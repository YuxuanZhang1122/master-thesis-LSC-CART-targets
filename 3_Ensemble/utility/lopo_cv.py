import scanpy as sc
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, accuracy_score
from ..run_ensemble import run_ensemble

def calculate_metrics(y_true, y_pred, positive_label='LSPC'):
    """Calculate binary classification metrics"""
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {'recall': recall, 'precision': precision, 'accuracy': accuracy}

def lopo_cv(data_path, output_dir='lopo_cv_results', fixed_train_patients=None):
    """
    Leave-One-Patient-Out Cross-Validation

    Args:
        data_path: Path to full dataset h5ad
        output_dir: Output directory for results
        fixed_train_patients: List of patient IDs that must always be in training set
    """
    if fixed_train_patients is None:
        fixed_train_patients = ['BM5-34p', 'BM5-34p38n']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    fold_dir = f"{output_dir}/folds"
    os.makedirs(fold_dir, exist_ok=True)

    # Load full dataset
    logger.info(f"Loading dataset from {data_path}")
    adata = sc.read_h5ad(data_path)

    # Get all unique patients
    all_patients = adata.obs['patient_id'].unique().tolist()
    logger.info(f"Total patients: {len(all_patients)}")
    logger.info(f"Fixed training patients: {fixed_train_patients}")

    # Patients to use as test sets (exclude fixed training patients)
    test_patients = [p for p in all_patients if p not in fixed_train_patients]
    logger.info(f"Test patients for LOPO-CV: {len(test_patients)} patients")
    logger.info(f"Patients: {test_patients}")

    # Storage for results
    all_predictions = []
    per_patient_metrics = []

    # Run LOPO-CV
    for fold_idx, test_patient in enumerate(test_patients, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx}/{len(test_patients)}: Testing on {test_patient}")
        logger.info(f"{'='*60}")

        # Split data
        train_mask = adata.obs['patient_id'] != test_patient
        test_mask = adata.obs['patient_id'] == test_patient

        train_data = adata[train_mask].copy()
        test_data = adata[test_mask].copy()

        logger.info(f"Train: {train_data.n_obs} cells from {train_data.obs['patient_id'].nunique()} patients")
        logger.info(f"Test: {test_data.n_obs} cells from patient {test_patient}")
        logger.info(f"Test labels - HSPC: {(test_data.obs['CellType_Merged']=='HSPC').sum()}, "
                   f"LSPC: {(test_data.obs['CellType_Merged']=='LSPC').sum()}")

        # Save temporary train/test files
        train_path = f"{fold_dir}/train_fold{fold_idx}.h5ad"
        test_path = f"{fold_dir}/test_fold{fold_idx}.h5ad"
        train_data.write(train_path)
        test_data.write(test_path)

        # Run ensemble
        fold_output = f"{fold_dir}/fold{fold_idx}_output"
        os.makedirs(fold_output, exist_ok=True)

        logger.info(f"Running ensemble for fold {fold_idx}...")
        run_ensemble(
            ref_path=train_path,
            query_path=test_path,
            output_dir=fold_output,
            status='eval'
        )

        # Load predictions
        results_path = f"{fold_output}/ensemble_results.h5ad"
        results = sc.read_h5ad(results_path)

        # Store predictions with patient ID
        pred_df = results.obs.copy()
        pred_df['patient_id'] = test_patient
        pred_df['fold'] = fold_idx
        all_predictions.append(pred_df)

        # Calculate per-patient metrics
        y_true = pred_df['CellType_Merged']

        # Consensus
        consensus_acc = accuracy_score(y_true, pred_df['consensus_prediction'])

        # Consensus 5 votes (exclude uncertain)
        mask_5 = pred_df['consensus_label_5votes'] != 'uncertain'
        consensus_5_acc = accuracy_score(
            y_true[mask_5],
            pred_df['consensus_label_5votes'][mask_5]
        ) if mask_5.sum() > 0 else np.nan

        # Consensus 6 votes (exclude uncertain)
        mask_6 = pred_df['consensus_label_6votes'] != 'uncertain'
        consensus_6_acc = accuracy_score(
            y_true[mask_6],
            pred_df['consensus_label_6votes'][mask_6]
        ) if mask_6.sum() > 0 else np.nan

        per_patient_metrics.append({
            'fold': fold_idx,
            'test_patient': test_patient,
            'n_cells': len(pred_df),
            'n_HSPC': (y_true == 'HSPC').sum(),
            'n_LSPC': (y_true == 'LSPC').sum(),
            'consensus_accuracy': consensus_acc,
            'consensus_5votes_accuracy': consensus_5_acc,
            'consensus_5votes_n_confident': mask_5.sum(),
            'consensus_6votes_accuracy': consensus_6_acc,
            'consensus_6votes_n_confident': mask_6.sum()
        })

        logger.info(f"Fold {fold_idx} complete - Consensus accuracy: {consensus_acc:.4f}")

        # Clean up temporary files to save space
        os.remove(train_path)
        os.remove(test_path)

    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    per_patient_df = pd.DataFrame(per_patient_metrics)

    # Save per-patient metrics
    per_patient_df.to_csv(f"{output_dir}/per_patient_metrics.csv", index=False)

    # Calculate overall metrics for individual models
    model_names = ['CellTypist', 'RandomForest', 'SVM', 'XGBoost', 'LightGBM', 'MLP', 'scANVI']
    model_metrics = []

    y_true_all = all_pred_df['CellType_Merged']

    for model in model_names:
        pred_col = f"{model}_prediction"
        if pred_col in all_pred_df.columns:
            y_pred = all_pred_df[pred_col]
            metrics = calculate_metrics(y_true_all, y_pred, positive_label='LSPC')
            model_metrics.append({
                'model': model,
                'LSPC_recall': metrics['recall'],
                'LSPC_precision': metrics['precision'],
                'accuracy': metrics['accuracy']
            })

    model_metrics_df = pd.DataFrame(model_metrics)
    model_metrics_df.to_csv(f"{output_dir}/model_metrics.csv", index=False)

    # Generate visualizations
    generate_visualizations(per_patient_df, model_metrics_df, output_dir)

    return all_pred_df, per_patient_df, model_metrics_df

def generate_visualizations(per_patient_df, model_metrics_df, output_dir):
    """Generate accuracy and radar chart visualizations"""

    # 1. Accuracy across test samples
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(per_patient_df))
    width = 0.25

    ax.bar([i - width for i in x], per_patient_df['consensus_accuracy'],
           width, label='Consensus (4+ votes)', alpha=0.8)
    ax.bar(x, per_patient_df['consensus_5votes_accuracy'],
           width, label='Consensus (5+ votes)', alpha=0.8)
    ax.bar([i + width for i in x], per_patient_df['consensus_6votes_accuracy'],
           width, label='Consensus (6+ votes)', alpha=0.8)

    ax.set_xlabel('Test Patient', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('LOPO-CV: Accuracy Across Test Samples', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(per_patient_df['test_patient'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_per_patient.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_per_patient.pdf", bbox_inches='tight')
    plt.close()
    logger.info(f"Saved accuracy visualization to {output_dir}/accuracy_per_patient.png")

if __name__ == "__main__":
    data_path = '../../vanGalen_raw.h5ad'
    output_dir = '../evaluation/lopo_cv_results'

    lopo_cv(
        data_path=data_path,
        output_dir=output_dir,
        fixed_train_patients=['BM5-34p', 'BM5-34p38n']
    )
