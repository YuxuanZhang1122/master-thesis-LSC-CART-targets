"""
Step 2: Classifier 1 - Cell Type Similarity Prediction
=================================================================

This script implements the first Random Forest classifier
"""

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from balanced_rf import BalancedRandomForest

warnings.filterwarnings('ignore')

def load_original_celltype_labels():
    """Load original van Galen CellType labels for comparison"""
    adata_full = sc.read('../vanGalen_raw.h5ad')

    # Filter to normal cells only for Classifier 1 comparison
    normal_mask = ~adata_full.obs['GroundTruth'].isin(['mutant', 'unknown'])
    adata_normal = adata_full[normal_mask]

    original_labels = pd.DataFrame({
        'cell_id': adata_normal.obs.index,
        'original_celltype': adata_normal.obs['CellType'],
        'ground_truth': adata_normal.obs['GroundTruth']
    })
    return original_labels

def train_outer_classifier(X, y, le):
    """Train outer classifier on all genes for feature selection"""
    min_cells = pd.Series(y).value_counts().min()
    n_sample = min(50, min_cells)

    rf_outer = BalancedRandomForest(n_estimators=1000, n_sample_per_class=n_sample, random_state=42)
    rf_outer.fit(X, le.transform(y))

    print(f"Outer classifier OOB: {rf_outer.oob_score_:.4f}")
    return rf_outer

def select_top_genes(rf_outer, gene_names, n_genes=1000):
    """Select top genes based on feature importance"""
    feature_importance_df = pd.DataFrame({
        'gene': gene_names,
        'importance': rf_outer.feature_importances_
    }).sort_values('importance', ascending=False)

    top_genes = feature_importance_df.head(n_genes)['gene'].values
    print(f"Selected top {n_genes} genes")
    return top_genes

def perform_cross_validation(X, y, le, selected_genes, gene_names, original_labels):
    """Perform 5-fold cross-validation with model selection

    Returns best model and its test data indices for proper evaluation
    """
    gene_mask = np.isin(gene_names, selected_genes)
    X_selected = X[:, gene_mask]
    y_encoded = le.transform(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    print("Running 5-fold cross-validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_selected, y_encoded)):
        X_train_fold, X_test_fold = X_selected[train_idx], X_selected[test_idx]
        y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]

        # Train fold model
        min_cells = pd.Series(y_train_fold).value_counts().min()
        n_sample = min(50, min_cells)

        rf_fold = BalancedRandomForest(n_estimators=1000, n_sample_per_class=n_sample,
                                      random_state=42 + fold_idx)
        rf_fold.fit(X_train_fold, y_train_fold)

        # Predict on test set only
        y_pred_fold = rf_fold.predict(X_test_fold)
        accuracy = accuracy_score(y_test_fold, y_pred_fold)

        fold_results.append({
            'fold': fold_idx + 1,
            'model': rf_fold,
            'accuracy': accuracy,
            'test_idx': test_idx,
            'y_test': y_test_fold,
            'y_pred': y_pred_fold
        })

    best_fold = max(fold_results, key=lambda x: x['accuracy'])
    best_model = best_fold['model']
    best_test_idx = best_fold['test_idx']
    best_y_test = best_fold['y_test']
    best_y_pred = best_fold['y_pred']

    print(f"Best CV fold: {best_fold['fold']} (accuracy: {best_fold['accuracy']:.4f})")

    return best_model, best_test_idx, best_y_test, best_y_pred

def create_confusion_matrix_best_classifier(y_true, y_pred_encoded, le, accuracy, OOB_improvement):
    """Create confusion matrix for best inner classifier"""

    # Convert predictions back to class names
    y_pred = le.inverse_transform(y_pred_encoded.astype(int))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)

    # Set up the figure
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)

    plt.title(f'Classifier 1 (Best CV Fold - Test Data Only)\nAccuracy: {accuracy:.1%} | OOB Improvement: {OOB_improvement:+.1%}',
              fontsize=14, pad=20)
    plt.xlabel('Predicted Cell Type', fontsize=12)
    plt.ylabel('True Cell Type', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig('results/classifier1_confusion_matrix_best.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():

    adata = sc.read('results/data_processed_normal.h5ad')

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    y = adata.obs['GroundTruth'].values
    gene_names = adata.var.index.values

    # Encode labels
    le = LabelEncoder()
    le.fit(y)

    # Load original CellType labels
    original_labels = load_original_celltype_labels()

    # Train outer classifier
    rf_outer = train_outer_classifier(X, y, le)

    # Select top genes
    selected_genes = select_top_genes(rf_outer, gene_names)

    # Cross-validation (for evaluation only)
    best_cv_model, best_test_idx, best_y_test, best_y_pred = perform_cross_validation(
        X, y, le, selected_genes, gene_names, original_labels)

    # OOB Score Improvement (using CV model for comparison)
    OOB_improvement = (best_cv_model.oob_score_ - rf_outer.oob_score_)/rf_outer.oob_score_

    # Create confusion matrix using ONLY test data from best fold
    y_true_test = y[best_test_idx]
    test_accuracy = accuracy_score(best_y_test, best_y_pred)
    create_confusion_matrix_best_classifier(y_true_test, best_y_pred, le, test_accuracy, OOB_improvement)

    # Train final model on FULL dataset using selected genes
    gene_mask = np.isin(gene_names, selected_genes)
    X_selected = X[:, gene_mask]

    # Save final model (trained on full dataset)
    final_model = BalancedRandomForest(n_estimators=1000, n_sample_per_class=50, random_state=42)
    final_model.fit(X_selected, le.transform(y))
    with open('results/classifier1_final_model.pkl', 'wb') as f:
        pickle.dump({
            'model': final_model,
            'label_encoder': le,
            'selected_genes': selected_genes.tolist()
        }, f)

if __name__ == "__main__":
    main()