"""
Step 4: Classifier 2 - Malignant vs Normal Detection
===================================================================

This script implements the second Random Forest classifier with:
- Best model selection during cross-validation for classifier 2
- Make predictions for other cells in the full dataset with classifier 2, benchmarking against published labels from van Galen
- Make predictions for other cells in the same way van Galen did with clf1 AND clf2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import warnings
from balanced_rf import BalancedRandomForest

warnings.filterwarnings('ignore')

def perform_cross_validation(X, y, le):
    y_encoded = le.transform(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    all_models = []

    print("Running 5-fold cross-validation...")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y_encoded[train_idx], y_encoded[test_idx]

        # Train fold model
        rf_fold = BalancedRandomForest(n_estimators=1000, n_sample_per_class=50,
                                      random_state=42 + fold_idx, track_feature_importance=False)
        rf_fold.fit(X_train_fold, y_train_fold)

        # Predict on test set
        y_pred_fold = rf_fold.predict(X_test_fold)

        # Calculate 21-class metrics
        accuracy_21class = accuracy_score(y_test_fold, y_pred_fold)

        # Binary classification metrics (normal vs malignant-like)
        test_binary_true = np.array(['malignant' if le.classes_[label].endswith('-like') else 'normal'
                                   for label in y_test_fold])
        test_binary_pred = np.array(['malignant' if le.classes_[pred].endswith('-like') else 'normal'
                                   for pred in y_pred_fold])

        accuracy_binary = accuracy_score(test_binary_true, test_binary_pred)

        # Calculate sensitivity and specificity
        tn = np.sum((test_binary_true == 'normal') & (test_binary_pred == 'normal'))
        fp = np.sum((test_binary_true == 'normal') & (test_binary_pred == 'malignant'))
        fn = np.sum((test_binary_true == 'malignant') & (test_binary_pred == 'normal'))
        tp = np.sum((test_binary_true == 'malignant') & (test_binary_pred == 'malignant'))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'model': rf_fold,
            'accuracy_21class': accuracy_21class,
            'accuracy_binary': accuracy_binary,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'test_idx': test_idx,
            'y_test': y_test_fold,
            'y_pred': y_pred_fold,
            'test_binary_true': test_binary_true,
            'test_binary_pred': test_binary_pred
        }

        fold_results.append(fold_result)
        all_models.append(rf_fold)

    # Select best model based on 21-class accuracy
    best_fold = max(fold_results, key=lambda x: x['accuracy_21class'])
    best_model = best_fold['model']
    best_test_idx = best_fold['test_idx']
    best_y_test = best_fold['y_test']
    best_y_pred = best_fold['y_pred']
    best_test_binary_true = best_fold['test_binary_true']
    best_test_binary_pred = best_fold['test_binary_pred']

    print(f"Best CV fold: {best_fold['fold']} (21-class accuracy: {best_fold['accuracy_21class']:.4f})")

    # Return best model and its TEST DATA ONLY
    return (best_model, best_fold['accuracy_21class'],
            best_fold['sensitivity'], best_fold['specificity'],
            best_test_binary_true, best_test_binary_pred,
            best_test_idx, best_y_test, best_y_pred)

def create_confusion_matrices(y_true_21class, y_pred_21class, y_true_binary, y_pred_binary,
                             sensitivity, specificity, accuracy_21class, title_prefix="", filename="confusion_matrix.png"):
    """Create confusion matrices for both 21-class and binary classification

    Args:
        y_true_21class: True 21-class labels
        y_pred_21class: Predicted 21-class labels
        y_true_binary: True binary labels
        y_pred_binary: Predicted binary labels
        sensitivity: Sensitivity score
        specificity: Specificity score
        accuracy_21class: 21-class accuracy
        title_prefix: Prefix for plot titles
        filename: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 21-class confusion matrix
    cm_21class = confusion_matrix(y_true_21class, y_pred_21class)
    all_labels = sorted(list(set(y_true_21class) | set(y_pred_21class)))
    normal_labels = [label for label in all_labels if not label.endswith('-like')]
    malignant_labels = [label for label in all_labels if label.endswith('-like')]
    ordered_labels = sorted(normal_labels) + sorted(malignant_labels)

    # Reorder confusion matrix
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    reorder_idx = [label_to_idx[label] for label in ordered_labels if label in label_to_idx]

    if len(reorder_idx) > 0:
        cm_21class = cm_21class[np.ix_(reorder_idx, reorder_idx)]
        ordered_labels = [ordered_labels[i] for i in range(len(reorder_idx))]

    # Plot 21-class confusion matrix
    sns.heatmap(cm_21class, annot=True, fmt='d', cmap='Blues',
                xticklabels=ordered_labels, yticklabels=ordered_labels, ax=ax1)

    title_21class = f'{title_prefix}21-Class Confusion Matrix\nAccuracy: {accuracy_21class:.1%}'
    if len(y_true_21class) > 1000:
        title_21class += f' (n={len(y_true_21class):,})'
    ax1.set_title(title_21class, fontsize=14, pad=20)
    ax1.set_xlabel('Predicted Cell Type', fontsize=12)
    ax1.set_ylabel('True Cell Type', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)

    # Add lines to separate normal and malignant cell types
    n_normal = len([label for label in ordered_labels if not label.endswith('-like')])
    if n_normal > 0 and len([label for label in ordered_labels if label.endswith('-like')]) > 0:
        ax1.axvline(x=n_normal, color='red', linewidth=2)
        ax1.axhline(y=n_normal, color='red', linewidth=2)

    # Binary confusion matrix
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=['normal', 'malignant'])
    cm_binary_pct = cm_binary.astype('float') / cm_binary.sum(axis=1)[:, np.newaxis] * 100

    annotations = np.array([[f'{count}\n({pct:.1f}%)'
                           for count, pct in zip(row_counts, row_pcts)]
                          for row_counts, row_pcts in zip(cm_binary, cm_binary_pct)])

    sns.heatmap(cm_binary, annot=annotations, fmt='', cmap='Oranges',
                xticklabels=['Normal', 'Malignant'], yticklabels=['Normal', 'Malignant'], ax=ax2)

    ax2.set_title(f'{title_prefix}Binary Classification\nSensitivity: {sensitivity:.1%} | Specificity: {specificity:.1%}',
                 fontsize=14, pad=20)
    ax2.set_xlabel('Predicted Classification', fontsize=12)
    ax2.set_ylabel('True Classification', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def van_galen_complete_pipeline():
    # Load full van Galen dataset
    adata_full = sc.read('../vanGalen_raw.h5ad')
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    # Load normal cell training data (for Classifier 1)
    adata_normal = sc.read('results/data_processed_normal.h5ad')

    # Load 21-class training data (for Classifier 2)
    adata_21class = sc.read('results/training_data_21class.h5ad')
    print(f"Loaded datasets: {adata_full.n_obs} total cells")

    # Train Classifier 1 (normal cells with feature selection)
    # Extract normal cell data
    X_normal = adata_normal.X.toarray() if hasattr(adata_normal.X, 'toarray') else adata_normal.X
    y_normal = adata_normal.obs['GroundTruth'].values
    gene_names_normal = adata_normal.var.index.values

    # Encode labels for Classifier 1
    from sklearn.preprocessing import LabelEncoder
    le_clf1 = LabelEncoder()
    y_normal_encoded = le_clf1.fit_transform(y_normal)

    # Train outer classifier for feature selection
    rf_outer = BalancedRandomForest(n_estimators=1000, n_sample_per_class=50, random_state=42, track_feature_importance=True)
    rf_outer.fit(X_normal, y_normal_encoded)

    # Select top 1000 genes
    feature_importance_df = pd.DataFrame({
        'gene': gene_names_normal,
        'importance': rf_outer.feature_importances_
    }).sort_values('importance', ascending=False)
    selected_genes = feature_importance_df.head(1000)['gene'].values

    # Train final Classifier 1 on selected genes
    gene_mask = np.isin(gene_names_normal, selected_genes)
    X_normal_selected = X_normal[:, gene_mask]
    classifier1 = BalancedRandomForest(n_estimators=1000, n_sample_per_class=50, random_state=42, track_feature_importance=True)
    classifier1.fit(X_normal_selected, y_normal_encoded)
    print(f"Classifier 1 OOB: {classifier1.oob_score_:.4f}")

    # Train Classifier 2 (21-class data)
    # Extract 21-class data
    X_21class = adata_21class.X.toarray() if hasattr(adata_21class.X, 'toarray') else adata_21class.X
    y_21class = adata_21class.obs['label_21class'].values
    gene_names_21class = adata_21class.var.index.values

    # Encode labels for Classifier 2
    le_clf2 = LabelEncoder()
    y_21class_encoded = le_clf2.fit_transform(y_21class)

    # Train Classifier 2
    classifier2 = BalancedRandomForest(n_estimators=1000, n_sample_per_class=50, random_state=42, track_feature_importance=False)
    classifier2.fit(X_21class, y_21class_encoded)

    # Prepare full dataset for Classifier 1 (use selected genes)
    gene_names_full = adata_full.var.index.values
    gene_mask_2 = np.isin(gene_names_full, selected_genes) # same process for training
    X_clf1 = adata_full.X.toarray() if hasattr(adata_full.X, 'toarray') else adata_full.X
    X_clf1 = X_clf1[:, gene_mask_2]

    # Prepare full dataset for Classifier 2 (use same genes as training)
    clf2_genes_in_full = [g for g in gene_names_21class if g in adata_full.var.index]
    clf2_gene_positions = [adata_full.var.index.get_loc(g) for g in clf2_genes_in_full]
    adata_clf2 = adata_full[:, clf2_gene_positions].copy()
    X_clf2 = adata_clf2.X.toarray() if hasattr(adata_clf2.X, 'toarray') else adata_clf2.X

    """
    Direct Classifier 2 predictions on full dataset
    """

    # Get direct clf2 predictions on full dataset
    clf2_direct_predictions = classifier2.predict(X_clf2)
    clf2_direct_labels = le_clf2.inverse_transform(clf2_direct_predictions)

    # Calculate performance metrics vs original CellType labels
    original_celltypes = adata_full.obs['CellType'].values
    clf2_direct_accuracy = accuracy_score(original_celltypes, clf2_direct_labels)

    # Binary classification metrics for direct predictions
    clf2_true_binary = np.array(['malignant' if label.endswith('-like') else 'normal' for label in original_celltypes])
    clf2_pred_binary = np.array(['malignant' if label.endswith('-like') else 'normal' for label in clf2_direct_labels])

    # Calculate sensitivity and specificity for direct predictions
    tn = np.sum((clf2_true_binary == 'normal') & (clf2_pred_binary == 'normal'))
    fp = np.sum((clf2_true_binary == 'normal') & (clf2_pred_binary == 'malignant'))
    fn = np.sum((clf2_true_binary == 'malignant') & (clf2_pred_binary == 'normal'))
    tp = np.sum((clf2_true_binary == 'malignant') & (clf2_pred_binary == 'malignant'))

    clf2_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    clf2_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Generate confusion matrices for direct clf2 predictions
    create_confusion_matrices(original_celltypes, clf2_direct_labels,
                             clf2_true_binary, clf2_pred_binary,
                             clf2_sensitivity, clf2_specificity, clf2_direct_accuracy,
                             title_prefix="Direct Classifier 2: ",
                             filename='results/classifier2_direct_full_dataset.png')

    """
    Apply two-stage classification to full dataset
    """

    # ====================================================================
    # CONFIGURATION: Choose malignancy detection method
    # ====================================================================
    USE_PROBABILITY_AGGREGATION = False # Toggle this to use aggregate

    # Get Classifier 2 probabilities (needed for VERSION 2)
    clf2_probabilities = classifier2.predict_proba(X_clf2)

    # Identify malignant vs normal class indices in Classifier 2
    malignant_indices = [i for i, cls in enumerate(le_clf2.classes_) if cls.endswith('-like')]
    normal_indices = [i for i, cls in enumerate(le_clf2.classes_) if not cls.endswith('-like')]

    if USE_PROBABILITY_AGGREGATION:
        # ====================================================================
        # VERSION 2: Probability Aggregation
        # ====================================================================
        # Sum probabilities across all malignant vs all normal classes

        malignant_scores = clf2_probabilities[:, malignant_indices].sum(axis=1)
        normal_scores = clf2_probabilities[:, normal_indices].sum(axis=1)
        is_malignant_clf2 = malignant_scores > normal_scores

    else:
        # ====================================================================
        # VERSION 1: Original Argmax Approach
        # ====================================================================
        # Use the single highest probability class from Classifier 2

        clf2_predictions = classifier2.predict(X_clf2)
        clf2_labels = le_clf2.inverse_transform(clf2_predictions)
        is_malignant_clf2 = np.array([label.endswith('-like') for label in clf2_labels])

        # Compute scores for comparison (optional, for analysis)
        malignant_scores = clf2_probabilities[:, malignant_indices].sum(axis=1)
        normal_scores = clf2_probabilities[:, normal_indices].sum(axis=1)

    # Stage 2: Cell type assignment based on Classifier 2 malignancy determination
    clf1_predictions = classifier1.predict(X_clf1)
    clf1_probabilities = classifier1.predict_proba(X_clf1)

    # Define HSC-to-myeloid cell types for malignant cells
    hsc_myeloid_types = ['HSC', 'Prog', 'GMP', 'ProMono', 'Mono', 'cDC']
    hsc_myeloid_indices = [i for i, cls in enumerate(le_clf1.classes_) if cls in hsc_myeloid_types]

    final_labels = []
    for i in range(len(adata_full)):
        if is_malignant_clf2[i]:
            # If malignant: use highest score among HSC-to-myeloid types from Classifier 1
            hsc_myeloid_probs = clf1_probabilities[i][hsc_myeloid_indices]
            best_hsc_myeloid_idx = np.argmax(hsc_myeloid_probs)
            final_labels.append(le_clf1.classes_[hsc_myeloid_indices[best_hsc_myeloid_idx]] + '-like')
        else:
            # If normal: use highest score from Classifier 1
            final_labels.append(le_clf1.classes_[clf1_predictions[i]])

    final_labels = np.array(final_labels)

    # Calculate performance metrics
    original_celltypes = adata_full.obs['CellType'].values
    overall_accuracy = accuracy_score(original_celltypes, final_labels)

    # Binary classification metrics
    true_binary = np.array(['malignant' if label.endswith('-like') else 'normal' for label in original_celltypes])
    pred_binary = np.array(['malignant' if label.endswith('-like') else 'normal' for label in final_labels])

    # Calculate sensitivity and specificity
    tn = np.sum((true_binary == 'normal') & (pred_binary == 'normal'))
    fp = np.sum((true_binary == 'normal') & (pred_binary == 'malignant'))
    fn = np.sum((true_binary == 'malignant') & (pred_binary == 'normal'))
    tp = np.sum((true_binary == 'malignant') & (pred_binary == 'malignant'))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results_df = pd.DataFrame({
        'cell_id': adata_full.obs.index,
        'true_label': original_celltypes,
        'pred_label': final_labels,
        'correct_21class': (original_celltypes == final_labels),
        'true_binary': true_binary,
        'pred_binary': pred_binary,
        'correct_binary': (true_binary == pred_binary),
        'classifier2_malignant': is_malignant_clf2,
        'clf2_malignant_score': malignant_scores,
        'clf2_normal_score': normal_scores
    })

    results_df.to_csv('results/van_galen_classifier_results.csv', index=False)

    # Generate confusion matrices
    create_confusion_matrices(original_celltypes, final_labels,
                             true_binary, pred_binary,
                             sensitivity, specificity, overall_accuracy,
                             title_prefix="Full Dataset: ",
                             filename='results/hierarchical_full_dataset.png')

    return

def main():

    # Load 21-class training data from AnnData
    adata_21class = sc.read('results/training_data_21class.h5ad')

    # Extract expression data and labels
    X = adata_21class.X.toarray() if hasattr(adata_21class.X, 'toarray') else adata_21class.X
    y = adata_21class.obs['label_21class'].values

    # Encode labels
    le = LabelEncoder()
    le.fit(y)

    # Cross-validation (for evaluation only)
    (cv_best_model, best_accuracy_21class, binary_sensitivity, binary_specificity,
     true_binary, pred_binary, best_test_idx, best_y_test, best_y_pred) = perform_cross_validation(X, y, le)

    # Get test data labels for confusion matrix (TEST DATA ONLY)
    y_test_labels = y[best_test_idx]
    y_pred_labels = le.inverse_transform(best_y_pred)

    # Generate confusion matrices using ONLY test data from best fold
    create_confusion_matrices(y_test_labels, y_pred_labels,
                             true_binary, pred_binary, binary_sensitivity, binary_specificity, best_accuracy_21class,
                             title_prefix="Classifier 2 (Best CV Fold - Test Data Only)\n",
                             filename='results/classifier2_cv_best_test_only.png')

    # Apply van Galen classification pipeline
    van_galen_complete_pipeline()

if __name__ == "__main__":
    main()