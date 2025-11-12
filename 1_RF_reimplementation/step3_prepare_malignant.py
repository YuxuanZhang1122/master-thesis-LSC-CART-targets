"""
Step 3: Prepare Malignant Cells for Classifier 2
===============================================================

This script prepares the malignant cell data with correct naming, ready for training
- Loads full dataset and applies Classifier 1 to malignant cells (GroundTruth='mutant')
- Applies reclassification rules (60 Prog→HSC, 29 earlyEry→Prog to have enough cells for these)
- Uses "-like" suffix for malignant cells (e.g., "Prog-like", "HSC-like")
- Selects top 6 malignant cell types with sufficient cells (>50)
- Creates 21-class dataset (15 normal + 6 malignant-like)
- Saves training data for Classifier 2
"""

import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import warnings
from balanced_rf import BalancedRandomForest

warnings.filterwarnings('ignore')

def load_and_process_full_data():

    adata_full = sc.read('../vanGalen_raw.h5ad')
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    return adata_full

def apply_classifier1_to_malignant(adata_full):

    with open('results/classifier1_final_model.pkl', 'rb') as f:
        clf1_data = pickle.load(f)

    final_model = clf1_data['model']
    selected_genes = clf1_data['selected_genes']
    le = clf1_data['label_encoder']

    # Filter to selected genes
    common_genes = adata_full.var.index.intersection(selected_genes)
    gene_mask = adata_full.var.index.isin(common_genes)
    adata_filtered = adata_full[:, gene_mask].copy()

    # Get malignant cells
    malignant_mask = adata_filtered.obs['GroundTruth'] == 'mutant'
    adata_malignant = adata_filtered[malignant_mask].copy()

    # Use Classifier 1 to predict malignant cell types
    X_malignant = adata_malignant.X.toarray() if hasattr(adata_malignant.X, 'toarray') else adata_malignant.X
    malignant_predictions_encoded = final_model.predict(X_malignant)
    malignant_probabilities = final_model.predict_proba(X_malignant)

    # Decode predictions to cell type names
    malignant_predictions = le.inverse_transform(malignant_predictions_encoded)

    # Create results DataFrame
    malignant_results = pd.DataFrame({
        'cell_id': adata_malignant.obs.index,
        'predicted_type': malignant_predictions
    })

    # Add probability scores
    for i, cell_type in enumerate(le.classes_):
        malignant_results[f'prob_{cell_type}'] = malignant_probabilities[:, i]

    return malignant_results, adata_filtered

def apply_reclassification_rules(malignant_results):

    # Rule 1: 60 Prog cells with highest HSC scores → HSC-like
    prog_cells = malignant_results[malignant_results['predicted_type'] == 'Prog'].copy()
    prog_sorted = prog_cells.sort_values('prob_HSC', ascending=False)
    reclassify_to_hsc = prog_sorted.head(60).index
    malignant_results.loc[reclassify_to_hsc, 'predicted_type'] = 'HSC'

    # Rule 2: earlyEry cells with higher Prog score than lateEry score → Prog-like
    early_ery_cells = malignant_results[malignant_results['predicted_type'] == 'earlyEry'].copy()
    # Find earlyEry cells where prob_Prog > prob_lateEry
    reclassify_mask = early_ery_cells['prob_Prog'] > early_ery_cells['prob_lateEry']
    reclassify_to_prog = early_ery_cells[reclassify_mask].index
    malignant_results.loc[reclassify_to_prog, 'predicted_type'] = 'Prog'

    return malignant_results

def create_21class_dataset_fixed_naming(malignant_results, adata_filtered):

    # Select top 6 malignant types with >50 cells
    malignant_counts = malignant_results['predicted_type'].value_counts()
    top_6_malignant = malignant_counts[malignant_counts >= 50].head(6)
    selected_malignant_types = top_6_malignant.index.tolist()

    # Filter malignant cells
    selected_malignant_cells = malignant_results[
        malignant_results['predicted_type'].isin(selected_malignant_types)
    ]

    # Get cell data
    normal_mask = ~adata_filtered.obs['GroundTruth'].isin(['mutant', 'unknown'])
    malignant_cell_ids = selected_malignant_cells['cell_id'].values
    malignant_mask = adata_filtered.obs.index.isin(malignant_cell_ids)

    # Combine datasets
    adata_normal = adata_filtered[normal_mask].copy()
    adata_malignant_selected = adata_filtered[malignant_mask].copy()
    adata_combined = sc.concat([adata_normal, adata_malignant_selected])

    # Create 21-class labels with -like naming
    labels_21class = []
    for cell_id in adata_combined.obs.index:
        if cell_id in adata_normal.obs.index:
            # Normal cell - keep original name
            labels_21class.append(adata_combined.obs.loc[cell_id, 'GroundTruth'])
        else:
            # Malignant cell - add "-like" suffix (van Galen convention)
            predicted_type = selected_malignant_cells[
                selected_malignant_cells['cell_id'] == cell_id
            ]['predicted_type'].values[0]
            labels_21class.append(f"{predicted_type}-like")

    adata_combined.obs['labels_21class'] = labels_21class

    # Save as AnnData
    adata_combined.obs['label_21class'] = labels_21class
    adata_combined.obs['is_malignant'] = [label.endswith('-like') for label in labels_21class]
    adata_combined.write('results/training_data_21class.h5ad')

    # Compare with original CellType labels (optional validation)
    if 'CellType' in adata_combined.obs.columns:
        comparison_stats = {}
        for cell_type in selected_malignant_types:
            our_label = f"{cell_type}-like"
            our_count = (adata_combined.obs['label_21class'] == our_label).sum()

            if our_label in adata_combined.obs['CellType'].cat.categories:
                original_count = (adata_combined.obs['CellType'] == our_label).sum()
                comparison_stats[our_label] = {'our': our_count, 'original': original_count}

    return

def main():

    # Load and process full dataset
    adata_full = load_and_process_full_data()

    # Apply Classifier 1 to malignant cells
    malignant_results, adata_filtered = apply_classifier1_to_malignant(adata_full)

    # Apply reclassification rules
    malignant_results = apply_reclassification_rules(malignant_results)

    # Create 21-class dataset with correct naming
    create_21class_dataset_fixed_naming(malignant_results, adata_filtered)

if __name__ == "__main__":
    main()