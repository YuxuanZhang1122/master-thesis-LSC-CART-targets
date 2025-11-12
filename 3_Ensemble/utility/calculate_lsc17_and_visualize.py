import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# LSC17 genes with alternative names
LSC17_genes_map = {
    'DNMT3B': 'DNMT3B',
    'ZBTB46': 'ZBTB46',
    'NYNRIN': 'NYNRIN',
    'ARHGAP22': 'ARHGAP22',
    'LAPTM4B': 'LAPTM4B',
    'MMRN1': 'MMRN1',
    'DPYSL3': 'DPYSL3',
    'KIAA0125': 'FAM30A',
    'CDK6': 'CDK6',
    'CPXM1': 'CPXM1',
    'SOCS2': 'SOCS2',
    'SMIM24': 'SMIM24',
    'EMP1': 'EMP1',
    'NGFRAP1': 'BEX3',
    'CD34': 'CD34',
    'AKR1C3': 'AKR1C3',
    'GPR56': 'ADGRG1'
}

LSC17_coefficients = {
    'DNMT3B': 0.0874,
    'ZBTB46': -0.0347,
    'NYNRIN': 0.00865,
    'ARHGAP22': -0.0138,
    'LAPTM4B': 0.00582,
    'MMRN1': 0.0258,
    'DPYSL3': 0.0284,
    'KIAA0125': 0.0196,
    'CDK6': -0.0704,
    'CPXM1': -0.0258,
    'SOCS2': 0.0271,
    'SMIM24': -0.0226,
    'EMP1': 0.0146,
    'NGFRAP1': 0.0465,
    'CD34': 0.0338,
    'AKR1C3': -0.0402,
    'GPR56': 0.0501
}

def normalize_and_log_transform(adata):
    """Normalize to CP10K and log2 transform, store in layer"""

    # Normalize to counts per 10K
    total_counts = np.array(adata.X.sum(axis=1)).flatten()

    # Avoid division by zero
    total_counts[total_counts == 0] = 1

    # CP10K normalization
    if hasattr(adata.X, 'toarray'):
        normalized = adata.X.toarray() / total_counts[:, np.newaxis] * 10000
    else:
        normalized = adata.X / total_counts[:, np.newaxis] * 10000

    # Log2 transform
    log_normalized = np.log2(normalized + 1)

    # Store in layer
    adata.layers['log-normalized'] = log_normalized

    return adata

def calculate_lsc17_score(adata, gene_map, coefficients):
    """Calculate LSC17 score for each cell using log-normalized data"""

    # Get actual gene names in the data
    actual_genes = []
    coef_list = []

    for orig_name, actual_name in gene_map.items():
        if actual_name in adata.var_names:
            actual_genes.append(actual_name)
            coef_list.append(coefficients[orig_name])
        else:
            print(f'Warning: {actual_name} not found')

    # Get log-normalized expression matrix for LSC17 genes
    gene_indices = [list(adata.var_names).index(g) for g in actual_genes]
    expr_matrix = adata.layers['log-normalized'][:, gene_indices]

    # Calculate weighted sum
    coef_array = np.array(coef_list)
    lsc17_scores = expr_matrix @ coef_array

    return lsc17_scores, actual_genes

def process_dataset(ds_name, time_points, merged_dir, results):
    """Process a single dataset"""

    print(f'\nProcessing {ds_name}...')
    adata = sc.read_h5ad(merged_dir / f'{ds_name}_merged.h5ad')

    # Normalize to CP10K and log2 transform
    print(f'  Normalizing and log-transforming...')
    adata = normalize_and_log_transform(adata)

    # Calculate LSC17 scores for all cells
    lsc17_scores, genes_used = calculate_lsc17_score(adata, LSC17_genes_map, LSC17_coefficients)
    adata.obs['LSC17_score'] = lsc17_scores

    print(f'  Genes used: {len(genes_used)}/17')

    # For HC controls (Setty, Mende), calculate on whole dataset
    if ds_name in ['Setty', 'Mende']:
        mean_score = np.mean(lsc17_scores)
        std_score = np.std(lsc17_scores)
        n_cells = len(lsc17_scores)

        results.append({
            'dataset': ds_name,
            'time_point': 'All',
            'cell_type': 'HC',
            'mean_LSC17': mean_score,
            'std_LSC17': std_score,
            'n_cells': n_cells,
            'scores': lsc17_scores
        })
        print(f'  {ds_name} (HC): mean={mean_score:.4f}, n={n_cells}')

    else:
        # For AML datasets, split by time_point and consensus_label_6votes
        for tp in time_points:
            # Filter by time point
            tp_mask = adata.obs['time_point'] == tp
            adata_tp = adata[tp_mask]

            # Split by HSPC vs LSPC (drop uncertain)
            for cell_type in ['HSPC', 'LSPC']:
                ct_mask = adata_tp.obs['consensus_label_6votes'] == cell_type

                if ct_mask.sum() > 0:
                    subset_scores = adata_tp.obs.loc[ct_mask, 'LSC17_score'].values
                    mean_score = np.mean(subset_scores)
                    std_score = np.std(subset_scores)
                    n_cells = len(subset_scores)

                    results.append({
                        'dataset': ds_name,
                        'time_point': tp,
                        'cell_type': cell_type,
                        'mean_LSC17': mean_score,
                        'std_LSC17': std_score,
                        'n_cells': n_cells,
                        'scores': subset_scores
                    })
                    print(f'  {ds_name}_{tp}_{cell_type}: mean={mean_score:.4f}, n={n_cells}')

    return adata

def visualize_lsc17_vs_distance(distance_csv_path):
    """Create visualization of LSC17 scores vs distance from HSC"""

    if not Path(distance_csv_path).exists():
        print(f'\nWarning: {distance_csv_path} not found. Skipping visualization.')
        return

    print(f'\nCreating LSC17 vs distance visualization...')
    df = pd.read_csv(distance_csv_path)

    sns.set_style("white")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 3.5))

    n_bins = 40
    bins = np.linspace(df['latent_distance_from_HSC'].min(),
                       df['latent_distance_from_HSC'].max(), n_bins)
    df['distance_bin'] = pd.cut(df['latent_distance_from_HSC'], bins=bins)

    bin_data = df.groupby('distance_bin', observed=True).agg({
        'LSC17_deviation': 'mean',
        'latent_distance_from_HSC': 'mean'
    }).dropna()

    bar_colors = ['#4D4D4D' if x > 0 else '#B8B8B8' for x in bin_data['LSC17_deviation']]
    ax.bar(range(len(bin_data)), bin_data['LSC17_deviation'],
           color=bar_colors, width=1, edgecolor='none', alpha=0.85)

    ax.axhline(0, color='#2D2D2D', linewidth=2, linestyle='-', zorder=3)

    ax.set_xlabel('Distance from HSC', fontsize=14, fontweight='bold')
    ax.set_ylabel('LSC17 Score\n(deviation from median)', fontsize=14, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('diverging_bars.png',
                dpi=600, bbox_inches='tight', transparent=True)

    print("Saved visualization as PNG (600 dpi) and PDF with transparent background")
    plt.close()

# Main processing
merged_dir = Path('LabelTransfer/PopularVote/PooledLSC_merged')

results = []

# Process each dataset with their respective time points
dataset_timepoints = {
    'Setty': [],
    'Mende': [],
    'Ennis': ['DG', 'MRD', 'REL'],
    'Naldini_V03': ['DG', 'MRD', 'REL'],
    'Naldini_V02': ['DG', 'MRD'],
    'Petti': ['DG'],
    'Henrik': ['DG']
}

adata_dict = {}
for ds_name, time_points in dataset_timepoints.items():
    adata = process_dataset(ds_name, time_points, merged_dir, results)
    adata_dict[ds_name] = adata

# Create DataFrame with results
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('lsc17_scores_summary.csv', index=False)

# Save merged AnnData objects with LSC17 scores and log-normalized layer
for ds_name, adata in adata_dict.items():
    output_path = merged_dir / f'{ds_name}_merged.h5ad'
    adata.write_h5ad(output_path, compression='gzip')

# Create visualization with pre-made distance data
distance_csv = 'lsc17_distance_data.csv'
visualize_lsc17_vs_distance(distance_csv)
