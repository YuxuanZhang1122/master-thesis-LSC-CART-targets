import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

def moving_average_smoothing(expression_data, positions, window_size=50, method='uniform'):
    """
    Apply moving average smoothing to expression data based on genomic positions
    
    Parameters:
    -----------
    expression_data : array
        Expression values for genes
    positions : array
        Genomic positions for genes
    window_size : int
        Number of genes to include in moving window
    method : str
        'uniform' for simple moving average, 'gaussian' for weighted
    
    Returns:
    --------
    smoothed_expression : array
        Smoothed expression values
    """
    
    # Sort by genomic position
    sorted_indices = np.argsort(positions)
    sorted_expr = expression_data[sorted_indices]

    if method == 'uniform':
        # Simple moving average, takes each data point and averages it with its neighbors within the window
        smoothed = uniform_filter1d(sorted_expr, size=window_size, mode='nearest')
    elif method == 'gaussian':
        # Gaussian weighted moving average, points closer to the center get more weight, distant points get less
        sigma = window_size / 6  # 99.7% of weight within window_size
        smoothed = gaussian_filter1d(sorted_expr, sigma=sigma, mode='nearest')
    
    # Return to original order
    original_order = np.argsort(sorted_indices)
    smoothed_original_order = smoothed[original_order]
    
    return smoothed_original_order

def compare_smoothing_methods(expression_data, positions, gene_names, cell_name):
    """
    Compare uniform and gaussian smoothing methods
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Smoothing Methods Comparison - {cell_name}', fontsize=14)
    
    # Sort for plotting
    sorted_indices = np.argsort(positions)
    sorted_pos = positions[sorted_indices] / 1e6  # Convert to Mb
    sorted_expr = expression_data[sorted_indices]
    
    # Different smoothing methods
    methods = ['uniform', 'gaussian']
    method_names = ['Uniform (Moving Average)', 'Gaussian Weighted']
    colors = ['blue', 'red']
    
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        smoothed = moving_average_smoothing(expression_data, positions, window_size=50, method=method)
        sorted_smoothed = smoothed[sorted_indices]
        
        axes[i].plot(sorted_pos, sorted_expr, 'o', alpha=0.3, markersize=1, color='gray', label='Raw')
        axes[i].plot(sorted_pos, sorted_smoothed, '-', alpha=0.8, linewidth=2, color=color, label=f'{name} Smoothed')
        axes[i].set_title(f'{name} Smoothing')
        axes[i].set_xlabel('Position (Mb)')
        axes[i].set_ylabel('Expression')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Load datasets
print("Loading datasets...")
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

# Get cell types and select representative cells
cell_types = adata_original.obs['CellType']

# Find HSC and Prog-like cells
hsc_cells = [i for i, ct in enumerate(cell_types) if str(ct).lower() == 'hsc']
proglike_cells = [i for i, ct in enumerate(cell_types) if str(ct).lower() == 'prog-like']

# Select same cells as before for consistency
normal_cell_id = 'AML328-D29_CGGGCACGATAG'  # HSC cell
malignant_cell_id = 'AML328-D29_CACACGCTGTAG'  # Prog-like cell

print(f"Selected cells for smoothed analysis:")
print(f"  Normal (HSC): {normal_cell_id}")
print(f"  Malignant (Prog-like): {malignant_cell_id}")

# Get Chr6 genes and positions
chr6_genes = gene_positions[gene_positions['chromosome'] == '6']['gene_symbol'].tolist()
chr6_genes_in_data = [g for g in chr6_genes if g in adata_filtered.var_names]
chr6_positions = gene_positions[gene_positions['gene_symbol'].isin(chr6_genes_in_data)].copy()
chr6_positions = chr6_positions.sort_values('start').reset_index(drop=True)

print(f"Chr6 genes for analysis: {len(chr6_genes_in_data)}")

# Extract expression data
normal_chr6_expr = adata_filtered[adata_filtered.obs_names == normal_cell_id, chr6_genes_in_data].X.toarray().flatten()
malignant_chr6_expr = adata_filtered[adata_filtered.obs_names == malignant_cell_id, chr6_genes_in_data].X.toarray().flatten()
positions_array = chr6_positions['start'].values

print("\n" + "="*60)
print("APPLYING SMOOTHING ALGORITHMS")
print("="*60)

# Compare different smoothing methods for both cells
print("Comparing smoothing methods for Normal cell...")
fig_normal = compare_smoothing_methods(normal_chr6_expr, positions_array, chr6_genes_in_data, "Normal (HSC)")
plt.savefig('cnv_analysis/smoothing_comparison_normal.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparing smoothing methods for Malignant cell...")
fig_malignant = compare_smoothing_methods(malignant_chr6_expr, positions_array, chr6_genes_in_data, "Malignant (Prog-like)")
plt.savefig('cnv_analysis/smoothing_comparison_malignant.png', dpi=300, bbox_inches='tight')
plt.show()

# Apply different window sizes and methods
window_sizes = [20, 50, 100]
smoothing_methods = ['uniform', 'gaussian']

print("\n" + "="*60)
print("SMOOTHED EXPRESSION ANALYSIS")
print("="*60)

for method in smoothing_methods:
    print(f"\n--- {method.upper()} SMOOTHING ---")
    
    fig, axes = plt.subplots(len(window_sizes), 2, figsize=(16, 4*len(window_sizes)))
    if len(window_sizes) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Chr6 Smoothed Expression Comparison - {method.capitalize()} Method', fontsize=16)
    
    for i, window_size in enumerate(window_sizes):
        # Apply smoothing
        normal_smoothed = moving_average_smoothing(normal_chr6_expr, positions_array, 
                                                 window_size=window_size, method=method)
        malignant_smoothed = moving_average_smoothing(malignant_chr6_expr, positions_array, 
                                                    window_size=window_size, method=method)
        
        # Sort by position for plotting
        sorted_indices = np.argsort(positions_array)
        sorted_pos = positions_array[sorted_indices] / 1e6  # Convert to Mb
        
        # Plot smoothed comparison
        axes[i,0].plot(sorted_pos, normal_chr6_expr[sorted_indices], 'o', alpha=0.3, markersize=1, color='lightblue', label='Raw Normal')
        axes[i,0].plot(sorted_pos, malignant_chr6_expr[sorted_indices], 'o', alpha=0.3, markersize=1, color='lightcoral', label='Raw Malignant')
        axes[i,0].plot(sorted_pos, normal_smoothed[sorted_indices], '-', linewidth=3, color='blue', label='Smoothed Normal')
        axes[i,0].plot(sorted_pos, malignant_smoothed[sorted_indices], '-', linewidth=3, color='red', label='Smoothed Malignant')
        axes[i,0].set_title(f'Smoothed Expression (Window={window_size})')
        axes[i,0].set_xlabel('Position (Mb)')
        axes[i,0].set_ylabel('Expression')
        axes[i,0].legend()
        axes[i,0].grid(True, alpha=0.3)
        
        # Plot smoothed difference
        smoothed_diff = malignant_smoothed - normal_smoothed
        colors = ['red' if x > 0 else 'blue' for x in smoothed_diff[sorted_indices]]
        axes[i,1].scatter(sorted_pos, smoothed_diff[sorted_indices], c=colors, alpha=0.7, s=20)
        axes[i,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[i,1].set_title(f'Smoothed Expression Difference (Window={window_size})')
        axes[i,1].set_xlabel('Position (Mb)')
        axes[i,1].set_ylabel('Expression Difference')
        axes[i,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'cnv_analysis/chr6_smoothed_comparison_{method}.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)