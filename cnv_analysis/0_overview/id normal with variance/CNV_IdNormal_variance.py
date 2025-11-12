import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

def median_absolute_deviation(data, axis=None):
    """Custom implementation of MAD"""
    median = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.abs(data - median), axis=axis)
    return mad

def moving_average_smoothing_by_chromosome(adata, gene_positions, window_size=100):
    """
    Apply uniform smoothing to all cells, chromosome by chromosome
    """
    print("Applying gaussian smoothing to all cells...")
    
    # Initialize smoothed expression matrix
    smoothed_matrix = np.zeros_like(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    
    # Get unique chromosomes
    chromosomes = sorted(gene_positions['chromosome'].unique(), 
                        key=lambda x: (x.isdigit() and int(x) or 99, x))
    
    chr_means = {}  # Store mean expression per chromosome per cell
    
    for chr_name in chromosomes:
        print(f"  Processing chromosome {chr_name}...")
        
        # Get genes for this chromosome
        chr_genes = gene_positions[gene_positions['chromosome'] == chr_name]['gene_symbol'].tolist()
        chr_genes_in_data = [g for g in chr_genes if g in adata.var_names]
        
        if len(chr_genes_in_data) < 10:  # Skip chromosomes with too few genes
            continue
            
        # Get gene indices
        gene_indices = [list(adata.var_names).index(g) for g in chr_genes_in_data]
        
        # Get positions for sorting
        chr_positions = gene_positions[gene_positions['gene_symbol'].isin(chr_genes_in_data)]
        chr_positions = chr_positions.sort_values('start').reset_index(drop=True)
        
        # Get expression data for this chromosome
        chr_expr = adata.X[:, gene_indices].toarray() if hasattr(adata.X, 'toarray') else adata.X[:, gene_indices]
        
        # Sort genes by position
        pos_order = [chr_genes_in_data.index(gene) for gene in chr_positions['gene_symbol']]
        chr_expr_sorted = chr_expr[:, pos_order]
        
        # Apply smoothing to each cell
        smoothed_chr = np.zeros_like(chr_expr_sorted)
        for cell_idx in range(chr_expr_sorted.shape[0]):
            if cell_idx % 200 == 0:
                print(f"    Cell {cell_idx+1}/{chr_expr_sorted.shape[0]}")
            sigma = min(window_size, len(pos_order)) / 6  # 99.7% of weight within window_size
            smoothed_chr[cell_idx, :] = gaussian_filter1d(chr_expr_sorted[cell_idx, :], 
                                                        sigma=sigma, 
                                                        mode='nearest')
        
        # Restore original gene order
        original_order = np.argsort(pos_order)
        smoothed_chr_original = smoothed_chr[:, original_order]
        
        # Store back in main matrix
        smoothed_matrix[:, gene_indices] = smoothed_chr_original
        
        # Calculate mean expression per chromosome per cell
        chr_means[chr_name] = smoothed_chr_original.mean(axis=1)
        
        print(f"    Chromosome {chr_name}: {len(chr_genes_in_data)} genes processed")
    
    return smoothed_matrix, chr_means

def calculate_cnv_variance_metrics(chr_means):
    """
    Calculate variance metrics for each cell across chromosomes
    """
    print("Calculating CNV variance metrics...")
    
    # Convert to matrix (cells x chromosomes)
    chr_names = sorted(chr_means.keys(), key=lambda x: (x.isdigit() and int(x) or 99, x))
    cnv_matrix = np.column_stack([chr_means[chr_name] for chr_name in chr_names])
    
    # Calculate metrics for each cell
    metrics = pd.DataFrame(index=range(cnv_matrix.shape[0]))
    
    # 1. Variance across chromosomes
    metrics['chr_variance'] = np.var(cnv_matrix, axis=1)
    
    # 2. Standard deviation
    metrics['chr_std'] = np.std(cnv_matrix, axis=1)
    
    # 3. Coefficient of variation (std/mean)
    chr_means_per_cell = np.mean(cnv_matrix, axis=1)
    metrics['chr_cv'] = metrics['chr_std'] / (chr_means_per_cell + 1e-8)
    
    # 4. Median absolute deviation
    metrics['chr_mad'] = [median_absolute_deviation(row) for row in cnv_matrix]
    
    # 5. Range (max - min)
    metrics['chr_range'] = np.max(cnv_matrix, axis=1) - np.min(cnv_matrix, axis=1)
    
    # 6. Interquartile range
    q75 = np.percentile(cnv_matrix, 75, axis=1)
    q25 = np.percentile(cnv_matrix, 25, axis=1)
    metrics['chr_iqr'] = q75 - q25
    
    return metrics, cnv_matrix, chr_names

def identify_normal_cells_by_variance(metrics, adata, top_percentile=10):
    """
    Identify normal cells as those with lowest variance across chromosomes
    """
    print(f"Identifying normal cells (lowest {top_percentile}% variance)...")
    
    # Rank cells by different variance metrics
    variance_cols = ['chr_variance', 'chr_std', 'chr_cv', 'chr_mad', 'chr_range', 'chr_iqr']
    
    results = {}
    for col in variance_cols:
        # Get bottom percentile (lowest variance = most normal)
        threshold = np.percentile(metrics[col], top_percentile)
        normal_indices = metrics[metrics[col] <= threshold].index.tolist()
        
        results[col] = {
            'indices': normal_indices,
            'cell_ids': [adata.obs_names[i] for i in normal_indices],
            'threshold': threshold,
            'metric_values': metrics.loc[normal_indices, col].values
        }
        
        print(f"  {col}: {len(normal_indices)} normal cells (threshold <= {threshold:.4f})")
    
    # Find consensus normal cells (appear in multiple metrics)
    all_candidates = set()
    for result in results.values():
        all_candidates.update(result['indices'])
    
    # Count how many metrics each cell appears in
    consensus_scores = {}
    for idx in all_candidates:
        score = sum(1 for result in results.values() if idx in result['indices'])
        consensus_scores[idx] = score
    
    # Get cells that appear in at least half of the metrics
    min_consensus = len(variance_cols) // 2 + 1
    consensus_normal = [idx for idx, score in consensus_scores.items() if score >= min_consensus]
    
    print(f"  Consensus normal cells (>= {min_consensus} metrics): {len(consensus_normal)} cells")
    
    return results, consensus_normal

# Load data
print("Loading data...")
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

print(f"Dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")
print(f"Cell types: {adata_original.obs['CellType'].value_counts().to_dict()}")

# Normalization steps
print("Applying normalization...")
print("  1. Total count normalization (cp10k)...")
sc.pp.normalize_total(adata_filtered, target_sum=1e4, key_added='n_counts_cp10k')

print("  2. Log transformation...")
sc.pp.log1p(adata_filtered)

# Apply smoothing to all cells
smoothed_matrix, chr_means = moving_average_smoothing_by_chromosome(adata_filtered, gene_positions, window_size=100)

# Calculate variance metrics
variance_metrics, cnv_matrix, chr_names = calculate_cnv_variance_metrics(chr_means)

# Identify normal cells using variance approach
variance_results, consensus_normal = identify_normal_cells_by_variance(variance_metrics, adata_filtered, top_percentile=10)

print("\n" + "="*60)
print("UMAP CLUSTERING ON COPY NUMBER MATRIX")  
print("="*60)

# Prepare data for UMAP
print("Preparing CNV matrix for UMAP...")
cnv_scaled = StandardScaler().fit_transform(cnv_matrix)

# Generate UMAP
print("Running UMAP...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
cnv_umap = umap_reducer.fit_transform(cnv_scaled)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Copy Number Variation Analysis - Normal Cell Identification Using Variance', fontsize=16)

# 1. UMAP colored by cell type
cell_types = adata_original.obs['CellType'].values
unique_types = cell_types.categories if hasattr(cell_types, 'categories') else np.unique(cell_types)
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
type_color_map = {cell_type: color for cell_type, color in zip(unique_types, colors)}

for i, cell_type in enumerate(unique_types):
    mask = cell_types == cell_type
    axes[0,0].scatter(cnv_umap[mask, 0], cnv_umap[mask, 1], 
                     c=[type_color_map[cell_type]], label=cell_type, alpha=0.7, s=20)
axes[0,0].set_title('UMAP: Original Cell Types')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. UMAP colored by chromosome variance
scatter = axes[0,1].scatter(cnv_umap[:, 0], cnv_umap[:, 1], 
                           c=variance_metrics['chr_variance'], cmap='viridis_r', s=20, alpha=0.7)
axes[0,1].set_title('UMAP: Chromosome Variance (lower = more normal)')
plt.colorbar(scatter, ax=axes[0,1], label='Variance')
axes[0,1].grid(True, alpha=0.3)

# 3. UMAP highlighting consensus normal cells
normal_mask = np.zeros(len(cnv_umap), dtype=bool)
normal_mask[consensus_normal] = True

axes[0,2].scatter(cnv_umap[~normal_mask, 0], cnv_umap[~normal_mask, 1], 
                 c='lightgray', label='Other cells', alpha=0.5, s=15)
axes[0,2].scatter(cnv_umap[normal_mask, 0], cnv_umap[normal_mask, 1], 
                 c='red', label=f'Normal candidates ({len(consensus_normal)})', alpha=0.8, s=30)
axes[0,2].set_title('UMAP: Identified Normal Cells')
axes[0,2].legend()
axes[0,2].grid(True, alpha=0.3)

# 4. Variance metrics distribution
axes[1,0].hist(variance_metrics['chr_variance'], bins=50, alpha=0.7, color='blue', density=True)
axes[1,0].axvline(np.percentile(variance_metrics['chr_variance'], 10), 
                 color='red', linestyle='--', label='10th percentile')
axes[1,0].set_title('Distribution of Chromosome Variance')
axes[1,0].set_xlabel('Variance')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 5. Chromosome expression heatmap for top normal candidates
top_normal_indices = sorted(consensus_normal)[:20]  # Show top 20
if len(top_normal_indices) > 0:
    normal_cnv_subset = cnv_matrix[top_normal_indices, :]
    im = axes[1,1].imshow(normal_cnv_subset.T, aspect='auto', cmap='RdBu_r', 
                         vmin=np.percentile(cnv_matrix, 5), vmax=np.percentile(cnv_matrix, 95))
    axes[1,1].set_title(f'Chr Expression: Top {len(top_normal_indices)} Normal Candidates')
    axes[1,1].set_xlabel('Cells')
    axes[1,1].set_ylabel('Chromosomes')
    axes[1,1].set_yticks(range(len(chr_names)))
    axes[1,1].set_yticklabels(chr_names)
    plt.colorbar(im, ax=axes[1,1], label='Expression')

# 6. Compare normal candidates with different cell types
cell_type_normal_overlap = {}
for cell_type in unique_types:
    type_indices = set(np.where(cell_types == cell_type)[0])
    normal_indices_set = set(consensus_normal)
    overlap = len(type_indices & normal_indices_set)
    total_type = len(type_indices)
    cell_type_normal_overlap[cell_type] = {'overlap': overlap, 'total': total_type, 
                                          'percentage': overlap/total_type*100 if total_type > 0 else 0}

overlap_data = [(ct, data['overlap'], data['total'], data['percentage']) 
                for ct, data in cell_type_normal_overlap.items()]
cell_types_plot, overlaps, totals, percentages = zip(*overlap_data)

x = np.arange(len(cell_types_plot))
width = 0.35
axes[1,2].bar(x - width/2, overlaps, width, label='Normal candidates', alpha=0.7)
axes[1,2].bar(x + width/2, [t-o for t, o in zip(totals, overlaps)], width, 
             bottom=overlaps, label='Other cells', alpha=0.7)
axes[1,2].set_title('Normal Candidates by Cell Type')
axes[1,2].set_xlabel('Cell Type')
axes[1,2].set_ylabel('Number of Cells')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(cell_types_plot, rotation=45)
axes[1,2].legend()

plt.tight_layout()
plt.savefig('cnv_analysis/normal_cell_identification_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\nTotal cells analyzed: {len(cnv_matrix)}")
print(f"Consensus normal cells identified: {len(consensus_normal)} ({len(consensus_normal)/len(cnv_matrix)*100:.1f}%)")

print(f"\nNormal candidates by cell type:")
for cell_type, data in cell_type_normal_overlap.items():
    print(f"  {cell_type}: {data['overlap']}/{data['total']} ({data['percentage']:.1f}%)")