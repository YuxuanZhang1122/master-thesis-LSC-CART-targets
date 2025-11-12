import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

def create_smoothed_cnv_matrix(adata, gene_positions, normal_indices, chromosomes=['6', '7', '8'], window_size=50):
    """
    Create smoothed copy number matrix using normal reference
    """
    print("Creating smoothed CNV matrix...")
    
    all_cnv_data = []
    feature_names = []
    
    for chr_name in chromosomes:
        print(f"  Processing Chr{chr_name}...")
        
        # Get genes for this chromosome
        chr_genes = gene_positions[gene_positions['chromosome'] == chr_name]['gene_symbol'].tolist()
        chr_genes_in_data = [g for g in chr_genes if g in adata.var_names]
        
        if len(chr_genes_in_data) < 10:
            continue
            
        # Get gene positions and sort
        chr_positions = gene_positions[gene_positions['gene_symbol'].isin(chr_genes_in_data)].copy()
        chr_positions = chr_positions.sort_values('start').reset_index(drop=True)
        
        gene_indices = [list(adata.var_names).index(gene) for gene in chr_positions['gene_symbol']]
        
        # Extract expression data for all cells
        if hasattr(adata.X, 'toarray'):
            chr_expr_all = adata.X[:, gene_indices].toarray()
        else:
            chr_expr_all = adata.X[:, gene_indices]
        
        # Create reference from normal cells
        chr_expr_normal = chr_expr_all[normal_indices, :]
        normal_reference = chr_expr_normal.mean(axis=0)
        normal_reference_smoothed = uniform_filter1d(normal_reference, size=min(window_size, len(normal_reference)), mode='nearest')
        
        # Apply smoothing to each cell and calculate log2 ratios
        n_cells = chr_expr_all.shape[0]
        chr_smoothed_all = np.zeros_like(chr_expr_all)
        
        for i in range(n_cells):
            chr_smoothed_all[i, :] = uniform_filter1d(chr_expr_all[i, :], 
                                                    size=min(window_size, chr_expr_all.shape[1]), 
                                                    mode='nearest')
        
        # Calculate log2 ratios relative to normal reference
        log2_ratios = np.log2((chr_smoothed_all + 1e-8) / (normal_reference_smoothed + 1e-8))
        
        # Bin the chromosome into segments for dimensionality reduction
        n_bins = 20  # Reduce to manageable features
        bin_size = max(1, log2_ratios.shape[1] // n_bins)
        
        chr_features = []
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = min((bin_idx + 1) * bin_size, log2_ratios.shape[1])
            
            if start_idx < log2_ratios.shape[1]:
                # Mean log2 ratio in this bin for all cells
                bin_mean = log2_ratios[:, start_idx:end_idx].mean(axis=1)
                chr_features.append(bin_mean)
                feature_names.append(f'Chr{chr_name}_bin{bin_idx+1}')
        
        if chr_features:
            all_cnv_data.extend(chr_features)
            print(f"    Chr{chr_name}: {len(chr_features)} features created from {len(chr_genes_in_data)} genes")
    
    # Convert to matrix (cells x features)
    cnv_matrix = np.column_stack(all_cnv_data)
    print(f"Final CNV matrix: {cnv_matrix.shape[0]} cells x {cnv_matrix.shape[1]} features")
    
    return cnv_matrix, feature_names

# Load data
print("Loading data for PCA + UMAP analysis...")
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

# Define cell groups
cell_types = adata_original.obs['CellType']
normal_mask = cell_types.isin(['HSC', 'Prog'])
normal_indices = np.where(normal_mask)[0]
malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])

print(f"Dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")
print(f"Cell type distribution: {cell_types.value_counts().to_dict()}")

print("\n" + "="*60)
print("CREATING SMOOTHED CNV MATRIX")
print("="*60)

# Create smoothed CNV matrix focused on Chr6, 7, 8
cnv_matrix, feature_names = create_smoothed_cnv_matrix(adata_filtered, gene_positions, normal_indices, 
                                                     chromosomes=['6', '7', '8'])

# Scale the CNV matrix
print("\nScaling CNV matrix...")
scaler = StandardScaler()
cnv_scaled = scaler.fit_transform(cnv_matrix)

print("\n" + "="*60)
print("APPLYING PCA")
print("="*60)

# Apply PCA
n_components = min(50, cnv_scaled.shape[1], cnv_scaled.shape[0] - 1)  # Don't exceed matrix dimensions
print(f"Applying PCA with {n_components} components...")

pca = PCA(n_components=n_components)
cnv_pca = pca.fit_transform(cnv_scaled)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:10]}")
print(f"Cumulative variance explained by first 10 PCs: {np.cumsum(pca.explained_variance_ratio_[:10])[-1]:.3f}")

print("\n" + "="*60)
print("APPLYING UMAP TO PCA COMPONENTS")
print("="*60)

# Apply UMAP to PCA components
print("Running UMAP on PCA components...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
cnv_umap = umap_reducer.fit_transform(cnv_pca)

print("UMAP completed")

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('CNV Analysis: PCA + UMAP on Smoothed Copy Number Matrix (Chr6-8)', fontsize=16)

# 1. PCA explained variance
axes[0,0].plot(range(1, 11), pca.explained_variance_ratio_[:10], 'bo-', markersize=6)
axes[0,0].set_title('PCA Explained Variance', fontsize=12)
axes[0,0].set_xlabel('Principal Component')
axes[0,0].set_ylabel('Explained Variance Ratio')
axes[0,0].grid(True, alpha=0.3)

# 2. UMAP colored by original cell types
unique_types = cell_types.unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_types)))
type_color_map = {cell_type: color for cell_type, color in zip(unique_types, colors)}

for cell_type in unique_types:
    mask = cell_types == cell_type
    axes[0,1].scatter(cnv_umap[mask, 0], cnv_umap[mask, 1], 
                     c=[type_color_map[cell_type]], label=cell_type, alpha=0.7, s=20)
axes[0,1].set_title('UMAP: Original Cell Types')
axes[0,1].legend()
axes[0,1].set_xlabel('UMAP 1')
axes[0,1].set_ylabel('UMAP 2')
axes[0,1].grid(True, alpha=0.3)

# 3. UMAP colored by Normal vs Malignant
normal_malignant = ['Normal' if t in ['HSC', 'Prog'] else 'Malignant' for t in cell_types]
for group in ['Normal', 'Malignant']:
    mask = np.array(normal_malignant) == group
    color = 'blue' if group == 'Normal' else 'red'
    axes[0,2].scatter(cnv_umap[mask, 0], cnv_umap[mask, 1], 
                     c=color, label=f'{group} (n={np.sum(mask)})', alpha=0.7, s=20)
axes[0,2].set_title('UMAP: Normal vs Malignant')
axes[0,2].legend()
axes[0,2].set_xlabel('UMAP 1')
axes[0,2].set_ylabel('UMAP 2')
axes[0,2].grid(True, alpha=0.3)

# 4. UMAP colored by first PC
scatter = axes[1,0].scatter(cnv_umap[:, 0], cnv_umap[:, 1], 
                           c=cnv_pca[:, 0], cmap='viridis', s=20, alpha=0.7)
axes[1,0].set_title('UMAP: Colored by PC1')
axes[1,0].set_xlabel('UMAP 1')
axes[1,0].set_ylabel('UMAP 2')
plt.colorbar(scatter, ax=axes[1,0], label='PC1 Score')
axes[1,0].grid(True, alpha=0.3)

# 5. UMAP colored by CNV burden (mean absolute log2 ratio)
cnv_burden = np.mean(np.abs(cnv_matrix), axis=1)
scatter2 = axes[1,1].scatter(cnv_umap[:, 0], cnv_umap[:, 1], 
                            c=cnv_burden, cmap='plasma', s=20, alpha=0.7)
axes[1,1].set_title('UMAP: CNV Burden')
axes[1,1].set_xlabel('UMAP 1')
axes[1,1].set_ylabel('UMAP 2')
plt.colorbar(scatter2, ax=axes[1,1], label='CNV Burden')
axes[1,1].grid(True, alpha=0.3)

# 6. PCA biplot (first 2 PCs)
axes[1,2].scatter(cnv_pca[:, 0], cnv_pca[:, 1], 
                 c=[type_color_map[t] for t in cell_types], alpha=0.6, s=20)
axes[1,2].set_title('PCA: First 2 Components')
axes[1,2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
axes[1,2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnv_analysis/pca_umap_cnv_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and print statistics
print("\n" + "="*60)
print("ANALYSIS STATISTICS")
print("="*60)

print(f"CNV matrix dimensions: {cnv_matrix.shape}")
print(f"PCA components used: {n_components}")
print(f"Variance explained by first 5 PCs: {pca.explained_variance_ratio_[:5]}")
print(f"Cumulative variance (first 10 PCs): {np.cumsum(pca.explained_variance_ratio_[:10])[-1]:.3f}")

# Analyze separation by cell type
from sklearn.metrics import silhouette_score
try:
    # Silhouette score for original cell types
    cell_type_labels = [list(unique_types).index(ct) for ct in cell_types]
    sil_original = silhouette_score(cnv_umap, cell_type_labels)
    
    # Silhouette score for normal vs malignant
    normal_malignant_labels = [0 if nm == 'Normal' else 1 for nm in normal_malignant]
    sil_normal_malignant = silhouette_score(cnv_umap, normal_malignant_labels)
    
    print(f"\nUMAP separation quality:")
    print(f"  Original cell types silhouette: {sil_original:.3f}")
    print(f"  Normal vs Malignant silhouette: {sil_normal_malignant:.3f}")
except:
    print("Could not calculate silhouette scores")

# CNV burden statistics
normal_burden = cnv_burden[normal_mask]
malignant_burden = cnv_burden[malignant_mask]

print(f"\nCNV burden statistics:")
print(f"  Normal cells: {np.mean(normal_burden):.3f} ± {np.std(normal_burden):.3f}")
print(f"  Malignant cells: {np.mean(malignant_burden):.3f} ± {np.std(malignant_burden):.3f}")