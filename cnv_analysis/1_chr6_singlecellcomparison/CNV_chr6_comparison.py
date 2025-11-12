import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the original and filtered datasets
print("Loading datasets...")
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')
adata_filtered = sc.read_h5ad('cnv_analysis/filtered_adata.h5ad')
gene_positions = pd.read_csv('cnv_analysis/gene_positions.csv')

print(f"Original dataset: {adata_original.shape}")
print(f"Filtered dataset: {adata_filtered.shape}")

cell_types = adata_original.obs["CellType"]

# Find normal and malignant cells
# In AML context: HSC, Progenitor = normal/healthy; HSC-like, Prog-like = malignant/leukemic
normal_cells = []
malignant_cells = []

for idx, cell_type in enumerate(cell_types):
    cell_type_str = str(cell_type).lower()
    if ('HSC' in cell_type_str and 'like' not in cell_type_str) or ('prog' in cell_type_str and 'like' not in cell_type_str):  # Normal cells
        normal_cells.append(idx)
    elif 'hsc-like' in cell_type_str or 'prog-like' in cell_type_str:  # Malignant cells
        malignant_cells.append(idx)

print(f"Found {len(normal_cells)} normal cells")
print(f"Found {len(malignant_cells)} malignant cells")

# Select one representative cell from each type
normal_cell_idx = normal_cells[0]
malignant_cell_idx = malignant_cells[0]

normal_cell_id = adata_original.obs_names[normal_cell_idx]
malignant_cell_id = adata_original.obs_names[malignant_cell_idx]

print(f"\nSelected cells:")
print(f"  Normal cell: {normal_cell_id} (type: {cell_types.iloc[normal_cell_idx]})")
print(f"  Malignant cell: {malignant_cell_id} (type: {cell_types.iloc[malignant_cell_idx]})")

# Get Chr6 genes
chr6_genes = gene_positions[gene_positions['chromosome'] == '6']['gene_symbol'].tolist()
chr6_genes_in_data = [g for g in chr6_genes if g in adata_filtered.var_names]

print(f"\nChromosome 6 analysis:")
print(f"  Total Chr6 genes with positions: {len(chr6_genes)}")
print(f"  Chr6 genes in filtered data: {len(chr6_genes_in_data)}")

# Extract Chr6 expression for both cells
normal_chr6_expr = adata_filtered[adata_filtered.obs_names == normal_cell_id, chr6_genes_in_data].X.toarray().flatten()
malignant_chr6_expr = adata_filtered[adata_filtered.obs_names == malignant_cell_id, chr6_genes_in_data].X.toarray().flatten()

# Get gene positions for Chr6 for plotting
chr6_positions = gene_positions[gene_positions['gene_symbol'].isin(chr6_genes_in_data)].copy()
chr6_positions = chr6_positions.sort_values('start').reset_index(drop=True)

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f'Chromosome 6 Expression Comparison\nNormal vs Malignant Cell', fontsize=16)

# 1. Side-by-side expression along Chr6
positions = chr6_positions['start'].values / 1e6  # Convert to Mb

axes[0].plot(positions, normal_chr6_expr, 'o-', alpha=0.7, color='blue', markersize=3, linewidth=1, label='Normal')
axes[0].plot(positions, malignant_chr6_expr, 'o-', alpha=0.7, color='red', markersize=3, linewidth=1, label='Malignant')
axes[0].set_title('Chr6 Expression Along Chromosome')
axes[0].set_xlabel('Position (Mb)')
axes[0].set_ylabel('Expression Level')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Expression difference (Malignant - Normal)
expr_diff = malignant_chr6_expr - normal_chr6_expr
colors = ['red' if x > 0 else 'blue' for x in expr_diff]
axes[1].scatter(positions, expr_diff, c=colors, alpha=0.6, s=20)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].set_title('Expression Difference (Malignant - Normal) [Red > 0, Blue < 0]')
axes[1].set_xlabel('Position (Mb)')
axes[1].set_ylabel('Expression Difference')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnv_analysis/chr6_normal_vs_malignant_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate summary statistics
print(f"\n" + "="*50)
print("Chr6 EXPRESSION STATISTICS")
print("="*50)
print(f"Normal cell Chr6 expression:")
print(f"  Mean: {normal_chr6_expr.mean():.4f}")
print(f"  Median: {np.median(normal_chr6_expr):.4f}")
print(f"  Std: {normal_chr6_expr.std():.4f}")
print(f"  Non-zero genes: {np.sum(normal_chr6_expr > 0)}/{len(normal_chr6_expr)}")

print(f"\nMalignant cell Chr6 expression:")
print(f"  Mean: {malignant_chr6_expr.mean():.4f}")
print(f"  Median: {np.median(malignant_chr6_expr):.4f}")
print(f"  Std: {malignant_chr6_expr.std():.4f}")
print(f"  Non-zero genes: {np.sum(malignant_chr6_expr > 0)}/{len(malignant_chr6_expr)}")