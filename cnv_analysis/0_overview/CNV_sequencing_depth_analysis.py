import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data for sequencing depth analysis...")
adata_original = sc.read_h5ad('/Users/axuan/Desktop/Thesis/AML328_purified.h5ad')

# Define cell groups
cell_types = adata_original.obs['CellType']
normal_mask = cell_types.isin(['HSC', 'Prog'])
malignant_mask = cell_types.isin(['HSC-like', 'Prog-like'])

normal_indices = np.where(normal_mask)[0]
malignant_indices = np.where(malignant_mask)[0]

print(f"Normal cells: {len(normal_indices)} (HSC + Prog)")
print(f"Malignant cells: {len(malignant_indices)} (HSC-like + Prog-like)")

# Calculate sequencing depth metrics
print("\nCalculating sequencing depth metrics...")

# Total UMI counts per cell (library size)
if hasattr(adata_original.X, 'toarray'):
    total_counts_per_cell = np.array(adata_original.X.sum(axis=1)).flatten()
else:
    total_counts_per_cell = adata_original.X.sum(axis=1)

# Number of genes detected per cell (genes with >0 counts)
if hasattr(adata_original.X, 'toarray'):
    genes_per_cell = np.array((adata_original.X > 0).sum(axis=1)).flatten()
else:
    genes_per_cell = (adata_original.X > 0).sum(axis=1)

# Extract metrics for each group
normal_total_counts = total_counts_per_cell[normal_indices]
malignant_total_counts = total_counts_per_cell[malignant_indices]

normal_genes_detected = genes_per_cell[normal_indices]
malignant_genes_detected = genes_per_cell[malignant_indices]

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Sequencing Depth Comparison: Normal vs Malignant Cells', fontsize=16, y=0.99)

# 1. Total UMI counts distribution
ax1 = axes[0, 0]
ax1.hist(normal_total_counts, bins=50, alpha=0.7, color='blue', label=f'Normal (n={len(normal_indices)})', density=True)
ax1.hist(malignant_total_counts, bins=50, alpha=0.7, color='red', label=f'Malignant (n={len(malignant_indices)})', density=True)
ax1.set_xlabel('Total UMI Counts per Cell')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Total UMI Counts')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Genes detected distribution
ax2 = axes[0, 1]
ax2.hist(normal_genes_detected, bins=50, alpha=0.7, color='blue', label=f'Normal (n={len(normal_indices)})', density=True)
ax2.hist(malignant_genes_detected, bins=50, alpha=0.7, color='red', label=f'Malignant (n={len(malignant_indices)})', density=True)
ax2.set_xlabel('Genes Detected per Cell')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of Genes Detected')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Box plots for total UMI counts
ax3 = axes[0, 2]
box_data_counts = [normal_total_counts, malignant_total_counts]
box_plot1 = ax3.boxplot(box_data_counts, labels=['Normal', 'Malignant'], patch_artist=True)
box_plot1['boxes'][0].set_facecolor('blue')
box_plot1['boxes'][1].set_facecolor('red')
ax3.set_ylabel('Total UMI Counts per Cell')
ax3.set_title('Total UMI Counts Comparison')
ax3.grid(True, alpha=0.3)

# 4. Box plots for genes detected
ax4 = axes[1, 0]
box_data_genes = [normal_genes_detected, malignant_genes_detected]
box_plot2 = ax4.boxplot(box_data_genes, labels=['Normal', 'Malignant'], patch_artist=True)
box_plot2['boxes'][0].set_facecolor('blue')
box_plot2['boxes'][1].set_facecolor('red')
ax4.set_ylabel('Genes Detected per Cell')
ax4.set_title('Genes Detected Comparison')
ax4.grid(True, alpha=0.3)

# 5. Scatter plot: Total counts vs Genes detected
ax5 = axes[1, 1]
ax5.scatter(normal_total_counts, normal_genes_detected, alpha=0.6, c='blue', s=20, label='Normal')
ax5.scatter(malignant_total_counts, malignant_genes_detected, alpha=0.6, c='red', s=20, label='Malignant')
ax5.set_xlabel('Total UMI Counts per Cell')
ax5.set_ylabel('Genes Detected per Cell')
ax5.set_title('UMI Counts vs Genes Detected')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Violin plots
ax6 = axes[1, 2]
violin_parts = ax6.violinplot([normal_total_counts, malignant_total_counts], positions=[1, 2], showmeans=True)
violin_parts['bodies'][0].set_facecolor('blue')
violin_parts['bodies'][1].set_facecolor('red')
ax6.set_xticks([1, 2])
ax6.set_xticklabels(['Normal', 'Malignant'])
ax6.set_ylabel('Total UMI Counts per Cell')
ax6.set_title('UMI Count Distribution (Violin)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sequencing_depth_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical analysis
print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)

# Summary statistics
normal_counts_stats = {
    'mean': np.mean(normal_total_counts),
    'median': np.median(normal_total_counts),
    'std': np.std(normal_total_counts),
    'min': np.min(normal_total_counts),
    'max': np.max(normal_total_counts)
}

malignant_counts_stats = {
    'mean': np.mean(malignant_total_counts),
    'median': np.median(malignant_total_counts),
    'std': np.std(malignant_total_counts),
    'min': np.min(malignant_total_counts),
    'max': np.max(malignant_total_counts)
}

normal_genes_stats = {
    'mean': np.mean(normal_genes_detected),
    'median': np.median(normal_genes_detected),
    'std': np.std(normal_genes_detected),
    'min': np.min(normal_genes_detected),
    'max': np.max(normal_genes_detected)
}

malignant_genes_stats = {
    'mean': np.mean(malignant_genes_detected),
    'median': np.median(malignant_genes_detected),
    'std': np.std(malignant_genes_detected),
    'min': np.min(malignant_genes_detected),
    'max': np.max(malignant_genes_detected)
}

print("TOTAL UMI COUNTS PER CELL:")
print(f"Normal cells   - Mean: {normal_counts_stats['mean']:.0f}, Median: {normal_counts_stats['median']:.0f}, Std: {normal_counts_stats['std']:.0f}")
print(f"Malignant cells - Mean: {malignant_counts_stats['mean']:.0f}, Median: {malignant_counts_stats['median']:.0f}, Std: {malignant_counts_stats['std']:.0f}")
print(f"Fold difference (Malignant/Normal): {malignant_counts_stats['mean']/normal_counts_stats['mean']:.2f}")

print("\nGENES DETECTED PER CELL:")
print(f"Normal cells   - Mean: {normal_genes_stats['mean']:.0f}, Median: {normal_genes_stats['median']:.0f}, Std: {normal_genes_stats['std']:.0f}")
print(f"Malignant cells - Mean: {malignant_genes_stats['mean']:.0f}, Median: {malignant_genes_stats['median']:.0f}, Std: {malignant_genes_stats['std']:.0f}")
print(f"Fold difference (Malignant/Normal): {malignant_genes_stats['mean']/normal_genes_stats['mean']:.2f}")

# Statistical tests
print("\nSTATISTICAL TESTS:")

# Mann-Whitney U test for total counts (non-parametric)
stat_counts, p_counts = stats.mannwhitneyu(normal_total_counts, malignant_total_counts, alternative='two-sided')
print(f"Total UMI Counts - Mann-Whitney U test:")
print(f"  Statistic: {stat_counts:.2f}, p-value: {p_counts:.2e}")
print(f"  Significant difference: {'Yes' if p_counts < 0.05 else 'No'}")

# Mann-Whitney U test for genes detected
stat_genes, p_genes = stats.mannwhitneyu(normal_genes_detected, malignant_genes_detected, alternative='two-sided')
print(f"Genes Detected - Mann-Whitney U test:")
print(f"  Statistic: {stat_genes:.2f}, p-value: {p_genes:.2e}")
print(f"  Significant difference: {'Yes' if p_genes < 0.05 else 'No'}")

# Effect size (Cohen's d)
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

d_counts = cohens_d(malignant_total_counts, normal_total_counts)
d_genes = cohens_d(malignant_genes_detected, normal_genes_detected)

print(f"\nEFFECT SIZES (Cohen's d):")
print(f"Total UMI Counts: {d_counts:.3f} ({'Large' if abs(d_counts) > 0.8 else 'Medium' if abs(d_counts) > 0.5 else 'Small'})")
print(f"Genes Detected: {d_genes:.3f} ({'Large' if abs(d_genes) > 0.8 else 'Medium' if abs(d_genes) > 0.5 else 'Small'})")

# Create summary dataframe
summary_data = {
    'Cell_Group': ['Normal', 'Malignant'],
    'N_Cells': [len(normal_indices), len(malignant_indices)],
    'Mean_UMI_Counts': [normal_counts_stats['mean'], malignant_counts_stats['mean']],
    'Median_UMI_Counts': [normal_counts_stats['median'], malignant_counts_stats['median']],
    'Std_UMI_Counts': [normal_counts_stats['std'], malignant_counts_stats['std']],
    'Mean_Genes_Detected': [normal_genes_stats['mean'], malignant_genes_stats['mean']],
    'Median_Genes_Detected': [normal_genes_stats['median'], malignant_genes_stats['median']],
    'Std_Genes_Detected': [normal_genes_stats['std'], malignant_genes_stats['std']]
}

summary_df = pd.DataFrame(summary_data)
print(f"\nSUMMARY TABLE:")
print(summary_df.to_string(index=False, float_format='%.1f'))

# Save results
summary_df.to_csv('sequencing_depth_summary.csv', index=False)

print("\n" + "="*60)
print("SEQUENCING DEPTH ANALYSIS COMPLETE!")
print("="*60)
print("Files created:")
print("  1. sequencing_depth_comparison.png - Comprehensive visualization")
print("  2. sequencing_depth_summary.csv - Summary statistics")

# Interpretation
print(f"\n✅ Analysis based on {len(normal_indices)} normal vs {len(malignant_indices)} malignant cells")
print(f"✅ Statistical tests performed (Mann-Whitney U)")
print(f"✅ Effect sizes calculated (Cohen's d)")

if p_counts < 0.05:
    direction = "higher" if malignant_counts_stats['mean'] > normal_counts_stats['mean'] else "lower"
    print(f"✅ Malignant cells have significantly {direction} UMI counts (p={p_counts:.2e})")
else:
    print(f"✅ No significant difference in UMI counts between groups (p={p_counts:.2e})")

if p_genes < 0.05:
    direction = "more" if malignant_genes_stats['mean'] > normal_genes_stats['mean'] else "fewer"
    print(f"✅ Malignant cells detect significantly {direction} genes (p={p_genes:.2e})")
else:
    print(f"✅ No significant difference in genes detected between groups (p={p_genes:.2e})")