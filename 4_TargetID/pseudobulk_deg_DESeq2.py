#!/usr/bin/env python3
"""
Pseudo-Bulk Differential Gene Expression Analysis
Pseudo-bulk analysis (paired donors with both statuses)
Compares results with single-cell LMM approach
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configuration
INPUT_FILE = 'HSC_MPP.h5ad'
MIN_CELLS_PER_DONOR = 50
MIN_CELLS_PER_STATUS = 10  # Minimum cells per status per donor (paired)
PVAL_THRESH = 0.05
FC_THRESH = 1
N_TOP_GENES_LABEL = 20
OUTPUT_DIR = 'DEG_results_pseudobulk_DESeq2'
LMM_tocompare = 'DEG_results_singlecell_LMM'

# ============================================================================
# STEP 1: Load and filter data
# ============================================================================

adata = sc.read_h5ad(INPUT_FILE)

# Filter genes expressed in at least 1% of cells
min_cells = int(0.01 * adata.n_obs)
sc.pp.filter_genes(adata, min_cells=min_cells)

# Filter donors with >= MIN_CELLS_PER_DONOR total cells
donor_counts = adata.obs['Donor'].value_counts()
donors_to_keep = donor_counts[donor_counts >= MIN_CELLS_PER_DONOR].index
adata_filtered = adata[adata.obs['Donor'].isin(donors_to_keep)].copy()

# Check status distribution per donor and identify paired donors
donor_status = adata_filtered.obs.groupby(['Donor', 'consensus_label_6votes'], observed=True).size().unstack(fill_value=0)

# Filter to donors with both statuses (>=MIN_CELLS_PER_STATUS each)
paired_donors = donor_status[
    (donor_status['HSPC'] >= MIN_CELLS_PER_STATUS) &
    (donor_status['LSPC'] >= MIN_CELLS_PER_STATUS)
].index

# Extract counts
if sparse.issparse(adata_filtered.X):
    counts = adata_filtered.X
else:
    counts = sparse.csr_matrix(adata_filtered.X)

# ============================================================================
# STEP 2: Donor-level pseudo-bulk aggregation (paired)
# ============================================================================

donor_pseudobulk_data = []
donor_metadata = []

# Iterate through paired donors only, both statuses
for donor in paired_donors:
    for status in adata_filtered.obs['consensus_label_6votes'].cat.categories:
        mask = ((adata_filtered.obs['Donor'] == donor) &
                (adata_filtered.obs['consensus_label_6votes'] == status)).values
        n_cells = mask.sum()

        pseudobulk_counts = counts[mask, :].sum(axis=0).A1
        study = adata_filtered.obs[adata_filtered.obs['Donor'] == donor]['Study'].iloc[0]

        donor_pseudobulk_data.append(pseudobulk_counts)
        donor_metadata.append({
            'Donor': donor,
            'Study': study,
            'consensus_label_6votes': status,
            'n_cells': n_cells
        })

donor_counts_df = pd.DataFrame(donor_pseudobulk_data, columns=adata_filtered.var_names)
donor_meta_df = pd.DataFrame(donor_metadata)
donor_meta_df['sample_id'] = donor_meta_df['Donor'] + '_' + donor_meta_df['consensus_label_6votes']
donor_meta_df = donor_meta_df.set_index('sample_id')
donor_counts_df.index = donor_meta_df.index

# Save
donor_counts_df.to_csv(f'{OUTPUT_DIR}/pseudobulk_donor_counts.csv')
donor_meta_df.to_csv(f'{OUTPUT_DIR}/pseudobulk_donor_metadata.csv')

# ============================================================================
# STEP 3: DESeq2 - Donor-level
# ============================================================================

# Filter lowly expressed genes
gene_filter = (donor_counts_df >= 10).sum(axis=0) >= 10
donor_counts_filtered = donor_counts_df.loc[:, gene_filter]

# Run DESeq2 with paired design
dds_donor = DeseqDataSet(
    counts=donor_counts_filtered,
    metadata=donor_meta_df,
    design_factors=['Donor', 'consensus_label_6votes'],
    refit_cooks=True
)
dds_donor.deseq2()

stat_donor = DeseqStats(dds_donor, contrast=['consensus_label_6votes', 'LSPC', 'HSPC'])
stat_donor.summary()

results_donor = stat_donor.results_df.sort_values('padj')
results_donor['gene'] = results_donor.index
results_donor.to_csv(f'{OUTPUT_DIR}/deseq2_results.csv',index=False)

# ============================================================================
# STEP 4: Create volcano plot
# ============================================================================

from plot_volcano import create_volcano_plot

# Prepare results with 'gene' column
results_donor_plot = results_donor.copy()
results_donor_plot['gene'] = results_donor_plot.index

df_donor = create_volcano_plot(
    results_df=results_donor_plot,
    output_dir=OUTPUT_DIR,
    output_prefix='pseudobulk',
    title='Pseudo-Bulk DEG with DESeq2 \n(Malignant vs Healthy in Paired Donors)',
    fc_thresh=FC_THRESH,
    pval_thresh=PVAL_THRESH,
    n_labels=N_TOP_GENES_LABEL
)

# ============================================================================
# STEP 5: Compare with single-cell LMM results
# ============================================================================

# Load LMM results
lmm_file = f'{LMM_tocompare}/singlecell_deg_significant.csv'
lmm_results = pd.read_csv(lmm_file)
lmm_results = lmm_results.set_index('gene')

# Get significant genes
donor_sig = results_donor[results_donor['padj'] < PVAL_THRESH].copy()

# Find common genes tested in both methods
common_genes = set(results_donor.index) & set(lmm_results.index)

# Merge results
donor_common = results_donor.loc[list(common_genes)]
lmm_common = lmm_results.loc[list(common_genes)]
merged = pd.DataFrame({
    'pseudobulk_log2FC': donor_common['log2FoldChange'],
    'lmm_coefficient': lmm_common['coefficient']
}).dropna()

# Remove outliers (points beyond 3 std from mean)
z_scores = np.abs((merged - merged.mean()) / merged.std())
merged_filtered = merged[(z_scores < 3).all(axis=1)]
merged = merged_filtered

# Calculate correlation
correlation = merged['pseudobulk_log2FC'].corr(merged['lmm_coefficient'])

# Find genes significant in both methods
donor_sig_genes = set(donor_sig.index) & common_genes
lmm_sig_genes = set(lmm_results.index) & common_genes
overlap_sig = donor_sig_genes & lmm_sig_genes

# Save overlap genes
if len(overlap_sig) > 0:
    overlap_df = pd.DataFrame({
        'gene': list(overlap_sig),
        'pseudobulk_log2FC': [results_donor.loc[g, 'log2FoldChange'] for g in overlap_sig],
        'pseudobulk_padj': [results_donor.loc[g, 'padj'] for g in overlap_sig],
        'lmm_coefficient': [lmm_results.loc[g, 'coefficient'] for g in overlap_sig],
        'lmm_padj': [lmm_results.loc[g, 'padj'] for g in overlap_sig]
    }).sort_values('pseudobulk_padj')
    overlap_df.to_csv(f'{OUTPUT_DIR}/overlap_genes_pseudobulk_lmm.csv', index=False)

# Create correlation plot (4:3 aspect ratio)
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.scatter(merged['pseudobulk_log2FC'], merged['lmm_coefficient'],
           c='lightgray', s=20, alpha=0.5, label='All genes')

# Highlight genes significant in both (filter to those in merged)
overlap_merged = merged.loc[merged.index.intersection(overlap_sig)]
if len(overlap_merged) > 0:
    ax.scatter(overlap_merged['pseudobulk_log2FC'], overlap_merged['lmm_coefficient'],
               c='red', s=50, alpha=0.8, edgecolors='black', linewidths=0.5,
               label=f'Significant in both (n={len(overlap_merged)})', zorder=3)

# Add fitted regression line
z = np.polyfit(merged['pseudobulk_log2FC'], merged['lmm_coefficient'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged['pseudobulk_log2FC'].min(), merged['pseudobulk_log2FC'].max(), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.7, label=f'y={z[0]:.2f}x+{z[1]:.2f}')

ax.set_xlabel('Pseudobulk log2FC (DESeq2)', fontsize=14, fontweight='bold')
ax.set_ylabel('Single-cell LMM Coefficient', fontsize=14, fontweight='bold')
ax.set_title(f'Method Comparison\n(r = {correlation:.3f})', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)

# Remove grid and box frame
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pseudobulk_vs_lmm_correlation.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/pseudobulk_vs_lmm_correlation.pdf', bbox_inches='tight')
plt.close()

n_sig = (results_donor['padj'] < PVAL_THRESH).sum()
print(f'\nPseudobulk DESeq2 analysis complete: {n_sig} significant genes (FDR<{PVAL_THRESH})')
print(f'Correlation with LMM: r={correlation:.3f}, {len(overlap_sig)} genes significant in both methods')