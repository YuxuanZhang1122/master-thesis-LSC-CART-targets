#!/usr/bin/env python3
"""
Pseudo-Bulk Differential Gene Expression Analysis with DESeq2
Pseudo-bulk analysis (paired donors with both statuses)
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from adjustText import adjust_text
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Configuration
INPUT_FILE = 'HSC_MPP_full.h5ad'
MIN_CELLS_PER_DONOR = 50
MIN_CELLS_PER_STATUS = 10  # Minimum cells per status per donor (paired)

# Gene filtering criteria
MIN_EXPR_PCT = 0.01         # Minimum percentage of LSPC cells expressing
MIN_MEAN_EXPR = 0.1        # Minimum mean expression level of LSPC cells
LSPC_higherthan_HSPC = False # Compare mean cpm

# Toggle for excluding mitochondrial, cell cycle, and housekeeping genes
EXCLUDE_MT_CC_HK_GENES = True

PVAL_THRESH = 0.05
FC_THRESH = 1
N_TOP_GENES_LABEL = 10
OUTPUT_DIR = 'Outputs/DEG_results_pseudobulk_DESeq2'

def create_volcano_plot(results_df, output_dir, output_prefix, title,
                        fc_thresh=1, pval_thresh=0.05, n_labels=10,
                        genes_of_interest=['CD33','Clec12A','IL3RA']):
    """
    Create volcano plot from DEG results.

    Generates volcano plots with significance thresholds, gene labels, and optional highlighting of genes of interest.
    Saves both PNG and PDF formats.

    Parameters
    ----------
    results_df : pd.DataFrame
        DEG results containing columns: 'gene', 'log2FoldChange', 'padj'
    output_dir : str
        Output directory path for saving figures
    output_prefix : str
        Prefix for output filenames (e.g., 'singlecell', 'pseudobulk')
    title : str
        Plot title text
    fc_thresh : float, default=1
        Log2 fold change threshold for significance
    pval_thresh : float, default=0.05
        Adjusted p-value threshold for significance
    n_labels : int, default=10
        Number of top significant genes to label per direction (up/down)
    genes_of_interest : list of str, optional
        Gene names to highlight with red markers and priority labeling
    """

    df = results_df.copy()

    # Calculate -log10(padj)
    df['-log10_padj'] = -np.log10(df['padj'])
    max_log_p = df['-log10_padj'][np.isfinite(df['-log10_padj'])].max()
    df['-log10_padj'] = df['-log10_padj'].replace([np.inf, -np.inf], max_log_p + 1)
    #df['-log10_padj'] = df['-log10_padj'].clip(upper=80)

    # Classify significance
    df['significant'] = 'NS'
    df.loc[(df['padj'] < pval_thresh) & (df['log2FoldChange'] > fc_thresh), 'significant'] = 'UP'
    df.loc[(df['padj'] < pval_thresh) & (df['log2FoldChange'] < -fc_thresh), 'significant'] = 'DOWN'

    n_up = (df['significant'] == 'UP').sum()
    n_down = (df['significant'] == 'DOWN').sum()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    colors = {'NS': '#CCCCCC', 'UP': '#E67F33', 'DOWN': '#6B4C9A'}

    # Plot points by category
    for cat in ['NS', 'DOWN', 'UP']:
        subset = df[df['significant'] == cat]
        ax.scatter(subset['log2FoldChange'], subset['-log10_padj'],
                   c=colors[cat], s=20 if cat == 'NS' else 40,
                   alpha=0.5 if cat == 'NS' else 0.8,
                   edgecolors='black' if cat != 'NS' else 'none',
                   linewidths=1 if cat != 'NS' else 0,
                   zorder=1 if cat == 'NS' else 4)

    # Add threshold lines
    ax.axhline(-np.log10(pval_thresh), color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(fc_thresh, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(-fc_thresh, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    # Label genes
    texts = []

    # Highlight genes of interest
    if genes_of_interest:
        genes_of_interest_in_data = df[df['gene'].isin(genes_of_interest)]

        for _, row in genes_of_interest_in_data.iterrows():
            ax.scatter(row['log2FoldChange'], row['-log10_padj'],
                       c='red', s=70, alpha=0.9, edgecolors='black',
                       linewidths=1, zorder=5, marker='o')

            texts.append(ax.text(row['log2FoldChange'], row['-log10_padj'], ' ' + row['gene'],
                                 fontsize=9, fontweight='bold', color='black', zorder=6,
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                           edgecolor='grey', linewidth=1, alpha=0.8),
                                 ha='right', va='bottom'))

    # Label top N significant genes per direction
    genes_to_exclude = genes_of_interest if genes_of_interest else []

    sig_up = df[(df['significant'] == 'UP')].sort_values('padj').head(n_labels)
    sig_up = sig_up[~sig_up['gene'].isin(genes_to_exclude)]

    sig_down = df[(df['significant'] == 'DOWN')].sort_values('padj').head(n_labels)
    sig_down = sig_down[~sig_down['gene'].isin(genes_to_exclude)]

    sig_for_label = pd.concat([sig_up, sig_down])

    for _, row in sig_for_label.iterrows():
        texts.append(ax.text(row['log2FoldChange'], row['-log10_padj'], row['gene'],
                             fontsize=9, fontweight='bold', zorder=3))

    # Adjust text positions to avoid overlap
    if len(texts) > 0:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5, alpha=0.5), ax=ax)

    # Add gene count labels
    ax.text(0.02, 1.00, f'{n_down} genes', transform=ax.transAxes, fontsize=12,
            fontweight='bold', color=colors['DOWN'], verticalalignment='top')
    ax.text(0.98, 1.00, f'{n_up} genes', transform=ax.transAxes, fontsize=12,
            fontweight='bold', color=colors['UP'], verticalalignment='top', horizontalalignment='right')

    # Labels and styling
    ax.set_xlabel('Malignant vs Healthy (log2FC)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Significance (-log10 adjusted p value)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.text(0.98, 0.02, f'α={pval_thresh}', transform=ax.transAxes, fontsize=11,
            color='gray', verticalalignment='bottom', horizontalalignment='right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Save plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{output_prefix}_volcano.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig(f'{output_dir}/{output_prefix}_volcano.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# ============================================================================
# STEP 1: Load and filter data
# ============================================================================

adata = sc.read_h5ad(INPUT_FILE)

# ============================================================================
# Gene expression filtering
# ============================================================================

# Get raw counts for CPM calculation
if sparse.issparse(adata.X):
    raw_counts = adata.X.toarray()
else:
    raw_counts = adata.X.copy()

library_sizes = raw_counts.sum(axis=1, keepdims=True)
cpm = (raw_counts / library_sizes) * 1e4

# Identify LSPC and HSPC cells
malignant_mask = adata.obs['consensus_label_6votes'] == 'LSPC'
healthy_mask = adata.obs['consensus_label_6votes'] == 'HSPC'
n_malignant_cells = malignant_mask.sum()

# Calculate expression metrics in LSPC cells
expr_malignant = cpm[malignant_mask, :]
n_cells_expr_malignant = (expr_malignant > 0).sum(axis=0)
expr_pct_malignant = n_cells_expr_malignant / n_malignant_cells
mean_expr_malignant = expr_malignant.mean(axis=0)

# Calculate expression metrics in HSPC cells
expr_healthy = cpm[healthy_mask, :]
mean_expr_healthy = expr_healthy.mean(axis=0)

# Filter: genes must meet ALL criteria
if LSPC_higherthan_HSPC:
    gene_filter = (expr_pct_malignant >= MIN_EXPR_PCT) & \
                  (mean_expr_malignant >= MIN_MEAN_EXPR) & \
                  (mean_expr_malignant > mean_expr_healthy)
else:
    gene_filter = (expr_pct_malignant >= MIN_EXPR_PCT) & \
                  (mean_expr_malignant >= MIN_MEAN_EXPR)

# ============================================================================
# Optionally remove ribosomal, mitochondrial, cell cycle, housekeeping genes
# ============================================================================

if EXCLUDE_MT_CC_HK_GENES:
    genes_to_exclude = []

    # Ribosomal protein genes
    ribosomal_patterns = ['RPL', 'RPS', 'MRPL', 'MRPS']
    for gene in adata.var_names:
        if any(gene.startswith(pattern) for pattern in ribosomal_patterns):
            genes_to_exclude.append(gene)

    # Mitochondrial genes
    if any(gene.startswith('MT-') for gene in adata.var_names):
        mt_genes = [g for g in adata.var_names if g.startswith('MT-')]
        genes_to_exclude.extend(mt_genes)

    # Heat shock proteins
    hsp_genes = [g for g in adata.var_names if g.startswith('HSP')]
    genes_to_exclude.extend(hsp_genes)

    # Cell cycle genes
    cell_cycle_genes = [
        'MKI67', 'TOP2A', 'CCNA2', 'CCNB1', 'CCNB2', 'CCND1', 'CCND2', 'CCND3',
        'CCNE1', 'CCNE2', 'CDC20', 'CDC6', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN3',
        'CDK1', 'CDK2', 'CDK4', 'CDK6', 'PCNA', 'AURKA', 'AURKB', 'BUB1', 'BUB1B',
        'MAD2L1', 'PLK1', 'PTTG1', 'ZWINT', 'CENPE', 'CENPF', 'UBE2C', 'BIRC5',
        'HMGB2', 'TUBB', 'TUBA1B', 'SMC2', 'SMC4', 'H2AFZ', 'HIST1H1B', 'HIST1H4C'
    ]
    # Housekeeping genes
    housekeeping_genes = [
        'ACTB', 'GAPDH', 'HPRT1', 'B2M', 'TBP', 'UBC', 'PPIA', 'PGK1'
    ]

    for gene in cell_cycle_genes:
        if gene in adata.var_names:
            genes_to_exclude.append(gene)
    for gene in housekeeping_genes:
        if gene in adata.var_names:
            genes_to_exclude.append(gene)

    # Immunoglobulin genes
    ig_genes = [g for g in adata.var_names if g.startswith('IG')]
    genes_to_exclude.extend(ig_genes)

    genes_to_exclude = list(set(genes_to_exclude))

    # Create final gene filter combining expression filter and exclusion list
    exclude_mask = np.array([g not in genes_to_exclude for g in adata.var_names])
    final_gene_filter = gene_filter & exclude_mask
else:
    final_gene_filter = gene_filter

adata = adata[:,final_gene_filter].copy()


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
    design_factors=['Study', 'consensus_label_6votes'],
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

create_volcano_plot(
    results_df=results_donor,
    output_dir=OUTPUT_DIR,
    output_prefix='pseudobulk',
    title='Pseudo-Bulk DEG with DESeq2 \n(Malignant vs Healthy in Paired Donors)',
    fc_thresh=FC_THRESH,
    pval_thresh=PVAL_THRESH,
    n_labels=N_TOP_GENES_LABEL
)

n_sig = (results_donor['padj'] < PVAL_THRESH).sum()
print(f'\nPseudobulk DESeq2 analysis complete: {n_sig} significant genes (FDR<{PVAL_THRESH})')