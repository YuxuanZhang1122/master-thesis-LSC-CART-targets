#!/usr/bin/env python3
"""
Single-Cell Differential Expression Analysis with Linear Mixed Models (LMM)
Properly accounts for pseudoreplication by modeling Donor as a random effect
Model: expression ~ Status + Study + (1|Donor)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats, sparse
from statsmodels.regression.mixed_linear_model import MixedLM
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'HSC_MPP_surface.h5ad'
MIN_CELLS_PER_DONOR = 50

# Combined filtering criteria
MIN_EXPR_PCT = 0.05         # Minimum percentage of MALIGNANT cells expressing
MIN_MEAN_EXPR = 0.1        # Minimum mean expression level of MALIGNANT cells
LSPC_higherthan_HSPC = True # Compare mean cpm

FDR_THRESH = 0.05
FC_THRESH = 1
N_TOP_GENES_LABEL = 20
GENES_OF_INTEREST = ['CD33', 'IL3RA', 'CLEC12A', 'CSF1R', 'CD86'] # force include in analysis
OUTPUT_DIR = 'DEG_results_singlecell_LMM_surfaceproteinonly'

# ============================================================================
# STEP 1: Load and filter data
# ============================================================================

adata = sc.read_h5ad(INPUT_FILE)

donor_counts = adata.obs['Donor'].value_counts()
donors_to_keep = donor_counts[donor_counts >= MIN_CELLS_PER_DONOR].index
adata = adata[adata.obs['Donor'].isin(donors_to_keep)].copy()

if sparse.issparse(adata.X):
    raw_counts = adata.X.toarray()
else:
    raw_counts = adata.X.copy()

library_sizes = raw_counts.sum(axis=1, keepdims=True)
cpm = (raw_counts / library_sizes) * 1e4
expr_data = np.log2(cpm + 1)

# ============================================================================
# STEP 2: Filter genes by expression and remove irrelevant genes
# ============================================================================

# Identify malignant and healthy cells
malignant_mask = adata.obs['consensus_label_6votes'] == 'LSPC'
healthy_mask = adata.obs['consensus_label_6votes'] == 'HSPC'
n_malignant_cells = malignant_mask.sum()
n_healthy_cells = healthy_mask.sum()

# Calculate expression metrics in MALIGNANT CELLS
expr_malignant = cpm[malignant_mask, :]
n_cells_expr_malignant = (expr_malignant > 0).sum(axis=0)
expr_pct_malignant = n_cells_expr_malignant / n_malignant_cells
mean_expr_malignant = expr_malignant.mean(axis=0)

# Calculate expression metrics in HEALTHY CELLS
expr_healthy = cpm[healthy_mask, :]
mean_expr_healthy = expr_healthy.mean(axis=0)

# Filter: genes must meet ALL criteria:
# 1. Expressed in at least MIN_EXPR_PCT of MALIGNANT cells
# 2. Mean expression >= MIN_MEAN_EXPR in malignant cells
# 3. LSPC expression > HSPC expression
if LSPC_higherthan_HSPC==True:
    gene_filter = (expr_pct_malignant >= MIN_EXPR_PCT) & \
                  (mean_expr_malignant >= MIN_MEAN_EXPR) & \
                  (mean_expr_malignant > mean_expr_healthy)
else:
    gene_filter = (expr_pct_malignant >= MIN_EXPR_PCT) & \
                  (mean_expr_malignant >= MIN_MEAN_EXPR)

# Remove ribosomal, mitochondrial, cell cycle, housekeeping and heat shock genes
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

# Create final gene filter
exclude_mask = np.array([g not in genes_to_exclude for g in adata.var_names])
final_gene_filter = gene_filter & exclude_mask

# Force include genes of interest
for gene in GENES_OF_INTEREST:
    if gene in adata.var_names:
        gene_idx = adata.var_names.get_loc(gene)
        final_gene_filter[gene_idx] = True

genes_kept = adata.var_names[final_gene_filter]
expr_filtered = expr_data[:, final_gene_filter]
cpm_filtered = cpm[:, final_gene_filter]

# ============================================================================
# STEP 3: Create design matrix for LMM (fixed effects only)
# ============================================================================

# Create metadata dataframe
metadata = adata.obs[['consensus_label_6votes', 'Study', 'Donor']].copy()

# Encode Status as binary (malignant=1, healthy=0)
metadata['Status_binary'] = (metadata['consensus_label_6votes'] == 'LSPC').astype(int)

# One-hot encode Study (drop first to avoid collinearity)
study_dummies = pd.get_dummies(metadata['Study'], prefix='Study', drop_first=True)

# Design matrix for FIXED effects: Status + Study
# NOTE: Donor is NOT included - it will be the random effect grouping variable
design_matrix = pd.concat([
    metadata[['Status_binary']],
    study_dummies
], axis=1)

# Add intercept
design_matrix.insert(0, 'Intercept', 1)
design_matrix = design_matrix.astype(float)

# Prepare groups variable (Donor IDs for random effects)
groups = metadata['Donor'].astype('category').cat.codes.values

# ============================================================================
# STEP 4: Run LMM-based DEG for each gene
# expression ~ Status + Study + (1|Donor)
# ============================================================================

results_list = []
convergence_failures = 0
numerical_errors = 0

for i in tqdm(range(len(genes_kept)), desc='  Testing genes'):
    gene = genes_kept[i]
    y = expr_filtered[:, i]

    # Skip genes with zero variance
    if np.var(y) == 0:
        continue

    # Fit LMM: y ~ Status + Study + (1|Donor)
    # groups parameter specifies the random effect grouping
    model = MixedLM(y, design_matrix, groups=groups)

    # Fit with reasonable convergence criteria
    result = model.fit(
        maxiter=100,
        method='lbfgs',
        reml=True  # Use REML for unbiased variance estimates
    )

    # Check convergence
    if not result.converged:
        convergence_failures += 1
        continue

    # Extract coefficient for Status
    status_coef = result.params['Status_binary']
    status_pval = result.pvalues['Status_binary']
    status_stderr = result.bse['Status_binary']

    # Calculate mean expression per group for log2FC
    malignant_idx = metadata['consensus_label_6votes'] == 'LSPC'
    healthy_idx = metadata['consensus_label_6votes'] == 'HSPC'

    # Calculate log2FC from CPM values
    cpm_malignant = cpm_filtered[:, i][malignant_idx]
    cpm_healthy = cpm_filtered[:, i][healthy_idx]

    mean_cpm_malignant = np.mean(cpm_malignant)
    mean_cpm_healthy = np.mean(cpm_healthy)

    log2fc = np.log2((mean_cpm_malignant + 1e-8) / (mean_cpm_healthy + 1e-8))

    # Percentage of cells expressing
    pct_malignant = (y[malignant_idx] > 0).sum() / malignant_idx.sum() * 100
    pct_healthy = (y[healthy_idx] > 0).sum() / healthy_idx.sum() * 100

    # Random effect variance (donor-level variance)
    random_effect_var = result.cov_re.iloc[0, 0] if hasattr(result, 'cov_re') else np.nan

    results_list.append({
        'gene': gene,
        'log2FoldChange': log2fc,
        'coefficient': status_coef,
        'stderr': status_stderr,
        'pvalue': status_pval,
        'mean_malignant': mean_cpm_malignant,
        'mean_healthy': mean_cpm_healthy,
        'pct_malignant': pct_malignant,
        'pct_healthy': pct_healthy,
        'pct_expressed': (y > 0).sum() / len(y) * 100,
        'random_effect_variance': random_effect_var
    })

results_df = pd.DataFrame(results_list)

# FDR correction
from statsmodels.stats.multitest import multipletests
results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]

# Sort by p-value
results_df = results_df.sort_values('padj').reset_index(drop=True)

# Save results
results_df.to_csv(f'{OUTPUT_DIR}/singlecell_deg_all_results.csv', index=False)

# ============================================================================
# STEP 5: Create volcano plot
# ============================================================================

from plot_volcano import create_volcano_plot

create_volcano_plot(
    results_df=results_df,
    output_dir=OUTPUT_DIR,
    output_prefix='singlecell',
    title='Single-Cell DEG with Linear Mixed Model\n(Malignant vs Healthy, Donor as Random Effect)',
    fc_thresh=FC_THRESH,
    pval_thresh=FDR_THRESH,
    n_labels=N_TOP_GENES_LABEL,
    genes_of_interest=GENES_OF_INTEREST
)

# ============================================================================
# STEP 6: Apply filtering and save results
# ============================================================================

sig_genes = results_df[
    (results_df['padj'] < FDR_THRESH) &
    (abs(results_df['log2FoldChange']) > FC_THRESH)
].copy()

sig_genes.to_csv(f'{OUTPUT_DIR}/singlecell_deg_significant.csv', index=False)

print(f'\nSingle-cell LMM analysis complete: {len(sig_genes)} significant genes (FDR<{FDR_THRESH}, |log2FC|>{FC_THRESH})')