#!/usr/bin/env python3
"""
Plot CD9 and EMB expression rates in HSPC/LSPC stratified by NPM1 mutation status.
Expression is defined as % cells with raw count > 0 per donor.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).parent
ADATA_PATH = BASE_DIR / "subtype_analysis.h5ad"
OUTPUT_DIR = BASE_DIR / "Outputs/npm1_hspc_lspc_expression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENES = ['CD9', 'EMB']
MIN_CELLS = 10  # Minimum cells per group per donor

# Load data
adata = sc.read_h5ad(ADATA_PATH)

# Check for consensus prediction column - try different naming conventions
consensus_col = None
for col in ['consensus_prediction', 'consensus_label_6votes', 'consensus_label_5votes']:
    if col in adata.obs.columns:
        consensus_col = col
        break

if consensus_col is None:
    raise ValueError("No consensus prediction column found")

# Filter to HSPC and LSPC only
valid_cells = adata.obs[consensus_col].isin(['LSPC', 'HSPC'])
adata = adata[valid_cells, :].copy()

# Check NPM1 column
npm1_col = 'NPM1_Mutation'
if npm1_col not in adata.obs.columns:
    raise ValueError(f"{npm1_col} column not found")

# Map NPM1 values to more readable labels
npm1_map = {'Pos': 'NPM1 Mut', 'Neg': 'NPM1 WT'}
adata.obs['NPM1_Status'] = adata.obs[npm1_col].map(npm1_map)


def calculate_expression_rate_by_donor(adata, gene, consensus_col):
    """Calculate expression rate (% cells with count > 0) per donor per cell type per NPM1 status."""
    donors = adata.obs['Donor'].unique()
    results = []

    for donor in donors:
        donor_mask = adata.obs['Donor'] == donor
        donor_adata = adata[donor_mask, :]

        npm1_status = donor_adata.obs['NPM1_Status'].iloc[0]

        for cell_type in ['HSPC', 'LSPC']:
            ct_mask = donor_adata.obs[consensus_col] == cell_type
            n_cells = ct_mask.sum()

            if n_cells < MIN_CELLS:
                continue

            gene_expr = donor_adata[:, gene].X.toarray().flatten() if hasattr(donor_adata[:, gene].X, 'toarray') else donor_adata[:, gene].X.flatten()
            expr_rate = (gene_expr[ct_mask] > 0).mean() * 100

            results.append({
                'Donor': donor,
                'Cell_Type': cell_type,
                'NPM1_Status': npm1_status,
                'Expression_Rate': expr_rate,
                'n_cells': n_cells,
                'Group': f"{npm1_status} - {cell_type}"
            })

    return pd.DataFrame(results)


# Calculate expression rates for each gene
gene_data = {}
for gene in GENES:
    if gene not in adata.var_names:
        continue
    gene_data[gene] = calculate_expression_rate_by_donor(adata, gene, consensus_col)

# Define group order and colors
group_order = ['NPM1 Mut - HSPC', 'NPM1 Mut - LSPC', 'NPM1 WT - HSPC', 'NPM1 WT - LSPC']
colors = {
    'NPM1 Mut - HSPC': '#6baed6',  # Light blue
    'NPM1 Mut - LSPC': '#d62728',  # Red
    'NPM1 WT - HSPC': '#1f77b4',   # Dark blue
    'NPM1 WT - LSPC': '#ff7f0e',   # Orange
}

# Create figure
fig, axes = plt.subplots(1, len(gene_data), figsize=(5 * len(gene_data), 6))
if len(gene_data) == 1:
    axes = [axes]

for idx, (gene, df) in enumerate(gene_data.items()):
    ax = axes[idx]

    # Filter to valid groups
    df = df[df['Group'].isin(group_order)].copy()
    df['Group'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)

    # Box plot with individual points
    sns.boxplot(
        data=df,
        x='Group',
        y='Expression_Rate',
        order=group_order,
        palette=colors,
        width=0.6,
        fliersize=0,
        ax=ax
    )

    # Add individual donor points
    sns.stripplot(
        data=df,
        x='Group',
        y='Expression_Rate',
        order=group_order,
        color='black',
        size=4,
        alpha=0.6,
        ax=ax
    )

    ax.set_xlabel('')
    ax.set_ylabel('Expression Rate (%)', fontsize=12)
    ax.set_title(f'{gene}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_xticklabels([g.replace(' - ', '\n') for g in group_order], fontsize=10)

    # Add sample sizes
    for i, group in enumerate(group_order):
        n = (df['Group'] == group).sum()
        ax.text(i, -10, f'n={n}', ha='center', va='top', fontsize=9)

    # Statistical tests: compare LSPC vs HSPC within each NPM1 status
    for npm1 in ['NPM1 Mut', 'NPM1 WT']:
        hspc_vals = df[(df['NPM1_Status'] == npm1) & (df['Cell_Type'] == 'HSPC')]['Expression_Rate']
        lspc_vals = df[(df['NPM1_Status'] == npm1) & (df['Cell_Type'] == 'LSPC')]['Expression_Rate']

        if len(hspc_vals) >= 3 and len(lspc_vals) >= 3:
            stat, pval = stats.mannwhitneyu(hspc_vals, lspc_vals, alternative='two-sided')

            # Position for significance annotation
            if npm1 == 'NPM1 Mut':
                x1, x2 = 0, 1
                y_pos = 92
            else:
                x1, x2 = 2, 3
                y_pos = 92

            if pval < 0.001:
                sig_text = '***'
            elif pval < 0.01:
                sig_text = '**'
            elif pval < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'

            ax.plot([x1, x1, x2, x2], [y_pos-2, y_pos, y_pos, y_pos-2], 'k-', linewidth=1)
            ax.text((x1+x2)/2, y_pos+1, sig_text, ha='center', va='bottom', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figures
output_pdf = OUTPUT_DIR / "npm1_hspc_lspc_expression_boxplot.pdf"
output_png = OUTPUT_DIR / "npm1_hspc_lspc_expression_boxplot.png"
plt.savefig(output_pdf, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')

# Save summary statistics
summary_rows = []
for gene, df in gene_data.items():
    for group in group_order:
        group_df = df[df['Group'] == group]
        if len(group_df) > 0:
            summary_rows.append({
                'Gene': gene,
                'Group': group,
                'Mean_Expression_Rate': group_df['Expression_Rate'].mean(),
                'Std_Expression_Rate': group_df['Expression_Rate'].std(),
                'Median_Expression_Rate': group_df['Expression_Rate'].median(),
                'n_donors': len(group_df)
            })

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUTPUT_DIR / "npm1_hspc_lspc_expression_summary.csv"
summary_df.to_csv(summary_csv, index=False)