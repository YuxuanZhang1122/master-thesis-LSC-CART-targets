#!/usr/bin/env python3
"""
Violin plots showing dual coverage distribution for multiple gene pairs.
Plots: CD9, CD9+CD47, CD9+CD99 and EMB, EMB+CD47, EMB+HCST
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scanpy as sc
import seaborn as sns

BASE_DIR = Path(__file__).parent
ADATA_PATH = BASE_DIR / "HSC_MPP_full_surface_filtered.h5ad"
OUTPUT_DIR = BASE_DIR / "Pair_search/statistical_interaction_binary_interaction/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_PAIRS = {
    'CD9': ['CD47', 'CD99'],
    'EMB': ['CD47', 'HCST']
}

print("Loading data...")
adata = sc.read_h5ad(ADATA_PATH)
valid_cells = adata.obs['consensus_label_6votes'].isin(['LSPC', 'HSPC'])
adata = adata[valid_cells, :].copy()

print(f"Total cells: {adata.n_obs}")
print(f"LSPC: {(adata.obs['consensus_label_6votes'] == 'LSPC').sum()}")
print(f"HSPC: {(adata.obs['consensus_label_6votes'] == 'HSPC').sum()}")

def calculate_single_gene_coverage_by_donor(adata, gene):
    """Calculate coverage percentage for a single gene across donors."""
    donors = adata.obs['Donor'].unique()
    results = []

    for donor in donors:
        donor_cells = adata.obs['Donor'] == donor
        donor_adata = adata[donor_cells, :].copy()

        lspc_mask = donor_adata.obs['consensus_label_6votes'] == 'LSPC'
        hspc_mask = donor_adata.obs['consensus_label_6votes'] == 'HSPC'

        n_lspc = lspc_mask.sum()
        n_hspc = hspc_mask.sum()

        if n_lspc < 20 or n_hspc < 20:
            continue

        gene_expr = donor_adata[:, gene].X.toarray().flatten() if hasattr(donor_adata[:, gene].X, 'toarray') else donor_adata[:, gene].X.flatten()

        lspc_coverage = (gene_expr[lspc_mask] > 0).mean() * 100
        hspc_coverage = (gene_expr[hspc_mask] > 0).mean() * 100

        results.append({
            'donor': donor,
            'lspc_coverage_pct': lspc_coverage,
            'hspc_coverage_pct': hspc_coverage,
            'n_lspc': n_lspc,
            'n_hspc': n_hspc
        })

    return pd.DataFrame(results)

def calculate_dual_expression_by_donor(adata, gene1, gene2):
    """Calculate co-expression percentage for gene pair across donors."""
    donors = adata.obs['Donor'].unique()
    results = []

    for donor in donors:
        donor_cells = adata.obs['Donor'] == donor
        donor_adata = adata[donor_cells, :].copy()

        lspc_mask = donor_adata.obs['consensus_label_6votes'] == 'LSPC'
        hspc_mask = donor_adata.obs['consensus_label_6votes'] == 'HSPC'

        n_lspc = lspc_mask.sum()
        n_hspc = hspc_mask.sum()

        if n_lspc < 20 or n_hspc < 20:
            continue

        g1_expr = donor_adata[:, gene1].X.toarray().flatten() if hasattr(donor_adata[:, gene1].X, 'toarray') else donor_adata[:, gene1].X.flatten()
        g2_expr = donor_adata[:, gene2].X.toarray().flatten() if hasattr(donor_adata[:, gene2].X, 'toarray') else donor_adata[:, gene2].X.flatten()

        lspc_coexpr = ((g1_expr[lspc_mask] > 0) & (g2_expr[lspc_mask] > 0)).mean() * 100
        hspc_coexpr = ((g1_expr[hspc_mask] > 0) & (g2_expr[hspc_mask] > 0)).mean() * 100

        results.append({
            'donor': donor,
            'lspc_coexpression_pct': lspc_coexpr,
            'hspc_coexpression_pct': hspc_coexpr,
            'n_lspc': n_lspc,
            'n_hspc': n_hspc
        })

    return pd.DataFrame(results)

def prepare_plot_data(anchor_gene, partner_genes, adata):
    """Prepare data for plotting anchor gene and its pairs."""
    all_data = []

    # Anchor gene coverage
    print(f"\nCalculating coverage for {anchor_gene}...")
    anchor_coverage = calculate_single_gene_coverage_by_donor(adata, anchor_gene)

    anchor_lspc = anchor_coverage[['lspc_coverage_pct']].copy()
    anchor_lspc['cell_type'] = 'LSPC'
    anchor_lspc['metric'] = anchor_gene
    anchor_lspc.rename(columns={'lspc_coverage_pct': 'coverage_pct'}, inplace=True)

    anchor_hspc = anchor_coverage[['hspc_coverage_pct']].copy()
    anchor_hspc['cell_type'] = 'HSPC'
    anchor_hspc['metric'] = anchor_gene
    anchor_hspc.rename(columns={'hspc_coverage_pct': 'coverage_pct'}, inplace=True)

    all_data.extend([anchor_lspc, anchor_hspc])

    # Partner gene pairs
    for partner in partner_genes:
        print(f"Calculating co-expression for {anchor_gene}+{partner}...")
        dual_coexpr = calculate_dual_expression_by_donor(adata, anchor_gene, partner)

        coexpr_lspc = dual_coexpr[['lspc_coexpression_pct']].copy()
        coexpr_lspc['cell_type'] = 'LSPC'
        coexpr_lspc['metric'] = f'{anchor_gene}+{partner}'
        coexpr_lspc.rename(columns={'lspc_coexpression_pct': 'coverage_pct'}, inplace=True)

        coexpr_hspc = dual_coexpr[['hspc_coexpression_pct']].copy()
        coexpr_hspc['cell_type'] = 'HSPC'
        coexpr_hspc['metric'] = f'{anchor_gene}+{partner}'
        coexpr_hspc.rename(columns={'hspc_coexpression_pct': 'coverage_pct'}, inplace=True)

        all_data.extend([coexpr_lspc, coexpr_hspc])

    return pd.concat(all_data, ignore_index=True)

# Create plots for each anchor gene
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for idx, (anchor_gene, partner_genes) in enumerate(GENE_PAIRS.items()):
    print(f"\n{'='*70}")
    print(f"Processing {anchor_gene} with partners: {', '.join(partner_genes)}")
    print('='*70)

    plot_data = prepare_plot_data(anchor_gene, partner_genes, adata)

    ax = axes[idx]
    sns.violinplot(
        data=plot_data,
        x='metric',
        y='coverage_pct',
        hue='cell_type',
        palette={'LSPC': '#d62728', 'HSPC': '#1f77b4'},
        ax=ax,
        inner='quartile',
        split=True
    )

    ax.set_xlabel('', fontsize=13)
    ax.set_ylabel('Coverage (%)' if idx == 0 else '', fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_title(anchor_gene, fontsize=14, fontweight='bold')

    if idx == 1:
        ax.legend(title='', fontsize=11, frameon=False)
    else:
        ax.get_legend().remove()

    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
output_path = OUTPUT_DIR / "dual_coverage_multipairs_combined.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"\n{'='*70}")
print(f"Saved combined plot: {output_path}")
print('='*70)

# Create individual plots for each anchor gene
for anchor_gene, partner_genes in GENE_PAIRS.items():
    plot_data = prepare_plot_data(anchor_gene, partner_genes, adata)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(
        data=plot_data,
        x='metric',
        y='coverage_pct',
        hue='cell_type',
        palette={'LSPC': '#d62728', 'HSPC': '#1f77b4'},
        ax=ax,
        inner='quartile',
        split=True
    )

    ax.set_xlabel('', fontsize=13)
    ax.set_ylabel('Coverage (%)', fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_title(f'{anchor_gene} and partner combinations', fontsize=14, fontweight='bold')
    ax.legend(title='', fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3, axis='y')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"dual_coverage_{anchor_gene}_pairs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved {anchor_gene} plot: {output_path}")

# Save summary statistics
summary_stats = []

for anchor_gene, partner_genes in GENE_PAIRS.items():
    anchor_coverage = calculate_single_gene_coverage_by_donor(adata, anchor_gene)

    summary_stats.append({
        'gene_pair': anchor_gene,
        'lspc_mean': anchor_coverage['lspc_coverage_pct'].mean(),
        'lspc_std': anchor_coverage['lspc_coverage_pct'].std(),
        'hspc_mean': anchor_coverage['hspc_coverage_pct'].mean(),
        'hspc_std': anchor_coverage['hspc_coverage_pct'].std()
    })

    for partner in partner_genes:
        dual_coexpr = calculate_dual_expression_by_donor(adata, anchor_gene, partner)
        summary_stats.append({
            'gene_pair': f'{anchor_gene}+{partner}',
            'lspc_mean': dual_coexpr['lspc_coexpression_pct'].mean(),
            'lspc_std': dual_coexpr['lspc_coexpression_pct'].std(),
            'hspc_mean': dual_coexpr['hspc_coexpression_pct'].mean(),
            'hspc_std': dual_coexpr['hspc_coexpression_pct'].std()
        })

summary_df = pd.DataFrame(summary_stats)
summary_path = OUTPUT_DIR / "dual_coverage_multipairs_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary statistics: {summary_path}")

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
for _, row in summary_df.iterrows():
    print(f"\n{row['gene_pair']}:")
    print(f"  LSPC: {row['lspc_mean']:.1f}% ± {row['lspc_std']:.1f}%")
    print(f"  HSPC: {row['hspc_mean']:.1f}% ± {row['hspc_std']:.1f}%")
