#!/usr/bin/env python3
"""
Dot plot showing efficacy vs specificity for each anchor gene.
Efficacy = dual coverage on LSPC
Specificity = dual coverage on LSPC / dual coverage on HSPC
Color = Synergy Score (lspc_deviation - hspc_deviation)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc

BASE_DIR = Path(__file__).parent
RESULTS_PATH = BASE_DIR / "Pair_search/statistical_interaction/results/positive_interactions.csv"
ADATA_PATH = BASE_DIR / "HSC_MPP_full_surface_filtered.h5ad"
OUTPUT_DIR = BASE_DIR / "Pair_search/statistical_interaction/figures/anchor_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
df = pd.read_csv(RESULTS_PATH)

# Convert coverage to percentage
df['efficacy_pct'] = df['lspc_coverage'] * 100
df['hspc_coverage_pct'] = df['hspc_coverage'] * 100

# Load h5ad to calculate individual gene coverages
print("Loading h5ad to calculate synergy scores...")
adata = sc.read_h5ad(ADATA_PATH)
valid_cells = adata.obs['consensus_label_6votes'].isin(['LSPC', 'HSPC'])
adata = adata[valid_cells, :].copy()

lspc_mask = adata.obs['consensus_label_6votes'] == 'LSPC'
hspc_mask = adata.obs['consensus_label_6votes'] == 'HSPC'
n_lspc = lspc_mask.sum()
n_hspc = hspc_mask.sum()

X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# Calculate individual gene coverages
def get_gene_coverage(gene):
    if gene not in adata.var_names:
        return np.nan, np.nan
    idx = adata.var_names.get_loc(gene)
    lspc_cov = 100 * (X[lspc_mask, idx] > 0).sum() / n_lspc
    hspc_cov = 100 * (X[hspc_mask, idx] > 0).sum() / n_hspc
    return lspc_cov, hspc_cov

# Calculate for all gene pairs
gene1_lspc, gene1_hspc = [], []
gene2_lspc, gene2_hspc = [], []

for _, row in df.iterrows():
    g1_lspc, g1_hspc = get_gene_coverage(row['gene1'])
    g2_lspc, g2_hspc = get_gene_coverage(row['gene2'])
    gene1_lspc.append(g1_lspc)
    gene1_hspc.append(g1_hspc)
    gene2_lspc.append(g2_lspc)
    gene2_hspc.append(g2_hspc)

df['gene1_lspc'] = gene1_lspc
df['gene1_hspc'] = gene1_hspc
df['gene2_lspc'] = gene2_lspc
df['gene2_hspc'] = gene2_hspc

# Calculate expected co-expression (assuming independence)
df['expected_lspc'] = (df['gene1_lspc'] * df['gene2_lspc']) / 100
df['expected_hspc'] = (df['gene1_hspc'] * df['gene2_hspc']) / 100

# Calculate deviations (observed - expected)
df['lspc_deviation'] = df['efficacy_pct'] - df['expected_lspc']
df['hspc_deviation'] = df['hspc_coverage_pct'] - df['expected_hspc']

# Calculate synergy score
df['synergy_score'] = df['lspc_deviation'] - df['hspc_deviation']

print(f"Calculated synergy scores for {len(df)} pairs")

def plot_anchor_efficacy_specificity(anchor_gene, df, output_dir, top_n_label=10):
    """
    Create dot plot for a specific anchor gene.

    Parameters
    ----------
    anchor_gene : str
        Anchor gene to plot
    df : pd.DataFrame
        Results dataframe
    output_dir : Path
        Output directory
    top_n_label : int
        Number of top partners to label
    """
    # Filter to anchor
    anchor_df = df[df['gene1'] == anchor_gene].copy()

    if len(anchor_df) == 0:
        print(f"No data for anchor {anchor_gene}")
        return

    # Sort by efficacy for labeling
    anchor_df = anchor_df.sort_values('efficacy_pct', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter plot with uniform color
    scatter = ax.scatter(
        anchor_df['specificity_ratio'],
        anchor_df['efficacy_pct'],
        c='steelblue',
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Label top partners
    for idx, row in anchor_df.head(top_n_label).iterrows():
        ax.annotate(
            row['gene2'],
            (row['specificity_ratio'], row['efficacy_pct']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none')
        )

    # Labels and title
    ax.set_xlabel('Specificity (Dual LSPC% / Dual HSPC%)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Efficacy (Dual Coverage on LSPC, %)', fontsize=15, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(f'{anchor_gene}: Partner Gene Efficacy vs Specificity\n(n={len(anchor_df)} partners)',
                 fontsize=14, fontweight='bold', pad=15)

    # Grid
    ax.grid(True, alpha=0, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save
    output_path = output_dir / f'{anchor_gene}_efficacy_specificity.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    # Print summary statistics
    print(f"\n{anchor_gene} Summary:")
    print(f"  Total partners: {len(anchor_df)}")
    print(f"  Efficacy range: {anchor_df['efficacy_pct'].min():.1f}% - {anchor_df['efficacy_pct'].max():.1f}%")
    print(f"  Specificity range: {anchor_df['specificity_ratio'].min():.1f}x - {anchor_df['specificity_ratio'].max():.1f}x")
    print(f"  Top 5 by efficacy:")
    for idx, row in anchor_df.head(5).iterrows():
        print(f"    {row['gene2']:12s}: {row['efficacy_pct']:5.1f}% efficacy, {row['specificity_ratio']:5.1f}x specificity")

# Start with EMB
print("Generating plot for EMB anchor...")
plot_anchor_efficacy_specificity('EMB', df, OUTPUT_DIR, top_n_label=15)

print("\nDone!")
