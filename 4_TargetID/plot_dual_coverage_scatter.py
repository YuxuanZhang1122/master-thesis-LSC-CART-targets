#!/usr/bin/env python3
"""
Scatter plots showing dual coverage distribution across donors for gene pairs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scanpy as sc
import argparse

parser = argparse.ArgumentParser(description='Plot dual coverage scatter plots for gene pairs')
parser.add_argument('--gene1', default='CD9', help='First gene (default: CD9)')
parser.add_argument('--gene2', default='CD47', help='Second gene (default: CD47)')
parser.add_argument('--output-dir', default=None, help='Output directory (default: Pair_search/statistical_interaction_binary_interaction/figures/)')
parser.add_argument('--error-type', default='std', choices=['std', 'sem'], help='Error bar type: std (default) or sem')
args = parser.parse_args()

BASE_DIR = Path(__file__).parent
ADATA_PATH = BASE_DIR / "HSC_MPP_full_surface_filtered.h5ad"

if args.output_dir:
    OUTPUT_DIR = Path(args.output_dir)
else:
    OUTPUT_DIR = BASE_DIR / "Pair_search/statistical_interaction_binary_interaction/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / f"dual_coverage_scatter_{args.gene1}_{args.gene2}.png"

print(f"Analyzing gene pair: {args.gene1} + {args.gene2}")
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

# Calculate anchor gene coverage and co-expression
print(f"\nCalculating coverage for {args.gene1} (anchor)...")
gene1_coverage = calculate_single_gene_coverage_by_donor(adata, args.gene1)

print(f"\nCalculating co-expression for {args.gene1}+{args.gene2}...")
dual_coexpr = calculate_dual_expression_by_donor(adata, args.gene1, args.gene2)

print(f"Valid donors: {len(gene1_coverage)}")

# Prepare data for 4 columns
data_groups = [
    (gene1_coverage['lspc_coverage_pct'], f'{args.gene1}\nLSPC', '#d62728'),
    (gene1_coverage['hspc_coverage_pct'], f'{args.gene1}\nHSPC', '#1f77b4'),
    (dual_coexpr['lspc_coexpression_pct'], f'{args.gene1}+{args.gene2}\nLSPC', '#d62728'),
    (dual_coexpr['hspc_coexpression_pct'], f'{args.gene1}+{args.gene2}\nHSPC', '#1f77b4')
]

# Create scatter plot
fig, ax = plt.subplots(figsize=(6, 6))

x_positions = [1, 2, 3, 4]
x_labels = [label for _, label, _ in data_groups]

for idx, (data, label, color) in enumerate(data_groups):
    x_pos = x_positions[idx]

    # Add jitter to x-axis for visibility
    x_jitter = np.random.normal(0, 0.04, size=len(data))
    ax.scatter(x_pos + x_jitter, data, color=color, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

    # Calculate mean and error
    mean_val = data.mean()
    if args.error_type == 'sem':
        error_val = data.sem()
    else:
        error_val = data.std()

    # Plot mean as horizontal line
    ax.plot([x_pos - 0.15, x_pos + 0.15], [mean_val, mean_val], color='black', linewidth=2)

    # Plot error bars
    ax.errorbar(x_pos, mean_val, yerr=error_val, fmt='none', color='black', linewidth=1.5, capsize=0)

# Styling
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel('Coverage (%)', fontsize=13)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')

# Remove frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', transparent=True)

print(f"\nSaved: {OUTPUT_PATH}")

# Save coverage data to CSV
csv_path = OUTPUT_DIR / f"coverage_data_{args.gene1}_{args.gene2}.csv"
coverage_csv = pd.DataFrame({
    'donor': gene1_coverage['donor'],
    f'{args.gene1}_lspc_coverage': gene1_coverage['lspc_coverage_pct'],
    f'{args.gene1}_hspc_coverage': gene1_coverage['hspc_coverage_pct'],
    'coexpr_lspc_coverage': dual_coexpr['lspc_coexpression_pct'],
    'coexpr_hspc_coverage': dual_coexpr['hspc_coexpression_pct'],
    'n_lspc': gene1_coverage['n_lspc'],
    'n_hspc': gene1_coverage['n_hspc']
})
coverage_csv.to_csv(csv_path, index=False)
print(f"Saved coverage data: {csv_path}")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\n{args.gene1} (anchor) Coverage (%):")
print(f"  LSPC: mean={gene1_coverage['lspc_coverage_pct'].mean():.1f}%, {args.error_type}={gene1_coverage['lspc_coverage_pct'].std() if args.error_type == 'std' else gene1_coverage['lspc_coverage_pct'].sem():.1f}%")
print(f"  HSPC: mean={gene1_coverage['hspc_coverage_pct'].mean():.1f}%, {args.error_type}={gene1_coverage['hspc_coverage_pct'].std() if args.error_type == 'std' else gene1_coverage['hspc_coverage_pct'].sem():.1f}%")

print(f"\n{args.gene1}+{args.gene2} Co-expression (%):")
print(f"  LSPC: mean={dual_coexpr['lspc_coexpression_pct'].mean():.1f}%, {args.error_type}={dual_coexpr['lspc_coexpression_pct'].std() if args.error_type == 'std' else dual_coexpr['lspc_coexpression_pct'].sem():.1f}%")
print(f"  HSPC: mean={dual_coexpr['hspc_coexpression_pct'].mean():.1f}%, {args.error_type}={dual_coexpr['hspc_coexpression_pct'].std() if args.error_type == 'std' else dual_coexpr['hspc_coexpression_pct'].sem():.1f}%")
