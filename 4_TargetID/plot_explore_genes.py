#!/usr/bin/env python3
"""
Interactive Gene Expression Explorer

Usage:
  # Scatter plot for a gene pair
  python3 utility_explore_genes.py --scatter GENE1 GENE2

  # Distribution for single gene
  python3 utility_explore_genes.py --dist GENE

  # Distribution for multiple genes
  python3 utility_explore_genes.py --dist GENE1 GENE2 GENE3

  # Both scatter and distributions
  python3 utility_explore_genes.py --scatter GENE1 GENE2 --dist GENE1 GENE2

  # Save to specific file
  python3 utility_explore_genes.py --scatter GENE1 GENE2 --output my_plot.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(adata_path):
    """Load and prepare data."""
    adata = sc.read_h5ad(adata_path)

    # Filter to LSPC and HSPC
    valid_cells = adata.obs['consensus_label_6votes'].isin(['LSPC', 'HSPC'])
    adata = adata[valid_cells, :].copy()

    # Normalize
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    total_counts = X.sum(axis=1, keepdims=True)
    cpm = (X / total_counts) * 1e6
    log_cpm = np.log1p(cpm)

    return adata, log_cpm


def plot_scatter(adata, log_cpm, gene1, gene2, ax=None):
    """Plot scatter for gene pair."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if gene1 not in adata.var_names:
        ax.text(0.5, 0.5, f'{gene1} not found', ha='center', va='center', transform=ax.transAxes)
        return ax

    if gene2 not in adata.var_names:
        ax.text(0.5, 0.5, f'{gene2} not found', ha='center', va='center', transform=ax.transAxes)
        return ax

    idx1 = list(adata.var_names).index(gene1)
    idx2 = list(adata.var_names).index(gene2)

    x = log_cpm[:, idx1]
    y = log_cpm[:, idx2]

    cell_types = adata.obs['consensus_label_6votes'].values
    color_map = {'LSPC': '#d62728', 'HSPC': '#1f77b4'}

    # Plot LSPC on top
    lspc_mask = cell_types == 'LSPC'
    ax.scatter(x[lspc_mask], y[lspc_mask],
              c=color_map['LSPC'], s=5, alpha=0.4,
              rasterized=True, label=f'LSPC (n={lspc_mask.sum()})')

    # Plot HSPC
    hspc_mask = cell_types == 'HSPC'
    ax.scatter(x[hspc_mask], y[hspc_mask],
              c=color_map['HSPC'], s=5, alpha=0.4,
              rasterized=True, label=f'HSPC (n={hspc_mask.sum()})')

    # Calculate stats
    lspc_both = ((x[lspc_mask] > 0) & (y[lspc_mask] > 0)).sum()
    hspc_both = ((x[hspc_mask] > 0) & (y[hspc_mask] > 0)).sum()
    lspc_cov = lspc_both / lspc_mask.sum() * 100
    hspc_cov = hspc_both / hspc_mask.sum() * 100

    corr_lspc = np.corrcoef(x[lspc_mask], y[lspc_mask])[0, 1]
    corr_hspc = np.corrcoef(x[hspc_mask], y[hspc_mask])[0, 1]

    ax.set_xlabel(f'{gene1} [log1p(CPM)]', fontsize=18, fontweight='bold')
    ax.set_ylabel(f'{gene2} [log1p(CPM)]', fontsize=18, fontweight='bold')
    ax.set_title(
        f'{gene1} × {gene2}\n'
        f'Co-expr: LSPC={lspc_cov:.1f}%, HSPC={hspc_cov:.1f}%\n'
        f'Corr: LSPC={corr_lspc:.2f}, HSPC={corr_hspc:.2f}',
        fontsize=16.5, fontweight='bold'
    )
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)

    # Add marginal density
    ax_top = ax.inset_axes([0, 1.02, 1, 0.15])
    ax_top.hist([x[hspc_mask], x[lspc_mask]], bins=50,
                color=[color_map['HSPC'], color_map['LSPC']],
                alpha=0.6, label=['HSPC', 'LSPC'])
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_yticks([])
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    ax_right = ax.inset_axes([1.02, 0, 0.15, 1])
    ax_right.hist([y[hspc_mask], y[lspc_mask]], bins=50,
                  color=[color_map['HSPC'], color_map['LSPC']],
                  alpha=0.6, orientation='horizontal')
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_xticks([])
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    return ax


def plot_distribution(adata, log_cpm, genes, ax=None):
    """Plot distributions for genes."""
    if not isinstance(genes, list):
        genes = [genes]

    n_genes = len(genes)

    if ax is None:
        if n_genes == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            axes = [ax]
        else:
            n_cols = min(3, n_genes)
            n_rows = int(np.ceil(n_genes / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            axes = axes.flatten() if n_genes > 1 else [axes]
    else:
        axes = [ax]

    color_map = {'LSPC': '#d62728', 'HSPC': '#1f77b4'}
    cell_types = adata.obs['consensus_label_6votes'].values

    for idx, gene in enumerate(genes):
        if idx >= len(axes):
            break

        ax = axes[idx]

        if gene not in adata.var_names:
            ax.text(0.5, 0.5, f'{gene} not found', ha='center', va='center', transform=ax.transAxes)
            continue

        gene_idx = list(adata.var_names).index(gene)
        expr = log_cpm[:, gene_idx]

        lspc_mask = cell_types == 'LSPC'
        hspc_mask = cell_types == 'HSPC'

        lspc_expr = expr[lspc_mask]
        hspc_expr = expr[hspc_mask]

        # Violin plot
        parts = ax.violinplot(
            [lspc_expr, hspc_expr],
            positions=[0, 1],
            widths=0.6,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )

        for i, pc in enumerate(parts['bodies']):
            cell_type = ['LSPC', 'HSPC'][i]
            pc.set_facecolor(color_map[cell_type])
            pc.set_alpha(0.7)

        # Stats
        lspc_mean = lspc_expr.mean()
        hspc_mean = hspc_expr.mean()
        lspc_detect = (lspc_expr > 0).sum() / len(lspc_expr) * 100
        hspc_detect = (hspc_expr > 0).sum() / len(hspc_expr) * 100

        fold_change = lspc_mean - hspc_mean

        # Add box plot overlay
        bp = ax.boxplot(
            [lspc_expr, hspc_expr],
            positions=[0, 1],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='white', alpha=0.5),
            medianprops=dict(color='black', linewidth=2)
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f'LSPC\n({lspc_detect:.0f}% detected)',
            f'HSPC\n({hspc_detect:.0f}% detected)'
        ])
        ax.set_ylabel('log1p(CPM)', fontsize=16.5, fontweight='bold')
        ax.set_title(
            f'{gene}\n',
            #f'Mean: LSPC={lspc_mean:.2f}, HSPC={hspc_mean:.2f} (Δ={fold_change:.2f})',
            fontsize=16.5, fontweight='bold'
        )
        ax.grid(axis='y', alpha=0.3)

    # Remove empty subplots
    if ax is None:
        for idx in range(len(genes), len(axes)):
            fig.delaxes(axes[idx])

    return axes


def main():
    parser = argparse.ArgumentParser(
        description='Explore gene expression patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--scatter', nargs=2, metavar=('GENE1', 'GENE2'),
                       help='Create scatter plot for gene pair')
    parser.add_argument('--dist', nargs='+', metavar='GENE',
                       help='Create distribution plot(s) for gene(s)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: auto-generated)')
    parser.add_argument('--data', type=str,
                       default='4_TargetID/HSC_MPP_full_surface_filtered.h5ad',
                       help='Path to h5ad file')

    args = parser.parse_args()

    if not args.scatter and not args.dist:
        parser.print_help()
        sys.exit(1)

    # Load data
    print("Loading data...")
    adata, log_cpm = load_data(args.data)
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"LSPC: {(adata.obs['consensus_label_6votes']=='LSPC').sum()}")
    print(f"HSPC: {(adata.obs['consensus_label_6votes']=='HSPC').sum()}")

    # Determine layout
    n_plots = (1 if args.scatter else 0) + (1 if args.dist else 0)

    if n_plots == 1:
        fig = plt.figure(figsize=(10, 8))

        if args.scatter:
            ax = fig.add_subplot(111)
            plot_scatter(adata, log_cpm, args.scatter[0], args.scatter[1], ax)
            default_name = f"{args.scatter[0]}_vs_{args.scatter[1]}_scatter.pdf"
        else:
            plot_distribution(adata, log_cpm, args.dist)
            default_name = f"{'_'.join(args.dist)}_distribution.pdf"

    elif n_plots == 2:
        fig = plt.figure(figsize=(18, 8))

        ax1 = fig.add_subplot(121)
        plot_scatter(adata, log_cpm, args.scatter[0], args.scatter[1], ax1)

        if len(args.dist) == 1:
            ax2 = fig.add_subplot(122)
            plot_distribution(adata, log_cpm, args.dist[0], ax2)
        else:
            # Create grid for distributions
            n_dist = len(args.dist)
            n_cols = min(2, n_dist)
            n_rows = int(np.ceil(n_dist / n_cols))

            gs = fig.add_gridspec(n_rows, 2, left=0.55, right=0.98, hspace=0.3)
            for i, gene in enumerate(args.dist):
                row = i // n_cols
                col = i % n_cols
                ax = fig.add_subplot(gs[row, col])
                plot_distribution(adata, log_cpm, gene, ax)

        default_name = f"{args.scatter[0]}_vs_{args.scatter[1]}_with_dist.pdf"

    plt.tight_layout()

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path('4_TargetID/Outputs/Explore_gene_distribution') / default_name
        output_path.parent.mkdir(exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
