#!/usr/bin/env python3
"""
Unified volcano plot generator for DEG analyses
Works with both single-cell LMM and pseudobulk DESeq2 results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text


def create_volcano_plot(results_df, output_dir, output_prefix, title,
                        fc_thresh=1, pval_thresh=0.05, n_labels=10,
                        genes_of_interest=None):
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
    df['-log10_padj'] = df['-log10_padj'].clip(upper=80)

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
    ax.text(0.02, 0.98, f'{n_down} genes', transform=ax.transAxes, fontsize=12,
            fontweight='bold', color=colors['DOWN'], verticalalignment='top')
    ax.text(0.98, 0.98, f'{n_up} genes', transform=ax.transAxes, fontsize=12,
            fontweight='bold', color=colors['UP'], verticalalignment='top', horizontalalignment='right')

    # Labels and styling
    ax.set_xlabel('Malignant vs Healthy (log2FC)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Significance (-log10 adjusted p value, clipped at 80)', fontsize=14, fontweight='bold')
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
