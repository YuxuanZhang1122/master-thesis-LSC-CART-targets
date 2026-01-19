#!/usr/bin/env python3
"""
UMAP visualization showing NPM1 mutation distribution relative to LSPC/HSPC populations.
- Gray background: all cells from HSC_MPP_scANVI.h5ad
- Dashed contour: LSPC population boundary
- Highlighted: cells with NPM1 mutation info from subtype_analysis.h5ad
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path

BASE_DIR = Path(__file__).parent
BG_ADATA_PATH = BASE_DIR / "HSC_MPP_scANVI.h5ad"
SUBTYPE_ADATA_PATH = BASE_DIR / "subtype_analysis.h5ad"
OUTPUT_DIR = BASE_DIR / "Outputs/npm1_hspc_lspc_expression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load background data (all cells with UMAP)
adata_bg = sc.read_h5ad(BG_ADATA_PATH)

# Load subtype data (cells with NPM1 info)
adata_sub = sc.read_h5ad(SUBTYPE_ADATA_PATH)

# Get UMAP coordinates for subtype cells from background
common_cells = list(set(adata_bg.obs_names) & set(adata_sub.obs_names))
cell_to_idx_bg = {c: i for i, c in enumerate(adata_bg.obs_names)}
cell_to_idx_sub = {c: i for i, c in enumerate(adata_sub.obs_names)}

# Create mapping for subtype cells
subtype_umap = np.zeros((len(common_cells), 2))
subtype_npm1 = []
subtype_celltype = []

for i, cell in enumerate(common_cells):
    bg_idx = cell_to_idx_bg[cell]
    sub_idx = cell_to_idx_sub[cell]
    subtype_umap[i] = adata_bg.obsm['X_umap'][bg_idx]
    subtype_npm1.append(adata_sub.obs['NPM1_Mutation'].iloc[sub_idx])
    subtype_celltype.append(adata_sub.obs['consensus_label_6votes'].iloc[sub_idx])

subtype_npm1 = np.array(subtype_npm1)
subtype_celltype = np.array(subtype_celltype)

# Get background UMAP
bg_umap = adata_bg.obsm['X_umap']
bg_celltype = adata_bg.obs['consensus_label_6votes'].values


def add_lspc_contour(ax, umap_coords, celltype_labels, color='grey', alpha=0.5):
    """Add dashed contour around LSPC population."""
    mask = celltype_labels == 'LSPC'
    points = umap_coords[mask]

    if len(points) > 10:
        padding = 2
        x_min, x_max = umap_coords[:, 0].min() - padding, umap_coords[:, 0].max() + padding
        y_min, y_max = umap_coords[:, 1].min() - padding, umap_coords[:, 1].max() + padding

        kde = gaussian_kde(points.T, bw_method=0.6)

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)

        ax.contour(xx, yy, density, levels=[density.max() * 0.1],
                   colors=color, linestyles='--', linewidths=2, alpha=alpha)


# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Colors for NPM1 status
colors = {'Pos': '#d62728', 'Neg': '#1f77b4'}  # Red for Mutated, Blue for WT

# Panel titles and masks
panels = [
    ('NPM1 Mutated', subtype_npm1 == 'Pos', 'Pos'),
    ('NPM1 Wild-type', subtype_npm1 == 'Neg', 'Neg')
]

# Get region centers for labels
lspc_center = bg_umap[bg_celltype == 'LSPC'].mean(axis=0)
hspc_center = bg_umap[bg_celltype == 'HSPC'].mean(axis=0)

for ax, (title, mask, npm1_status) in zip(axes, panels):
    # Plot all background cells in gray
    ax.scatter(bg_umap[:, 0], bg_umap[:, 1],
               c='lightgray', s=5, alpha=0.3, rasterized=True)

    # Add LSPC contour
    add_lspc_contour(ax, bg_umap, bg_celltype)

    # Highlight cells with NPM1-specific color
    n_cells = mask.sum()
    ax.scatter(subtype_umap[mask, 0], subtype_umap[mask, 1],
               c=colors[npm1_status], s=3, alpha=0.5, rasterized=True)

    # Add text annotations for regions (no box)
    ax.text(lspc_center[0], lspc_center[1] + 3, 'LSPC', fontsize=14, fontweight='bold',
            ha='center', va='bottom', color='black')
    ax.text(hspc_center[0], hspc_center[1] - 3, 'HSPC', fontsize=14, fontweight='bold',
            ha='center', va='top', color='black')

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(f'{title} (n={n_cells})', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save
output_pdf = OUTPUT_DIR / "npm1_umap_distribution.pdf"
output_png = OUTPUT_DIR / "npm1_umap_distribution.png"
plt.savefig(output_pdf, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')

# Calculate statistics
npm1_pos_mask = subtype_npm1 == 'Pos'
npm1_neg_mask = subtype_npm1 == 'Neg'

stats_data = []
for npm1, npm1_mask in [('NPM1 Mut', npm1_pos_mask), ('NPM1 WT', npm1_neg_mask)]:
    for ct in ['LSPC', 'HSPC']:
        ct_mask = subtype_celltype == ct
        combined = npm1_mask & ct_mask
        stats_data.append({
            'NPM1_Status': npm1,
            'Cell_Type': ct,
            'n_cells': combined.sum(),
            'pct_of_npm1_group': combined.sum() / npm1_mask.sum() * 100 if npm1_mask.sum() > 0 else 0
        })

stats_df = pd.DataFrame(stats_data)
stats_csv = OUTPUT_DIR / "npm1_umap_stats.csv"
stats_df.to_csv(stats_csv, index=False)
