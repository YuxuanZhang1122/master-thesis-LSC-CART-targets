#!/usr/bin/env python3
"""
Plot UMAP projections across different uncertainty thresholds.
Shows how cell populations change with increasing uncertainty cutoffs.
"""

from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CELL_TYPES = ['HSC MPP', 'LMPP', 'Early GMP']
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
EMBEDDINGS_DIR = Path('../outputs/embeddings')
OUTPUT_DIR = Path('../outputs/figures')

# Cell type colors
CELL_TYPE_COLORS = {
    'HSC MPP': '#1f77b4',
    'LMPP': '#ff7f0e',
    'Early GMP': '#2ca02c'
}

def load_reference_umap():
    """Load reference UMAP coordinates."""
    ref_path = EMBEDDINGS_DIR / 'reference_embeddings.h5ad'
    ref_adata = ad.read_h5ad(ref_path)
    return ref_adata.obsm['X_umap']

def load_query_data():
    """Load and combine query embeddings from specified datasets."""
    dataset_names = [
        'Henrik_DG_embeddings.h5ad',
        'Naldini_V02_embeddings.h5ad',
        'Naldini_V03_embeddings.h5ad',
        'Ennis_embeddings.h5ad',
        'Petti_DG_embeddings.h5ad'
    ]

    all_data = []
    for name in dataset_names:
        f = EMBEDDINGS_DIR / name
        if not f.exists():
            continue

        adata = ad.read_h5ad(f)
        if 'X_umap_combined' not in adata.obsm:
            continue

        data_dict = {
            'umap': adata.obsm['X_umap_combined'],
            'cell_type': adata.obs['predicted_CellType_Broad'].values,
            'uncertainty': adata.obs['uncertainty_CellType_Broad'].values
        }
        all_data.append(data_dict)

    # Concatenate all datasets
    combined = {
        'umap': np.vstack([d['umap'] for d in all_data]),
        'cell_type': np.concatenate([d['cell_type'] for d in all_data]),
        'uncertainty': np.concatenate([d['uncertainty'] for d in all_data])
    }

    return combined

def filter_cells(data, cell_type, threshold):
    """Filter cells by cell type and uncertainty threshold."""
    mask = (data['cell_type'] == cell_type) & (data['uncertainty'] <= threshold)
    return {
        'umap': data['umap'][mask],
        'count': mask.sum()
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    ref_umap = load_reference_umap()
    query_data = load_query_data()

    # Create figure
    fig, axes = plt.subplots(3, 7, figsize=(35, 15))

    for row_idx, cell_type in enumerate(CELL_TYPES):
        # Get total count at threshold=1.0
        total_cells = ((query_data['cell_type'] == cell_type) &
                      (query_data['uncertainty'] <= 1.0)).sum()

        for col_idx, threshold in enumerate(THRESHOLDS):
            ax = axes[row_idx, col_idx]

            # Plot reference background
            ax.scatter(ref_umap[:, 0], ref_umap[:, 1],
                      c='lightgray', s=1, alpha=0.3, rasterized=True)

            # Filter and plot query cells
            filtered = filter_cells(query_data, cell_type, threshold)

            if filtered['count'] > 0:
                ax.scatter(filtered['umap'][:, 0], filtered['umap'][:, 1],
                          c=CELL_TYPE_COLORS[cell_type], s=3, alpha=0.6,
                          rasterized=True)

            # Calculate percentage retained
            pct_retained = (filtered['count'] / total_cells * 100) if total_cells > 0 else 0

            # Annotations
            title = f"≤{threshold}"
            if row_idx == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')

            # Add cell count and percentage
            ax.text(0.02, 0.98, f"n={filtered['count']:,}\n({pct_retained:.1f}%)",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Y-axis label for first column
            if col_idx == 0:
                ax.set_ylabel(cell_type, fontsize=14, fontweight='bold')

            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.suptitle('UMAP Projections at Different Uncertainty Thresholds',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'KNN_uncertainty_threshold_umaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    logger.info(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
