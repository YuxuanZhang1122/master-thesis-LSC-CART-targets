#!/usr/bin/env python3
"""
Plot uncertainty distributions for three cell types.
Loads all datasets from outputs/PooledLSC_all_cells and creates histograms.
"""

from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cell types and their folders
CELL_TYPES = ['HSC_MPP', 'LMPP', 'Early_GMP']
BASE_DIR = Path('../outputs/PooledLSC_all_cells')
OUTPUT_DIR = Path('../outputs/figures')

def load_uncertainties(cell_type):
    """Load and concatenate uncertainty values for a cell type."""
    cell_dir = BASE_DIR / cell_type
    h5ad_files = sorted(cell_dir.glob('*.h5ad'))

    uncertainties = []
    for f in h5ad_files:
        adata = ad.read_h5ad(f)
        uncertainties.extend(adata.obs['uncertainty_CellType_Broad'].values)

    return np.array(uncertainties)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for all cell types
    data = {ct: load_uncertainties(ct) for ct in CELL_TYPES}

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ax, cell_type, color in zip(axes, CELL_TYPES, colors):
        uncertainties = data[cell_type]

        # Calculate percentage above 0.2 threshold
        pct_above_threshold = (uncertainties > 0.2).sum() / len(uncertainties) * 100

        ax.hist(uncertainties, bins=15, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(0.2, color='red', linestyle='--', linewidth=2, label='Threshold (0.2)')

        # Add annotation for percentage above threshold
        ax.text(0.98, 0.97, f'>{0.2}: {pct_above_threshold:.1f}%',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))

        ax.set_xlabel('Uncertainty Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{cell_type}\n(n={len(uncertainties):,})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'KNN_uncertainty_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    logger.info(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
