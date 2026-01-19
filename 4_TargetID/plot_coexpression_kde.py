import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Plot gene coexpression KDE')
parser.add_argument('--gene1', type=str, default='CD9', help='First gene name')
parser.add_argument('--gene2', type=str, default='CD47', help='Second gene name')
parser.add_argument('--input', type=str, default='4_TargetID/HSC_MPP_scANVI.h5ad', help='Input h5ad file')
parser.add_argument('--output', type=str, default='4_TargetID/Outputs/Umap', help='Output directory')
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_DIR = args.output
gene1, gene2 = args.gene1, args.gene2

adata = ad.read_h5ad(INPUT_FILE)

for gene in [gene1, gene2]:
    if gene not in adata.var_names:
        raise ValueError(f"{gene} not found in dataset")

raw_gene1 = adata[:, gene1].X.toarray().flatten() if hasattr(adata[:, gene1].X, 'toarray') else adata[:, gene1].X.flatten()
raw_gene2 = adata[:, gene2].X.toarray().flatten() if hasattr(adata[:, gene2].X, 'toarray') else adata[:, gene2].X.flatten()

binary_gene1 = (raw_gene1 > 0).astype(int)
binary_gene2 = (raw_gene2 > 0).astype(int)

umap_coords = adata.obsm['X_umap']
lspc_mask = adata.obs['consensus_label_6votes'] == 'LSPC'
hspc_mask = adata.obs['consensus_label_6votes'] == 'HSPC'
lspc_points = umap_coords[lspc_mask]

def plot_expression(expr_mask, gene_name, output_name):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
               c='lightgrey', s=5, alpha=0.3, rasterized=True)

    expr_points = umap_coords[expr_mask == 1]

    if len(expr_points) > 10:
        kde = gaussian_kde(expr_points.T, bw_method=0.15)
        density_at_points = kde(expr_points.T)

        ax.scatter(expr_points[:, 0], expr_points[:, 1],
                  c=density_at_points, s=7, cmap='plasma',
                  alpha=0.6, edgecolors='face', rasterized=True)
    else:
        ax.scatter(expr_points[:, 0], expr_points[:, 1],
                   c='black', s=20, alpha=0.7)

    if len(lspc_points) > 10:
        padding = 2
        x_min_pad = umap_coords[:, 0].min() - padding
        x_max_pad = umap_coords[:, 0].max() + padding
        y_min_pad = umap_coords[:, 1].min() - padding
        y_max_pad = umap_coords[:, 1].max() + padding

        kde_lspc = gaussian_kde(lspc_points.T, bw_method=0.6)

        xx_lspc, yy_lspc = np.meshgrid(np.linspace(x_min_pad, x_max_pad, 100),
                                        np.linspace(y_min_pad, y_max_pad, 100))

        positions_lspc = np.vstack([xx_lspc.ravel(), yy_lspc.ravel()])
        density_lspc = kde_lspc(positions_lspc).reshape(xx_lspc.shape)

        ax.contour(xx_lspc, yy_lspc, density_lspc,
                   levels=[density_lspc.max() * 0.1],
                   colors='lightgrey', linestyles='--', linewidths=2.5, alpha=0.9)

    lspc_expr = expr_mask[lspc_mask].sum()
    lspc_total = lspc_mask.sum()
    lspc_pct = lspc_expr / lspc_total * 100

    hspc_expr = expr_mask[hspc_mask].sum()
    hspc_total = hspc_mask.sum()
    hspc_pct = hspc_expr / hspc_total * 100

    ax.text(0.7, 0.85, f'LSPC: {lspc_pct:.1f}%', transform=ax.transAxes,
            fontsize=18, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5))
    ax.text(0.7, 0.15, f'HSPC: {hspc_pct:.1f}%', transform=ax.transAxes,
            fontsize=18, va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5))

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{output_name}.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

plot_expression(binary_gene1, gene1, f'{gene1}_expression_kde')
plot_expression(binary_gene2, gene2, f'{gene2}_expression_kde')

coexpression = binary_gene1 & binary_gene2
plot_expression(coexpression, f'{gene1} & {gene2}', f'{gene1}_{gene2}_coexpression_kde')