import scanpy as sc
import anndata as ad
import scvi
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'HSC_MPP.h5ad'
OUTPUT_DIR = 'umap'
SCVI_OUTPUT = 'HSC_MPP_scVI.h5ad'
SCANVI_OUTPUT = 'HSC_MPP_scANVI.h5ad'

sc.set_figure_params(dpi=100, frameon=False, figsize=(8, 8))

adata = ad.read_h5ad(INPUT_FILE)

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    batch_key='Study',
    flavor='seurat_v3',
    subset=False
)
adata_hvg = adata[:, adata.var['highly_variable']].copy()

scvi.model.SCVI.setup_anndata(
    adata_hvg,
    batch_key='Study',
    categorical_covariate_keys=['Donor','time_point'],
    layer=None
)

vae = scvi.model.SCVI(adata_hvg, n_latent=30)
vae.train(max_epochs=50, plan_kwargs={'lr': 0.005}, accelerator='mps')

adata.obsm['X_scVI'] = vae.get_latent_representation()

sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=15)
sc.tl.umap(adata, min_dist=1)

# In case there's a need
adata.write(SCVI_OUTPUT)

del adata.obsm['X_umap']

lvae = scvi.model.SCANVI.from_scvi_model(
    vae,
    unlabeled_category="Unknown",
    labels_key='consensus_label_6votes',
)

lvae.train(max_epochs=50, accelerator='mps')

adata.obsm['X_scANVI'] = lvae.get_latent_representation()

sc.pp.neighbors(adata, use_rep='X_scANVI', n_neighbors=15)
sc.tl.umap(adata, min_dist=1)

adata.write(SCANVI_OUTPUT)
plt.figure(figsize=(10, 10))
fig = sc.pl.umap(adata, color='consensus_label_6votes', frameon=False, show=False, legend_loc='on data', legend_fontoutline=3,
                 return_fig=True)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/UMAP_Status.pdf', bbox_inches='tight')
plt.close(fig)

plt.figure(figsize=(12, 10))
fig = sc.pl.umap(adata, color='Study', frameon=False, show=False,
                 return_fig=True)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/UMAP_Study.pdf', bbox_inches='tight')
plt.close(fig)

plt.figure(figsize=(16, 10))
fig = sc.pl.umap(adata, color='Donor', frameon=False, show=False,
                 return_fig=True)
fig.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/UMAP_Donor.pdf', bbox_inches='tight')
plt.close(fig)

# Feature plot with log-normalized data
adata_viz = adata.copy()
sc.pp.normalize_total(adata_viz, target_sum=1e4)
sc.pp.log1p(adata_viz)

genes = {
    'CD33': 'CD33',
    'CD123': 'IL3RA',
    'CLL-1': 'CLEC12A',
    'CSF1R': 'CSF1R',
    'CD86': 'CD86',
    'CD117': 'KIT',
    'CD44': 'CD44',
    'FLT3': 'FLT3',
    'CD70': 'CD70',

}

gene_list = list(genes.values())

for label, gene in genes.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(adata_viz, color=gene, ax=ax, show=False,
               frameon=False, title=f'{label} ({gene})',
               cmap='Oranges', vmin=0, vmax='p99', add_outline=True, outline_width=(0.015, 0.002), outline_color=('grey', 'white'),
               size=30, alpha=0.4, colorbar_loc=None)

    # Add custom small colorbar at bottom right
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.cm as cm
    cax = inset_axes(ax, width="3%", height="15%", loc='lower right',
                     bbox_to_anchor=(0.02, 0.07, 1.05, 1.05), bbox_transform=ax.transAxes, borderpad=1.5)

    # Get the actual mappable with correct colormap
    norm = plt.Normalize(vmin=0, vmax=np.percentile(adata_viz[:, gene].X.toarray() if hasattr(adata_viz[:, gene].X, 'toarray') else adata_viz[:, gene].X, 99))
    sm = cm.ScalarMappable(cmap='Oranges', norm=norm)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=6)

    # Highlight high expressors with larger dots
    expr = adata_viz[:, gene].X.toarray().flatten() if hasattr(adata_viz[:, gene].X, 'toarray') else adata_viz[:,
                                                                                                     gene].X.flatten()
    high_expr_mask = expr > np.percentile(expr[expr > 0], 50)

    if high_expr_mask.sum() > 0:
        high_coords = adata_viz.obsm['X_umap'][high_expr_mask]
        high_vals = expr[high_expr_mask]
        ax.scatter(high_coords[:, 0], high_coords[:, 1],
                   c=high_vals, cmap='Oranges',
                   s=15, alpha=0.7, edgecolors='grey', linewidths=0.6,
                   vmin=0, vmax=np.percentile(expr, 100))

    # Add smooth dashed contour for malignant cells
    umap_coords = adata_viz.obsm['X_umap']
    mask = adata_viz.obs['consensus_label_6votes'] == 'LSPC'
    points = umap_coords[mask]

    if len(points) > 10:
        # Add padding for contour
        padding = 2
        x_min, x_max = umap_coords[:, 0].min() - padding, umap_coords[:, 0].max() + padding
        y_min, y_max = umap_coords[:, 1].min() - padding, umap_coords[:, 1].max() + padding

        # Create density contour
        kde = gaussian_kde(points.T, bw_method=0.6)

        # Create grid with extended range
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Evaluate KDE
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)

        # Draw contour at 10% of max density
        ax.contour(xx, yy, density, levels=[density.max() * 0.1],
                   colors='grey', linestyles='--', linewidths=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/FeaturePlots/Feature_{label.replace("-", "_")}.png',
                dpi=300, bbox_inches='tight', transparent=True)
    plt.close()