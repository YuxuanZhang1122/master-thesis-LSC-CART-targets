import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_bm_projections(query_embed_path, ref_embed_path, output_path):
    """
    Create multi-panel UMAP projections for BM (normal bone marrow) samples.
    Layout: 3 samples per row (since BM only has HSPC).
    """

    logger.info(f"Loading query embeddings from {query_embed_path}")
    adata_query = sc.read_h5ad(query_embed_path)

    logger.info(f"Loading reference embeddings from {ref_embed_path}")
    adata_ref = sc.read_h5ad(ref_embed_path)

    # Filter for BM samples only
    bm_mask = adata_query.obs['patient_id'].astype(str).str.startswith('BM')
    adata_bm = adata_query[bm_mask].copy()

    logger.info(f"Found {adata_bm.n_obs} BM cells")

    # Get unique BM samples sorted
    bm_samples = sorted(adata_bm.obs['patient_id'].unique())
    n_samples = len(bm_samples)

    logger.info(f"Found {n_samples} BM samples")
    logger.info(f"Samples: {bm_samples}")

    # Get reference UMAP coordinates
    ref_umap = adata_ref.obsm['X_umap']

    # Setup figure: 3 samples per row
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols  # Round up

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Get color palette for cell types
    query_labels = adata_bm.obs['predicted_CellType'].values
    unique_labels = np.unique(query_labels)
    n_labels = len(unique_labels)

    if n_labels <= 10:
        palette = sns.color_palette("tab10", n_labels)
    elif n_labels <= 20:
        palette = sns.color_palette("tab20", n_labels)
    else:
        palette = sns.color_palette("husl", n_labels)

    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Plot each BM sample
    for sample_idx, sample_id in enumerate(bm_samples):
        ax = axes[sample_idx]

        # Plot reference background
        ax.scatter(
            ref_umap[:, 0],
            ref_umap[:, 1],
            s=0.5,
            alpha=0.05,
            c='lightgray',
            rasterized=True
        )

        # Get sample data
        sample_mask = adata_bm.obs['patient_id'] == sample_id
        sample_data = adata_bm[sample_mask]
        n_cells = sample_data.n_obs

        if n_cells > 0:
            # Get UMAP coordinates and labels
            umap_coords = sample_data.obsm['X_umap_combined']
            labels = sample_data.obs['predicted_CellType'].values

            # Plot each cell type with its color
            for label in np.unique(labels):
                mask = labels == label
                ax.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    s=3,
                    alpha=0.6,
                    c=[label_to_color[label]],
                    label=label,
                    rasterized=True
                )

        # Set title and labels
        ax.set_title(f"{sample_id} (n={n_cells})", fontsize=10, fontweight='bold')
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        ax.tick_params(labelsize=7)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide empty subplots if any
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    # Add legend to the side
    handles, labels = [], []
    for label in unique_labels:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=label_to_color[label],
                                 markersize=8, label=label))
        labels.append(label)

    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=10, frameon=False, title="Predicted Cell Type")

    plt.tight_layout(rect=[0, 0, 0.95, 1])

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved multi-panel UMAP projection to {output_path}")


if __name__ == "__main__":
    query_embed_path = "../outputs/embeddings/vanGalen_raw_HLSPC_embeddings.h5ad"
    ref_embed_path = "../outputs/embeddings/reference_embeddings.h5ad"
    output_path = "../outputs/figures/vanGalen_BM_samples_UMAP_projections.png"

    plot_bm_projections(query_embed_path, ref_embed_path, output_path)
