import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_patient_projections(query_embed_path, ref_embed_path, output_path):
    """
    Create multi-panel UMAP projections for AML patients split by HSPC/LSPC.
    Layout: 2 patients per row, 2 cell types per patient (4 columns per row).
    """

    logger.info(f"Loading query embeddings from {query_embed_path}")
    adata_query = sc.read_h5ad(query_embed_path)

    logger.info(f"Loading reference embeddings from {ref_embed_path}")
    adata_ref = sc.read_h5ad(ref_embed_path)

    # Filter for AML patients only
    aml_mask = adata_query.obs['patient_id'].astype(str).str.startswith('AML')
    adata_aml = adata_query[aml_mask].copy()

    # Filter for D0 timepoint
    d0_mask = adata_aml.obs['time_point'] != 'D0'
    adata_aml = adata_aml[d0_mask].copy()

    logger.info(f"Filtered to {adata_aml.n_obs} cells at timepoint D0")

    # Get unique AML patients sorted
    aml_patients = sorted(adata_aml.obs['patient_id'].unique())
    n_patients = len(aml_patients)

    logger.info(f"Found {n_patients} AML patients at D0")
    logger.info(f"Patients: {aml_patients}")

    # Get reference UMAP coordinates
    ref_umap = adata_ref.obsm['X_umap']

    # Setup figure: 2 patients per row, 2 cell types per patient
    n_cols = 4  # 2 patients × 2 cell types
    n_rows = (n_patients + 1) // 2  # Round up

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Get color palette for cell types
    query_labels = adata_aml.obs['predicted_CellType'].values
    unique_labels = np.unique(query_labels)
    n_labels = len(unique_labels)

    if n_labels <= 10:
        palette = sns.color_palette("tab10", n_labels)
    elif n_labels <= 20:
        palette = sns.color_palette("tab20", n_labels)
    else:
        palette = sns.color_palette("husl", n_labels)

    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    # Plot each patient
    for patient_idx, patient_id in enumerate(aml_patients):
        row = patient_idx // 2
        col_start = (patient_idx % 2) * 2  # 0 or 2

        # Get patient data
        patient_mask = adata_aml.obs['patient_id'] == patient_id
        patient_data = adata_aml[patient_mask]

        # Plot HSPC and LSPC separately
        for cell_type_idx, cell_type in enumerate(['HSPC', 'LSPC']):
            ax = axes[row, col_start + cell_type_idx]

            # Plot reference background
            ax.scatter(
                ref_umap[:, 0],
                ref_umap[:, 1],
                s=0.5,
                alpha=0.05,
                c='lightgray',
                rasterized=True
            )

            # Filter patient data by cell type
            cell_type_mask = patient_data.obs['CellType_Merged'] == cell_type
            cell_type_data = patient_data[cell_type_mask]
            n_cells = cell_type_data.n_obs

            if n_cells > 0:
                # Get UMAP coordinates and labels
                umap_coords = cell_type_data.obsm['X_umap_combined']
                labels = cell_type_data.obs['predicted_CellType'].values

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
            ax.set_title(f"{patient_id} - {cell_type} (n={n_cells})", fontsize=10, fontweight='bold')
            ax.set_xlabel("UMAP 1", fontsize=8)
            ax.set_ylabel("UMAP 2", fontsize=8)
            ax.tick_params(labelsize=7)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Hide empty subplots if odd number of patients
    if n_patients % 2 == 1:
        for col_idx in range(2, 4):
            axes[-1, col_idx].axis('off')

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
    output_path = "../outputs/figures/vanGalen_AML_patients_D30_UMAP_projections.png"

    plot_patient_projections(query_embed_path, ref_embed_path, output_path)
