import sys
import scvi
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import pickle

from utils import (
    load_data,
    save_embeddings,
    compute_knn_predictions,
    compute_combined_umap,
    plot_combined_umap,
    plot_combined_umap_contour,
    plot_uncertainty,
    create_output_dirs
)

# Configuration
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"

REFERENCE_LABELS_KEY = "CellType"
REFERENCE_LABELS_BROAD_KEY = "CellType_Broad"
QUERY_BATCH_KEY = "study"
QUERY_UNLABELED_CATEGORY = "Unknown"

KNN_PARAMS = {"n_neighbors": 20, "weights": "distance", "metric": "euclidean"}
UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.2, "random_state": 42, "preserve_reference": True}
SURGERY_PARAMS = {
    "freeze_dropout": True,
    "freeze_expression": True,
    "freeze_batchnorm_encoder": True,
    "freeze_batchnorm_decoder": False
}

REFERENCE_MODEL_NAME = "scanvi_reference"
REFERENCE_EMBEDDING_NAME = "reference_embeddings.h5ad"
CONFIDENCE_THRESHOLD = 0.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_query_name(query_path):
    """Extract query name from file path."""
    return Path(query_path).stem


def process_query(query_path, adata_ref_embed):
    """Process a single query file through the complete pipeline."""

    create_output_dirs([PREDICTIONS_DIR, FIGURES_DIR])

    query_name = get_query_name(query_path)
    logger.info(f"Processing query: {query_name}")

    adata_query = load_data(query_path)

    # Load reference model
    model_path = MODELS_DIR / REFERENCE_MODEL_NAME
    ref_path = PROJECT_ROOT.parent / 'Reference_raw_hvg.h5ad'
    ref_adata = ad.read_h5ad(ref_path)
    ref_model = scvi.model.SCANVI.load(model_path, adata=ref_adata)

    # Prepare query anndata for model: padding missing genes with zeros
    scvi.model.SCANVI.prepare_query_anndata(adata_query, ref_model)

    # Setting up query metadata
    adata_query.obs[QUERY_BATCH_KEY] = f"Query_{query_name}"
    adata_query.obs[REFERENCE_LABELS_KEY] = QUERY_UNLABELED_CATEGORY
    if 'patient_id' in adata_query.obs.columns:
        adata_query.obs['Donor'] = adata_query.obs.patient_id.values
    if 'PatientID' in adata_query.obs.columns:
        adata_query.obs['Donor'] = adata_query.obs.PatientID.values
    # Perform surgery to add query as new batch
    query_model = scvi.model.SCANVI.load_query_data(
        adata_query,
        ref_model,
        **SURGERY_PARAMS
    )

    # Train query model
    query_model.train(max_epochs=20, plan_kwargs={"weight_decay": 0.0}, accelerator='mps')

    # Extract query embeddings
    query_latent = query_model.get_latent_representation()

    adata_query_embed = ad.AnnData(
        X=query_latent,
        obs=adata_query.obs.copy(),
    )

    # Perform KNN label transfer
    ref_embed = adata_ref_embed.X
    ref_labels_fine = adata_ref_embed.obs[REFERENCE_LABELS_KEY].values
    ref_labels_broad = adata_ref_embed.obs[REFERENCE_LABELS_BROAD_KEY].values

    pred_fine, prob_fine, unc_fine = compute_knn_predictions(
        ref_embed, ref_labels_fine, query_latent, **KNN_PARAMS
    )

    pred_broad, prob_broad, unc_broad = compute_knn_predictions(
        ref_embed, ref_labels_broad, query_latent, **KNN_PARAMS
    )

    adata_query_embed.obs['predicted_CellType'] = pred_fine
    adata_query_embed.obs['uncertainty_CellType'] = unc_fine
    adata_query_embed.obs['predicted_CellType_Broad'] = pred_broad
    adata_query_embed.obs['uncertainty_CellType_Broad'] = unc_broad

    query_embed_path = EMBEDDINGS_DIR / f"{query_name}_embeddings.h5ad"
    save_embeddings(adata_query_embed, query_embed_path)

    # Save predictions
    predictions_df = pd.DataFrame({
        'cell_id': adata_query.obs_names,
        'predicted_CellType': pred_fine,
        'uncertainty_CellType': unc_fine,
        'predicted_CellType_Broad': pred_broad,
        'uncertainty_CellType_Broad': unc_broad,
    })

    pred_path = PREDICTIONS_DIR / f"{query_name}_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    ref_umap = adata_ref_embed.obsm['X_umap']

    umap_model_path = MODELS_DIR / 'umap_model.pkl'
    with open(umap_model_path, 'rb') as f:
        umap_model = pickle.load(f)

    query_umap = compute_combined_umap(
        ref_embed,
        query_latent,
        umap_model=umap_model,
        **UMAP_PARAMS
    )

    adata_query_embed.obsm['X_umap_combined'] = query_umap
    save_embeddings(adata_query_embed, query_embed_path)

    # Filter confident cells for visualization
    confident_mask = unc_broad <= CONFIDENCE_THRESHOLD
    n_confident = confident_mask.sum()
    n_total = len(unc_broad)
    pct_confident = n_confident / n_total * 100
    logger.info(f"Confident cells: {n_confident}/{n_total} ({pct_confident:.1f}%)")

    query_umap_confident = query_umap[confident_mask]
    pred_fine_confident = pred_fine[confident_mask]
    pred_broad_confident = pred_broad[confident_mask]

    FIGURES_DIR_QUERY = FIGURES_DIR / query_name
    FIGURES_DIR_QUERY.mkdir(parents=True, exist_ok=True)

    plot_combined_umap(
        ref_coords=ref_umap,
        query_coords=query_umap_confident,
        ref_labels=ref_labels_fine,
        query_labels=pred_fine_confident,
        title=f"{query_name} - CellType_Fine (Confident Cells)",
        output_path=FIGURES_DIR_QUERY / "celltype_fine.png"
    )

    plot_combined_umap(
        ref_coords=ref_umap,
        query_coords=query_umap_confident,
        ref_labels=ref_labels_broad,
        query_labels=pred_broad_confident,
        title=f"{query_name} - CellType_Broad (Confident Cells)",
        output_path=FIGURES_DIR_QUERY / "celltype_broad.png"
    )

    plot_combined_umap_contour(
        ref_coords=ref_umap,
        query_coords=query_umap,
        ref_labels=ref_labels_broad,
        query_labels=pred_broad,
        title=f"{query_name} - CellType_Broad (Contour)",
        output_path=FIGURES_DIR_QUERY / "celltype_broad_contour.png"
    )

    plot_uncertainty(
        coords=query_umap,
        uncertainty=unc_fine,
        title=f"{query_name} - Uncertainty (CellType)",
        output_path=FIGURES_DIR_QUERY / "uncertainty_celltype_fine.png",
        ref_coords=ref_umap
    )

    plot_uncertainty(
        coords=query_umap,
        uncertainty=unc_broad,
        title=f"{query_name} - Uncertainty (CellType_Broad)",
        output_path=FIGURES_DIR_QUERY / "uncertainty_celltype_broad.png",
        ref_coords=ref_umap
    )

    logger.info(f"Completed: {query_name} | {n_total} cells | {len(np.unique(pred_broad))} types")


if __name__ == "__main__":

    query_path = ''

    embed_path = EMBEDDINGS_DIR / REFERENCE_EMBEDDING_NAME
    adata_ref_embed = ad.read_h5ad(embed_path)

    process_query(query_path, adata_ref_embed)
