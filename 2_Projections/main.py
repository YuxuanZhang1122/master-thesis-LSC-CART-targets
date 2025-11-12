import scvi
import scanpy as sc
import anndata as ad
import logging
import pickle
from pathlib import Path

from process_query import process_query
from utils import (
    save_embeddings,
    plot_umap,
    plot_umap_with_labels,
    plot_umap_numbered,
    plot_umap_highlight,
    create_output_dirs,
    save_hvg_list,
    fit_umap_model
)

# Configuration
PROJECT_ROOT = Path(__file__).parent
REFERENCE_PATH = PROJECT_ROOT.parent / 'Reference_raw_hvg.h5ad'
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
FIGURES_DIR = OUTPUTS_DIR / "figures"

REFERENCE_BATCH_KEY = "Donor"
REFERENCE_LABELS_KEY = "CellType"
REFERENCE_LABELS_BROAD_KEY = "CellType_Broad"

SCVI_PARAMS = {
    "n_latent": 30,
    "n_layers": 2,
    "n_hidden": 128,
    "dropout_rate": 0.1,
    "dispersion": "gene",
    "gene_likelihood": "nb",
}

SCVI_TRAIN_PARAMS = {
    "max_epochs": 50,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "batch_size": 4096,
    "accelerator": "mps",
    "plan_kwargs": {"lr": 5e-4},
}

SCANVI_TRAIN_PARAMS = {
    "max_epochs": 30,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "batch_size": 4096,
    "accelerator": "mps",
    "plan_kwargs": {"lr": 5e-4},
}

UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.2, "random_state": 42}
REFERENCE_MODEL_NAME = "scanvi_reference"
REFERENCE_EMBEDDING_NAME = "reference_embeddings.h5ad"
HVG_LIST_PATH = OUTPUTS_DIR / "hvg_genes.txt"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_reference_model():
    """Train scVI then scANVI model on reference data."""
    create_output_dirs([MODELS_DIR, EMBEDDINGS_DIR, FIGURES_DIR])
    logger.info("Training reference model")

    adata_ref = ad.read_h5ad(REFERENCE_PATH)
    logger.info(f"Loaded {adata_ref.n_obs} cells, {adata_ref.n_vars} genes")

    if adata_ref.X.min() < 0:
        logger.warning("Data contains negative values - may not be raw counts")

    save_hvg_list(adata_ref.var_names.tolist(), HVG_LIST_PATH)

    # Train scVI
    scvi.model.SCVI.setup_anndata(adata_ref, batch_key=REFERENCE_BATCH_KEY)
    scvi_model = scvi.model.SCVI(adata_ref, **SCVI_PARAMS)
    logger.info("Training scVI...")
    scvi_model.train(**SCVI_TRAIN_PARAMS)

    # Convert to scANVI
    model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        labels_key=REFERENCE_LABELS_KEY,
        unlabeled_category="Unknown",
    )
    logger.info("Training scANVI...")
    model.train(**SCANVI_TRAIN_PARAMS)

    model_path = MODELS_DIR / REFERENCE_MODEL_NAME
    model.save(model_path, overwrite=True)
    logger.info(f"Saved model to {model_path}")

    # Extract and save embeddings
    ref_latent = model.get_latent_representation()
    adata_ref_embed = ad.AnnData(X=ref_latent, obs=adata_ref.obs.copy())

    # Fit UMAP
    umap_model, ref_umap = fit_umap_model(
        adata_ref_embed.X,
        n_neighbors=UMAP_PARAMS['n_neighbors'],
        min_dist=UMAP_PARAMS['min_dist'],
        random_state=UMAP_PARAMS['random_state']
    )
    adata_ref_embed.obsm['X_umap'] = ref_umap

    umap_model_path = MODELS_DIR / 'umap_model.pkl'
    with open(umap_model_path, 'wb') as f:
        pickle.dump(umap_model, f)

    embed_path = EMBEDDINGS_DIR / REFERENCE_EMBEDDING_NAME
    save_embeddings(adata_ref_embed, embed_path)
    logger.info("Training complete")
    return adata_ref_embed


def generate_visualizations(adata_ref_embed):
    """Generate UMAP visualizations for reference data."""
    logger.info("Generating reference visualizations")

    plot_umap_numbered(
        coords=adata_ref_embed.obsm['X_umap'],
        labels=adata_ref_embed.obs[REFERENCE_LABELS_KEY].values,
        title="Reference Atlas - CellType",
        output_path=FIGURES_DIR / "Reference_umap/reference_umap_celltype_fine.png"
    )

    plot_umap_with_labels(
        coords=adata_ref_embed.obsm['X_umap'],
        labels=adata_ref_embed.obs[REFERENCE_LABELS_BROAD_KEY].values,
        title="Reference Atlas - CellType_Broad",
        output_path=FIGURES_DIR / "Reference_umap/reference_umap_celltype_broad.png"
    )

    plot_umap(
        coords=adata_ref_embed.obsm['X_umap'],
        labels=adata_ref_embed.obs[REFERENCE_BATCH_KEY].values,
        title="Reference Atlas - Donor",
        output_path=FIGURES_DIR / "Reference_umap/reference_umap_batch.png",
        show_legend=False
    )

    plot_umap_highlight(
        coords=adata_ref_embed.obsm['X_umap'],
        labels=adata_ref_embed.obs[REFERENCE_LABELS_BROAD_KEY].values,
        highlight_labels=['HSC MPP'],
        title="Reference Atlas - HSC MPP Highlighted",
        output_path=FIGURES_DIR / "Reference_umap/reference_highlight_HSC_MPP.png"
    )


if __name__ == "__main__":
    import sys

    # Check if model exists to determine mode
    model_exists = (MODELS_DIR / REFERENCE_MODEL_NAME).exists()
    embed_exists = (EMBEDDINGS_DIR / REFERENCE_EMBEDDING_NAME).exists()

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # Force training mode
        logger.info("Training new reference model")
        adata_ref_embed = train_reference_model()
        generate_visualizations(adata_ref_embed)
    elif embed_exists:
        # Load existing embeddings
        logger.info("Loading existing embeddings")
        embed_path = EMBEDDINGS_DIR / REFERENCE_EMBEDDING_NAME
        adata_ref_embed = sc.read_h5ad(embed_path)
        generate_visualizations(adata_ref_embed)
    else:
        # No embeddings found, must train
        logger.info("No embeddings found - training required")
        adata_ref_embed = train_reference_model()
        generate_visualizations(adata_ref_embed)

    logger.info("Complete")
