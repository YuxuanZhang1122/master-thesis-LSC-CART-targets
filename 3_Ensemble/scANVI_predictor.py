import numpy as np
import scanpy as sc
import scvi
import torch
from ensemble import BasePredictor
import logging

logger = logging.getLogger(__name__)

class ScANVIPredictor(BasePredictor):
    """scANVI predictor for cell type classification"""

    def __init__(self, n_hvg: int = 3000, n_latent: int = 15, n_layers: int = 2, status = 'infer'):
        super().__init__("scANVI")
        self.n_hvg = n_hvg
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.status = status
        self.model = None
        self.hvg_genes = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for scANVI: raw counts"""
        return adata

    def train(self, ref_data):
        """Train scANVI model on reference data"""

        # Setup anndata for scvi
        ref_data.layers["counts"] = ref_data.X.copy()

        # Setup scvi
        scvi.model.SCVI.setup_anndata(
            ref_data,
            layer="counts",
            labels_key=self.cell_type_key
        )

        # Train VAE first
        logger.info("Training scVI...")
        vae = scvi.model.SCVI(
            ref_data,
            n_latent=self.n_latent,
            n_layers=self.n_layers
        )
        vae.train(max_epochs=50, early_stopping=True, accelerator=self.device)

        # Train scANVI
        logger.info("Training scANVI...")
        self.model = scvi.model.SCANVI.from_scvi_model(
            vae,
            labels_key=self.cell_type_key,
            unlabeled_category="Unknown"
        )
        self.model.train(max_epochs=50, early_stopping=True, accelerator=self.device)

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using scANVI"""

        logger.info("Fine-tuning scANVI and making predictions...")

        # Prepare query data
        query_data.layers["counts"] = query_data.X.copy()

        # Save true labels if they present: eval -> evaluation with labels, infer -> no labels
        if self.status == 'eval':
            query_data.obs['true_labels'] = query_data.obs[self.cell_type_key].copy()
        
        # Add dummy labels for query (required by scANVI)
        query_data.obs[self.cell_type_key] = "Unknown"

        # Setup query data
        scvi.model.SCANVI.setup_anndata(
            query_data,
            layer="counts",
            labels_key=self.cell_type_key,
            unlabeled_category="Unknown"
        )

        # Transfer learning: train on query with reference model
        query_model = scvi.model.SCANVI.load_query_data(
            query_data,
            self.model
        )
        query_model.train(max_epochs=50, early_stopping=True, accelerator=self.device)

        # Get predictions
        predictions = query_model.predict()

        # Put it back
        if self.status == 'eval':
            query_data.obs[self.cell_type_key] = query_data.obs['true_labels']

        return predictions
