import numpy as np
import pandas as pd
import scanpy as sc
from ensemble import BasePredictor
import logging
import celltypist
from celltypist import models

logger = logging.getLogger(__name__)

class CellTypistPredictor(BasePredictor):
    """CellTypist predictor for cell type classification with custom model training"""

    def __init__(self, n_hvg: int = 3000, use_SGD: bool = False, feature_selection: bool = True,
                 max_iter: int = 1000, C: float = 0.6, mini_batch: bool = False,
                 balance_cell_type: bool = True, random_state: int = 42):
        super().__init__("CellTypist")

        self.n_hvg = n_hvg
        self.use_SGD = use_SGD
        self.feature_selection = feature_selection
        self.max_iter = max_iter
        self.C = C
        self.mini_batch = mini_batch
        self.balance_cell_type = balance_cell_type
        self.random_state = random_state
        self.model = None
        self.hvg_genes = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for CellTypist: normalization + log-transformation"""

        adata_processed = adata.copy()
        sc.pp.normalize_total(adata_processed, target_sum=1e4)
        sc.pp.log1p(adata_processed)

        return adata_processed

    def train(self, ref_data):
        """Train custom CellTypist model from scratch on bone marrow reference data"""
        logger.info("Training CellTypist model...")

        # Extract labels
        y_train = ref_data.obs[self.cell_type_key].values
        # Prepare data for training
        ref_for_training = ref_data.copy()
        # Ensure gene names are in the right format
        ref_for_training.var_names_make_unique()

        self.model = celltypist.train(
            X=ref_for_training,
            labels=y_train,
            use_SGD=self.use_SGD,
            feature_selection=self.feature_selection,
            max_iter=self.max_iter,
            C=self.C,
            mini_batch=self.mini_batch,
            balance_cell_type=self.balance_cell_type,
            n_jobs=-1,
            check_expression=True,  # Verify normalization
        )

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using custom trained CellTypist model"""

        logger.info("Making CellTypist predictions with trained model...")

        # Prepare query data
        query_for_prediction = query_data.copy()
        query_for_prediction.var_names_make_unique()

        # Make predictions using custom trained model
        predictions_result = celltypist.annotate(
            query_for_prediction,
            model=self.model,
            majority_voting=False  # We'll do our own majority voting in ensemble
        )

        # Extract predictions
        predictions = predictions_result.predicted_labels['predicted_labels'].values

        return predictions