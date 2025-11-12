import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ensemble import BasePredictor
import logging

import xgboost as xgb

logger = logging.getLogger(__name__)

class XGBoostPredictor(BasePredictor):
    """XGBoost predictor for cell type classification"""

    def __init__(self, n_hvg: int = 3000, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.2, alpha=0.001, random_state: int = 42):
        super().__init__("XGBoost")

        self.n_hvg = n_hvg
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.alpha = alpha
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hvg_genes = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for XGBoost: log normalization + scaling"""

        adata_processed = adata.copy()

        if is_reference:
            # Log normalization
            sc.pp.normalize_total(adata_processed, target_sum=1e4)
            sc.pp.log1p(adata_processed)

            # Scale the data and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(adata_processed.X.toarray() if hasattr(adata_processed.X, 'toarray') else adata_processed.X)
            adata_processed.X = X_scaled

        else:
            # For query: use the fitted scaler from reference
            # Log normalization
            sc.pp.normalize_total(adata_processed, target_sum=1e4)
            sc.pp.log1p(adata_processed)

            # Scale using fitted scaler
            X_scaled = self.scaler.transform(adata_processed.X.toarray() if hasattr(adata_processed.X, 'toarray') else adata_processed.X)
            adata_processed.X = X_scaled

        return adata

    def train(self, ref_data):
        """Train XGBoost model on reference data"""

        logger.info("Training XGBoost model...")

        # Extract features and labels
        X_train = ref_data.X
        y_train = ref_data.obs[self.cell_type_key].values

        # Encode labels to integers (required by XGBoost)
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Calculate class weights for handling imbalanced data
        unique_classes, class_counts = np.unique(y_train_encoded, return_counts=True)
        total_samples = len(y_train_encoded)
        class_weights = total_samples / (len(unique_classes) * class_counts)
        sample_weights = np.array([class_weights[y] for y in y_train_encoded])

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            alpha=self.alpha,
            n_jobs=-1,  # Use all available cores
            eval_metric='mlogloss',  # For multiclass classification
            tree_method='hist'  # Faster training
        )

        self.model.fit(X_train, y_train_encoded, sample_weight=sample_weights)

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using XGBoost"""

        logger.info("Making XGBoost predictions...")

        # Extract features
        X_query = query_data.X

        # Make predictions (returns encoded labels)
        predictions_encoded = self.model.predict(X_query)

        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions
