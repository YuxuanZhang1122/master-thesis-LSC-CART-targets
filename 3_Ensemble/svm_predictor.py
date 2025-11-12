import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ensemble import BasePredictor
import logging

logger = logging.getLogger(__name__)

class SVMPredictor(BasePredictor):
    """Support Vector Machine predictor for cell type classification"""

    def __init__(self, n_hvg: int = 3000, kernel: str = 'rbf', C: float = 0.5,
                 gamma: str = 'scale', random_state: int = 42):
        super().__init__("SVM")
        self.n_hvg = n_hvg
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hvg_genes = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for SVM: log normalization + scaling"""

        adata_processed = adata.copy()

        if is_reference:
            # Log normalization
            sc.pp.normalize_total(adata_processed, target_sum=1e4)
            sc.pp.log1p(adata_processed)

            # Scale the data and fit scaler (critical for SVM)
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

        return adata_processed

    def train(self, ref_data):
        """Train SVM model on reference data"""
        logger.info("Training SVM model...")

        # Extract features and labels
        X_train = ref_data.X
        y_train = ref_data.obs[self.cell_type_key].values

        # Encode labels to integers (required by SVM)
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Calculate class weights for handling imbalanced data
        unique_classes, class_counts = np.unique(y_train_encoded, return_counts=True)
        n_samples = len(y_train_encoded)
        n_classes = len(unique_classes)
        class_weights = n_samples / (n_classes * class_counts)

        # Create class_weight dict for SVM
        class_weight_dict = dict(zip(unique_classes, class_weights))

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state,
            class_weight=class_weight_dict
        )

        # Train the model
        self.model.fit(X_train, y_train_encoded)

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using SVM"""

        logger.info("Making SVM predictions...")

        # Extract features
        X_query = query_data.X

        # Make predictions (returns encoded labels)
        predictions_encoded = self.model.predict(X_query)

        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions