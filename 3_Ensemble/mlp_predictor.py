import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from ensemble import BasePredictor
import logging

logger = logging.getLogger(__name__)

class MLPPredictor(BasePredictor):
    """Multi-layer Perceptron (Neural Network) predictor for cell type classification"""

    def __init__(self, n_hvg: int = 3000, hidden_layer_sizes: tuple = (512, 256, 128),
                 activation: str = 'relu', solver: str = 'adam', alpha: float = 0.008,
                 batch_size: str = 'auto', learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001, max_iter: int = 1000,
                 random_state: int = 42, early_stopping: bool = True,
                 validation_fraction: float = 0.1, n_iter_no_change: int = 10):
        super().__init__("MLP")

        self.n_hvg = n_hvg
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hvg_genes = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for MLP: log normalization + scaling"""

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
        """Train MLP model on reference data"""
        logger.info("Training MLP model...")

        # Extract features and labels
        X_train = ref_data.X
        y_train = ref_data.obs[self.cell_type_key].values

        # Encode labels to integers (required by MLP)
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Calculate class weights for handling imbalanced data
        unique_classes = np.unique(y_train_encoded)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=y_train_encoded
        )
        class_weight_dict = dict(zip(unique_classes, class_weights))

        # Create MLP model
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            verbose=False
        )

        # Train the model with class weights (using sample_weight)
        sample_weights = np.array([class_weight_dict[y] for y in y_train_encoded])
        self.model.fit(X_train, y_train_encoded, sample_weight=sample_weights)

        logger.info(f"Final training loss: {self.model.loss_:.6f}")

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using MLP"""

        logger.info("Making MLP predictions...")

        # Extract features
        X_query = query_data.X

        # Make predictions (returns encoded labels)
        predictions_encoded = self.model.predict(X_query)

        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions