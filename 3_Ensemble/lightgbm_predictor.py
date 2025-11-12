import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ensemble import BasePredictor
import logging

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError as e:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    logging.warning(f"LightGBM not available: {e}")

logger = logging.getLogger(__name__)

class LightGBMPredictor(BasePredictor):
    """LightGBM predictor for cell type classification"""

    def __init__(self, n_hvg: int = 3000, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.2, num_leaves: int = 20, feature_fraction: float = 0.9,
                 bagging_fraction: float = 0.8, bagging_freq: int = 5, min_child_samples: int = 20,
                 reg_alpha: float = 0.001, reg_lambda: float = 0.0, random_state: int = 42,
                 n_jobs: int = -1, verbose: int = -1):
        super().__init__("LightGBM")

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Please install it with: pip install lightgbm")

        self.n_hvg = n_hvg
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.hvg_genes = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for LightGBM: log normalization + scaling"""

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
        """Train LightGBM model on reference data"""
        logger.info("Training LightGBM model...")

        # Extract features and labels
        X_train = ref_data.X
        y_train = ref_data.obs[self.cell_type_key].values

        # Encode labels to integers (required by LightGBM)
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        # Calculate class weights for handling imbalanced data
        unique_classes, class_counts = np.unique(y_train_encoded, return_counts=True)
        total_samples = len(y_train_encoded)
        class_weights = total_samples / (len(unique_classes) * class_counts)
        sample_weights = np.array([class_weights[y] for y in y_train_encoded])

        # Determine the objective based on number of classes
        n_classes = len(unique_classes)
        if n_classes == 2:
            objective = 'binary'
            metric = 'binary_logloss'
        else:
            objective = 'multiclass'
            metric = 'multi_logloss'

        # Train LightGBM
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            objective=objective,
            metric=metric,
            class_weight='balanced'  # LightGBM built-in class balancing
        )

        # Fit the model
        self.model.fit(
            X_train,
            y_train_encoded,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train_encoded)],
            eval_names=['training'],
            eval_metric=metric,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using LightGBM"""

        logger.info("Making LightGBM predictions...")

        # Extract features
        X_query = query_data.X

        # Make predictions (returns encoded labels)
        predictions_encoded = self.model.predict(X_query)

        # Decode predictions back to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return predictions

