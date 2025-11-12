import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ensemble import BasePredictor
import logging

logger = logging.getLogger(__name__)

class RandomForestPredictor(BasePredictor):
    """Random Forest predictor for cell type classification"""

    def __init__(self, n_estimators: int = 300, max_depth: int = 20,
                 n_selected_genes: int = 1500, outer_n_estimators: int = 1500,
                 subset_size: int = 1000, random_state: int = 42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_selected_genes = n_selected_genes
        self.outer_n_estimators = outer_n_estimators
        self.subset_size = subset_size
        self.random_state = random_state
        self.model = None  # Inner classifier
        self.outer_model = None  # Outer classifier for feature selection
        self.scaler = None
        self.selected_genes = None  # Top informative genes from outer classifier
        self.selected_gene_indices = None
        self.cell_type_key = 'CellType_Merged'

    def preprocess_data(self, adata, is_reference: bool = True):
        """Preprocess data for Random Forest: log normalization + scaling"""

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

        return adata_processed

    def _train_outer_classifier(self, ref_data):
        """Train outer classifier for feature selection using random subsets"""
        logger.info("Training outer classifier for feature selection...")

        X_train = ref_data.X
        y_train = ref_data.obs[self.cell_type_key].values

        # Get unique cell types and their counts
        unique_cell_types, counts = np.unique(y_train, return_counts=True)
        min_cells = np.min(counts)

        # Use the minimum count or specified subset_size, whichever is smaller
        actual_subset_size = min(self.subset_size, min_cells)

        # Create balanced subsets for training
        subset_indices = []
        np.random.seed(self.random_state)

        for cell_type in unique_cell_types:
            cell_type_indices = np.where(y_train == cell_type)[0]
            if len(cell_type_indices) >= actual_subset_size:
                # Randomly sample actual_subset_size cells
                selected = np.random.choice(cell_type_indices, actual_subset_size, replace=False)
            else:
                # Use all available cells if fewer than actual_subset_size
                selected = cell_type_indices
            subset_indices.extend(selected)

        subset_indices = np.array(subset_indices)
        X_subset = X_train[subset_indices]
        y_subset = y_train[subset_indices]

        # Train outer classifier
        self.outer_model = RandomForestClassifier(
            n_estimators=self.outer_n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.outer_model.fit(X_subset, y_subset)

    def _select_informative_genes(self, ref_data):
        """Select top informative genes based on outer classifier feature importance"""

        logger.info(f"Selecting top {self.n_selected_genes} informative genes...")

        # Get feature importances from outer classifier
        importances = self.outer_model.feature_importances_

        # Get indices of top informative genes
        top_indices = np.argsort(importances)[::-1][:self.n_selected_genes]
        self.selected_gene_indices = top_indices

        # Store selected gene names using current dataset gene names
        gene_names = ref_data.var_names
        self.selected_genes = [gene_names[i] for i in top_indices if i < len(gene_names)]

        return top_indices

    def train(self, ref_data):
        """Train Random Forest model using two-stage approach"""
        logger.info("Training Random Forest model with two-stage approach...")

        # Stage 1: Train outer classifier for feature selection
        self._train_outer_classifier(ref_data)

        # Stage 2: Select informative genes
        selected_indices = self._select_informative_genes(ref_data)

        # Stage 3: Train inner classifier on selected genes
        logger.info("Training inner classifier on selected genes...")

        # Extract features and labels, using only selected genes
        X_train = ref_data.X[:, selected_indices]
        y_train = ref_data.obs[self.cell_type_key].values

        # Train inner Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.model.fit(X_train, y_train)

    def predict(self, query_data) -> np.ndarray:
        """Predict cell types for query data using inner Random Forest classifier"""

        logger.info("Making Random Forest predictions using selected genes...")

        # Extract features using only selected genes
        X_query = query_data.X[:, self.selected_gene_indices]

        # Make predictions
        predictions = self.model.predict(X_query)

        return predictions