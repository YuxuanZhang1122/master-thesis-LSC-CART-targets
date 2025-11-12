from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestModel:
    def __init__(self, n_estimators=300, max_depth=20, n_selected_genes=1500, outer_n_estimators=1500, subset_size=2000, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_selected_genes = n_selected_genes
        self.outer_n_estimators = outer_n_estimators
        self.subset_size = subset_size
        self.random_state = random_state
        self.model = None
        self.outer_model = None
        self.selected_gene_indices = None

    def train(self, ref):
        X = ref.X
        y = ref.obs['CellType_Merged'].values

        unique_types, counts = np.unique(y, return_counts=True)
        min_cells = np.min(counts)
        actual_subset_size = min(self.subset_size, min_cells)

        subset_indices = []
        np.random.seed(self.random_state)
        for cell_type in unique_types:
            type_indices = np.where(y == cell_type)[0]
            if len(type_indices) >= actual_subset_size:
                selected = np.random.choice(type_indices, actual_subset_size, replace=False)
            else:
                selected = type_indices
            subset_indices.extend(selected)

        subset_indices = np.array(subset_indices)
        X_subset = X[subset_indices]
        y_subset = y[subset_indices]

        self.outer_model = RandomForestClassifier(n_estimators=self.outer_n_estimators, random_state=self.random_state, n_jobs=-1, class_weight='balanced')
        self.outer_model.fit(X_subset, y_subset)

        importances = self.outer_model.feature_importances_
        self.selected_gene_indices = np.argsort(importances)[::-1][:self.n_selected_genes]

        X_train_selected = X[:, self.selected_gene_indices]
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=-1, class_weight='balanced')
        self.model.fit(X_train_selected, y)

    def predict(self, query):
        X_query_selected = query.X[:, self.selected_gene_indices]
        return self.model.predict(X_query_selected)