import celltypist

class CellTypistModel:
    def __init__(self):
        self.model = None

    def train(self, ref):
        y = ref.obs['CellType_Merged'].values
        ref.var_names_make_unique()
        self.model = celltypist.train(X=ref, labels=y, use_SGD=False, feature_selection=True, top_genes=300, max_iter=1000, C=0.6, mini_batch=False, balance_cell_type=True, n_jobs=-1, check_expression=True)

    def predict(self, query):
        query.var_names_make_unique()
        result = celltypist.annotate(query, model=self.model, majority_voting=False)
        return result.predicted_labels['predicted_labels'].values