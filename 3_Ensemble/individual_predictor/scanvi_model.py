import scvi
import numpy as np

class ScANVIModel:
    def __init__(self, n_latent=10, n_layers=1):
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.model = None

    def train(self, ref):
        ref.layers["counts"] = ref.X.copy()
        scvi.model.SCVI.setup_anndata(ref, layer="counts", labels_key='CellType_Merged')

        vae = scvi.model.SCVI(ref, n_latent=self.n_latent, n_layers=self.n_layers)
        vae.train(max_epochs=50, early_stopping=True, accelerator='mps')

        self.model = scvi.model.SCANVI.from_scvi_model(vae, labels_key='CellType_Merged', unlabeled_category="Unknown")
        self.model.train(max_epochs=50, early_stopping=True, accelerator='mps')

    def predict(self, query):
        query.layers["counts"] = query.X.copy()
        query.obs['true_labels'] = query.obs['CellType_Merged'].copy()
        query.obs['CellType_Merged'] = "Unknown"

        scvi.model.SCANVI.setup_anndata(query, layer="counts", labels_key='CellType_Merged', unlabeled_category="Unknown")
        query_model = scvi.model.SCANVI.load_query_data(query, self.model)
        query_model.train(max_epochs=50, early_stopping=True, accelerator='mps')

        predictions = query_model.predict()
        query.obs['CellType_Merged'] = query.obs['true_labels']
        return predictions