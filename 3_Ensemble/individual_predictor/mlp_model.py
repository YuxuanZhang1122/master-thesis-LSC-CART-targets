from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class MLPModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, ref):
        X = ref.X
        y = ref.obs['CellType_Merged'].values
        y_enc = self.label_encoder.fit_transform(y)

        classes = np.unique(y_enc)
        weights = compute_class_weight('balanced', classes=classes, y=y_enc)
        sample_weights = np.array([dict(zip(classes, weights))[label] for label in y_enc])

        self.model = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, alpha=0.008, early_stopping=True, random_state=42, verbose=False)
        self.model.fit(X, y_enc, sample_weight=sample_weights)

    def predict(self, query):
        X = query.X
        y_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_enc)