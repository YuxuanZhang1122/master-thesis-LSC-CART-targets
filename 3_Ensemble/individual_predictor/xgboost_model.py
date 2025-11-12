import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, ref):
        X = ref.X
        y = ref.obs['CellType_Merged'].values
        y_enc = self.label_encoder.fit_transform(y)

        classes, counts = np.unique(y_enc, return_counts=True)
        weights = len(y_enc) / (len(classes) * counts)
        sample_weights = np.array([weights[label] for label in y_enc])

        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.2, alpha=0.001, random_state=42, n_jobs=-1, eval_metric='mlogloss', tree_method='hist')
        self.model.fit(X, y_enc, sample_weight=sample_weights)

    def predict(self, query):
        y_enc = self.model.predict(query.X)
        return self.label_encoder.inverse_transform(y_enc)