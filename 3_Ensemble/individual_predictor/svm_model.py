from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SVMModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def train(self, ref):
        X = ref.X
        y = ref.obs['CellType_Merged'].values
        y_enc = self.label_encoder.fit_transform(y)

        #classes, counts = np.unique(y_enc, return_counts=True)
        #weights = len(y_enc) / (len(classes) * counts)
        #class_weight_dict = dict(zip(classes, weights))

        self.model = SVC(kernel='rbf', C=0.5, gamma='scale', random_state=42, class_weight='balanced')
        self.model.fit(X, y_enc)

    def predict(self, query):
        y_enc = self.model.predict(query.X)
        return self.label_encoder.inverse_transform(y_enc)