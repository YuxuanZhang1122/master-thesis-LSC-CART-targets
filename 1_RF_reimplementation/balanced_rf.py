"""
Balanced Random Forest Classifier

Custom Random Forest implementation with balanced sampling per class.
Used across all van Galen classifier steps.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class BalancedRandomForest:
    """Random Forest with balanced sampling per class"""

    def __init__(self, n_estimators=1000, n_sample_per_class=50, random_state=42, track_feature_importance=True):
        self.n_estimators = n_estimators
        self.n_sample_per_class = n_sample_per_class
        self.random_state = random_state
        self.track_feature_importance = track_feature_importance
        self.estimators_ = []
        self.classes_ = None
        self.oob_scores_ = []

        if self.track_feature_importance:
            self.feature_importances_all_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        np.random.seed(self.random_state)
        class_indices = {cls: np.where(y == cls)[0] for cls in self.classes_}

        print(f"Training {self.n_estimators} trees...")
        progress_points = [int(self.n_estimators * p) for p in [0.25, 0.5, 0.75, 1.0]]

        for tree_idx in range(self.n_estimators):
            if (tree_idx + 1) in progress_points:
                pct = ((tree_idx + 1) / self.n_estimators) * 100
                print(f"  Progress: {pct:.0f}% ({tree_idx + 1}/{self.n_estimators} trees)")

            # Sample n_sample_per_class cells from each class
            train_indices = []
            for cls in self.classes_:
                available_indices = class_indices[cls]
                n_to_sample = min(self.n_sample_per_class, len(available_indices))
                if n_to_sample > 0:
                    sampled_indices = np.random.choice(available_indices, size=n_to_sample, replace=False)
                    train_indices.extend(sampled_indices)

            train_indices = np.array(train_indices)
            all_indices = set(range(len(y)))
            oob_indices = np.array(list(all_indices - set(train_indices)))

            # Train tree
            rf_tree = RandomForestClassifier(n_estimators=1, random_state=self.random_state + tree_idx,
                                           bootstrap=False, max_features='sqrt')
            rf_tree.fit(X[train_indices], y[train_indices])

            # Calculate OOB score
            if len(oob_indices) > 0:
                oob_pred = rf_tree.predict(X[oob_indices])
                oob_score = accuracy_score(y[oob_indices], oob_pred)
                self.oob_scores_.append(oob_score)

            self.estimators_.append(rf_tree)

            if self.track_feature_importance:
                self.feature_importances_all_.append(rf_tree.feature_importances_)

        if self.track_feature_importance:
            self.feature_importances_ = np.mean(self.feature_importances_all_, axis=0)

        self.oob_score_ = np.mean(self.oob_scores_) if self.oob_scores_ else None
        return self

    def predict_proba(self, X):
        all_proba = [estimator.predict_proba(X) for estimator in self.estimators_]
        return np.mean(all_proba, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)