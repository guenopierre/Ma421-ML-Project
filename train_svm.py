import numpy as np
import warnings
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class HandmadeSVM:
    """
    One-vs-all SVM classifier.
    """

    def __init__(self, C=1.0):
        self.C        = C
        self.models   = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        print(f"  Training SVM ({len(self.classes_)} one-vs-all classifiers)...")

        for i, c in enumerate(self.classes_):
            y_bin = np.where(y == c, 1, -1).astype(float)
            model = LinearSVC(C=self.C, random_state=42, max_iter=2000)
            model.fit(X, y_bin)
            self.models.append(model)

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(self.classes_)} classifiers trained")

    def predict_proba(self, X):
        """
        Compute class probabilities via softmax over decision-function scores.
        Returns (n_samples, n_classes) array.
        """
        scores = np.column_stack([m.decision_function(X) for m in self.models])
        exp    = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def train_svm(X_train, y_train, C=15, random_state=42, **kwargs):

    svm = HandmadeSVM(C=C)
    svm.fit(X_train, y_train)
    return svm