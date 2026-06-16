import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Sigmoid probability calibrator with a predict() API like IsotonicRegression."""

    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", random_state=42)

    def fit(self, probabilities, y):
        x = np.asarray(probabilities, dtype=np.float64).reshape(-1, 1)
        self.model.fit(x, y)
        return self

    def predict(self, probabilities):
        x = np.asarray(probabilities, dtype=np.float64).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]


class IdentityCalibrator:
    """No-op calibrator used when ranking preservation is more important."""

    def fit(self, probabilities, y):
        return self

    def predict(self, probabilities):
        return np.asarray(probabilities, dtype=np.float64)
