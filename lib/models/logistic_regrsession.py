import numpy as np
from lib.models.baseModel import BaseModel
from typing import List


class LogisticRegression(BaseModel):

    def train(self,
              features: np.ndarray,
              labels: np.ndarray,
              _bias: float = 0.1,
              lr: float = 0.001,
              max_epochs: int = 100) -> 'LinearRegression':
        np.random.seed(self.seed)
        features = self._process_data(features)
        self._bias, self._lr = _bias, lr
        self._weights = np.random.rand(features.shape[1])
        self._weights[0] = _bias

        for i in range(max_epochs):
            old_weights = self._weights.copy()
            preds = self.predict(features)
            grad = np.dot(preds - labels, features)
            self._weights -= self._lr * grad
        return self

    def get_losses(self) -> List[float]:
        return self._losses

    def _predict(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] + 1 == len(self._weights):
            features = self._process_data(features)
        return self.sigmoid(np.dot(features, self._weights))
    
    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.vectorize(lambda x: x >= threshold)(self._predict(features))

    @staticmethod
    def _process_data(features: np.ndarray):
        return np.c_[features, np.ones(features.shape[0])]
