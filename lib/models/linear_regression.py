from typing import List
import numpy as np
from pathlib import Path
import pickle
from lib.models.baseModel import BaseModel


class LinearRegression(BaseModel):

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
            predictions = self.predict(features)
            loss = self.mse_loss(predictions, labels)
            if loss == float('inf'):
                self._weights = old_weights
                break
            self._losses += [loss]
            grad = -(labels - predictions).dot(features) / (2.0 * features.shape[0])
            self._weights -= self._lr * grad
        
        return self

    def _predict(self, features: np.ndarray) -> np.ndarray:
        return np.dot(features, self._weights)
    
    def get_losses(self) -> List[float]:
        return self._losses

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.shape[1] + 1 == len(self._weights):
            features = self._process_data(features)
        return self._predict(features)

    @staticmethod
    def mse_loss(predict: np.ndarray,
                 labels: np.ndarray) -> float:
        dif = np.power(predict - labels, 2)
        return dif.sum()/(2 * len(predict))

    @staticmethod
    def _process_data(features: np.ndarray):
        return np.c_[features, np.ones(features.shape[0])]
