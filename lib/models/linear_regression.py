from typing import List
import numpy as np
from pathlib import Path
import pickle

np.random.seed(0)


class LinearRegression:
    def __init__(self):
        self._bias = 0
        self._weights = None
        self._lr = 0
        self._losses = []
        
    def train(self, features: np.ndarray,
              labels: np.ndarray,
              _bias: float = 0.1,
              lr: float = 0.001,
              max_epochs: int = 100) -> 'LinearRegression':
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

    def load_weights(self, path_to_weights: Path) -> 'LinearRegression':
        with open(path_to_weights, 'rb') as f:
            last_version = pickle.load(f)
        self._weights = np.array(last_version['weights'])
        self._lr = last_version['lr']
        return self

    def save_weights(self, path_to_weights: Path) -> None:
        last_version = {"weights": self._weights.tolist(),
                        "lr": self._lr}
        with open(path_to_weights, 'wb') as f:
            pickle.dump(last_version, f)
            
    def _predict(self, features: np.ndarray) -> np.ndarray:
        return np.dot(features, self._weights)
    
    def get_loss(self) -> List[float]:
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