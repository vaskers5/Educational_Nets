from typing import List, Callable
import numpy as np
import yaml
import pandas as pd


class LinearRegression:
    def __init__(self):
        self.bias = 0
        self.weights = np.array([])
        self.lr = 0
        self.losses = []
        
    def train(self, features: np.ndarray,
              labels: np.ndarray,
              bias: float = 0.1,
              lr: float = 0.001,
              max_epochs: int = 100) -> 'LinearRegression':

        self.bias, self.lr = bias, lr
        self.weights = np.random.rand(features.shape[1])
        self.weights[0] = bias
        
        for i in range(max_epochs):
            old_weights = self.weights.copy()
            predictions = self.predict(features)
            loss = self.mse_loss(predictions, labels)
            if loss == float('inf'):
                self.weights = old_weights
                break
            self.losses += [loss]
            grad = -(labels - predictions).dot(features) / (2.0 * features.shape[0])
            self.weights -= self.lr * grad
        
        return self

    def load_weights(self, path_to_weights: str) -> None:
        with open(path_to_weights, 'r') as f:
            last_version = yaml.load(f)
        self.weights = np.array(last_version['weights'])
        self.lr = last_version['lr']

    def save_weights(self, path_to_weights: str) -> None:
        last_version = {"weights": self.weights.tolist(),
                        "lr": self.lr}
        with open(path_to_weights, 'w') as f:
            yaml.dump(last_version, f)
            
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.dot(features, self.weights)
    
    @staticmethod
    def mse_loss(predict: np.array, labels: np.array) -> float:
        dif = np.power(predict - labels, 2)
        return dif.sum()/(2 * len(predict))

