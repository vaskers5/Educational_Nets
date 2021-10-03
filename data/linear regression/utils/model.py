from typing import List, Callable
import numpy as np
import yaml
import pandas as pd


class LinearRegression:
    def __init__(self, metric: Callable):
        self.metric = metric
        self.bias = 0
        self.weights = np.array([])
        self.alpha = 0
        self.len_of_set = 0

    def train(self, features: pd.DataFrame,
              labels: pd.Series,
              bias: float = 0.2,
              alpha: float = 0.01,
              max_epochs: int = 100) -> List[float]:

        self.bias, self.alpha = bias, alpha
        self.len_of_set = len(features)
        self.weights = np.zeros(len(features.columns))
        metrics = []
        for i in range(max_epochs):
            old_weights = self.weights.copy()
            predictions = self.predict(features) + self.bias
            metric = self.metric(predictions, labels)

            if metric == float('inf'):
                self.weights = old_weights
                break
            metrics += [metric]
            self.weights -= 1/self.len_of_set * self.alpha * (features.to_numpy().T.dot(predictions - labels))

        return metrics

    def predict(self, features: pd.DataFrame) -> np.array:
        predict = []
        for sample in features.to_numpy():
            predict += [np.dot(sample.T, self.weights)]
        return np.array(predict)

    def load_weights(self, path_to_weights: str) -> None:
        with open(path_to_weights, 'r') as f:
            last_version = yaml.load(f)
        self.weights = np.array(last_version['weights'])
        self.alpha = last_version['alpha']
        self.bias = last_version['bias']

    def save_weights(self, path_to_weights: str) -> None:
        last_version = {"weights": self.weights.tolist(),
                        "alpha": self.alpha,
                        "bias": self.bias}
        with open(path_to_weights, 'w') as f:
            yaml.dump(last_version, f)
