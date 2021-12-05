from typing import List
import numpy as np
from pathlib import Path
import pickle


class BaseModel:
    def __init__(self, seed: int = 42) -> None:
        self._bias = 0
        self._weights = None
        self._lr = 0
        self._losses = []
        self.seed = seed

    def load_weights(self, path_to_weights: Path) -> 'BaseModel':
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

    def get_losses(self) -> List[float]:
        return self._losses

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))