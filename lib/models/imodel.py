from typing import List
import numpy as np
from pathlib import Path
import pickle


class IModel:
    def __init__(self, seed: int = 42) -> None:
        raise NotImplemented

    def train(self,
              features: np.ndarray,
              labels: np.ndarray,
              _bias: float = 0.1,
              lr: float = 0.001,
              max_epochs: int = 100) -> 'IModel':
        raise NotImplemented

    def load_weights(self, path_to_weights: Path) -> 'LinearRegression':
        raise NotImplemented

    def save_weights(self, path_to_weights: Path) -> None:
        raise NotImplemented

    def _predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def get_losses(self) -> List[float]:
        raise NotImplemented

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplemented
