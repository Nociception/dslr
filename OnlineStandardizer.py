import numpy as np


class OnlineStandardizer:
    def __init__(self, n_features: int):
        self.sample_size = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)

    def batch_update(self, X: np.ndarray):
        for x in X:
            self.sample_size += 1
            delta = x - self.mean
            self.mean += delta / self.sample_size
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        variance = self.M2 / (self.sample_size - 1)
        std = np.sqrt(variance)
        std[std == 0] = 1.0
        self.std = std

    def standardize(self, X: np.ndarray):
        return (X - self.mean) / self.std
