"""k-nearest neighbors classification without Python loops. Vectorized NumPy only."""
import numpy as np


def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of A (n, d) and rows of B (m, d), shape (n, m), never negative."""
    raise NotImplementedError


def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    """Majority label among the k nearest training rows for each test row, shape (m,). Ties go to the smallest label."""
    raise NotImplementedError
