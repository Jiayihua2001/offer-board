"""A CART classifier with Gini impurity. NumPy only; recursion over nodes, vectorized split search."""
import numpy as np


def gini(y: np.ndarray) -> float:
    """1 - sum_c p_c^2 for integer labels y."""
    raise NotImplementedError


def best_split(X: np.ndarray, y: np.ndarray, min_samples_leaf: int = 1):
    """Best (feature, threshold, gain) over all features and midpoints between consecutive distinct values.

    Left child is x[:, feature] <= threshold. Returns None if no valid split with positive gain exists.
    """
    raise NotImplementedError


class DecisionTree:
    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None    # leaf: {"value": label}; node: {"feature", "threshold", "left", "right"}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Labels of shape (N,)."""
        raise NotImplementedError
