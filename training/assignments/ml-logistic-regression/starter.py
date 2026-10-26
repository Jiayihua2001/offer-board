"""Binary logistic regression with an L2 penalty on the weights (not the bias). Vectorized NumPy only."""
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Elementwise logistic function. Must not overflow for |z| = 1000."""
    raise NotImplementedError


def log_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0) -> float:
    """Mean binary cross-entropy of labels y (0/1, shape (N,)) under logits X @ w + b, plus (l2 / 2) * ||w||^2."""
    raise NotImplementedError


def grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0):
    """Gradient of log_loss: returns (dw, db) with dw of shape (D,) and db a float."""
    raise NotImplementedError


def fit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 1000, l2: float = 0.0):
    """Full-batch gradient descent from w = 0, b = 0. Returns (w, b)."""
    raise NotImplementedError


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """P(y = 1) for each row of X, shape (N,)."""
    raise NotImplementedError
