"""Linear regression two ways: the closed form and full-batch gradient descent. Vectorized NumPy only."""
import numpy as np


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predictions of shape (N,). w has shape (D + 1,) with the intercept first: w[0] + X @ w[1:]."""
    raise NotImplementedError


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares weights of shape (D + 1,) for X (N, D) and y (N,). Must not crash on collinear features."""
    raise NotImplementedError


def fit_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 500):
    """Full-batch gradient descent on mean((pred - y)^2) from w = 0.

    Returns (w, losses): w of shape (D + 1,) and the MSE recorded once per iteration (length n_iter).
    """
    raise NotImplementedError
