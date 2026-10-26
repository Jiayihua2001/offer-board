import numpy as np


def _design(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # lstsq solves min ||Xb w - y||^2 via SVD: it never forms X^T X and returns the
    # minimum-norm solution when the columns are linearly dependent.
    w, *_ = np.linalg.lstsq(_design(X), y, rcond=None)
    return w


def fit_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 500):
    Xb = _design(X)
    n = Xb.shape[0]
    w = np.zeros(Xb.shape[1])
    losses = np.empty(n_iter)
    for t in range(n_iter):
        r = Xb @ w - y
        losses[t] = np.mean(r ** 2)
        w = w - lr * (2.0 / n) * (Xb.T @ r)
    return w, losses
