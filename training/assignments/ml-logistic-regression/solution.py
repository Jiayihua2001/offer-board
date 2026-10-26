import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))                       # always in (0, 1]: never overflows
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def log_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0) -> float:
    z = X @ w + b
    # -[y log p + (1 - y) log(1 - p)] = softplus(z) - y z, with softplus(z) = log(1 + e^z) = logaddexp(0, z).
    nll = np.mean(np.logaddexp(0.0, z) - y * z)
    return float(nll + 0.5 * l2 * (w @ w))


def grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, l2: float = 0.0):
    r = sigmoid(X @ w + b) - y                   # d(softplus(z) - y z)/dz = sigmoid(z) - y
    dw = X.T @ r / X.shape[0] + l2 * w
    db = float(np.mean(r))
    return dw, db


def fit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 1000, l2: float = 0.0):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(n_iter):
        dw, db = grad(X, y, w, b, l2)
        w = w - lr * dw
        b = b - lr * db
    return w, b


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)
