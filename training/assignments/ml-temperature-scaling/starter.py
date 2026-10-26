"""Temperature scaling (fit one scalar on held-out logits) and expected calibration error. Vectorized NumPy only."""
import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise, stable. Provided."""
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise, stable. Provided."""
    return np.exp(log_softmax(z))


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    """Mean negative log-likelihood of labels y (N,) under softmax(logits / T)."""
    raise NotImplementedError


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0, tol: float = 1e-4) -> float:
    """The T in [lo, hi] minimizing nll(logits, y, T), to within about tol."""
    raise NotImplementedError


def calibration_bins(probs: np.ndarray, y: np.ndarray, n_bins: int):
    """Per equal-width confidence bin on (0, 1]: (mean accuracy, mean confidence, count), each of shape (n_bins,)."""
    raise NotImplementedError


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: sum_b (count_b / N) * |acc_b - conf_b|."""
    raise NotImplementedError
