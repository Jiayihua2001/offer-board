"""Softmax, log-softmax, cross-entropy and its gradient. Vectorized NumPy only."""
import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax of logits with shape (N, C). Must be stable for large |z|."""
    raise NotImplementedError


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of shape (N, C), computed stably."""
    raise NotImplementedError


def cross_entropy(z: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of integer labels y (shape (N,)) under logits z (shape (N, C))."""
    raise NotImplementedError


def cross_entropy_grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """d cross_entropy / d z, shape (N, C)."""
    raise NotImplementedError
