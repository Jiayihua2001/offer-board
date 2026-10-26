"""LayerNorm, GELU and the position-wise feed-forward block in NumPy."""
import numpy as np


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Normalize x (..., d) over its last axis (population variance, eps inside the sqrt), then scale by g and shift by b."""
    raise NotImplementedError


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    raise NotImplementedError


def ffn(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """gelu(x @ W1 + b1) @ W2 + b2, applied to every position of x (..., d). W1: (d, d_ff), W2: (d_ff, d)."""
    raise NotImplementedError
