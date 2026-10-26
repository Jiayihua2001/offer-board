"""Scaled dot-product attention in NumPy."""
import numpy as np


def causal_mask(T: int) -> np.ndarray:
    """Boolean (T, T) array where mask[i, j] is True iff j <= i."""
    raise NotImplementedError


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None):
    """Return (output, weights).

    q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v)
    mask: boolean, broadcastable to (..., T_q, T_k); True = may attend.
    """
    raise NotImplementedError
