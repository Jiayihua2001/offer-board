"""Rotary position embeddings in NumPy."""
import numpy as np


def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate each pair (x[..., 2i], x[..., 2i+1]) of x (..., T, d) by positions[t] * base ** (-2i / d).

    x'[2i] = x[2i] cos - x[2i+1] sin,  x'[2i+1] = x[2i] sin + x[2i+1] cos.  Returns a new array, same shape as x.
    """
    raise NotImplementedError
