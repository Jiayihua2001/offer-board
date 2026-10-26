"""2-D convolution (cross-correlation, PyTorch convention) for NCHW inputs. Vectorized NumPy only."""
import numpy as np


def out_size(n: int, k: int, stride: int = 1, padding: int = 0) -> int:
    """Length of one output axis for input length n and kernel length k."""
    raise NotImplementedError


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None, stride: int = 1, padding: int = 0) -> np.ndarray:
    """x: (N, C, H, W), w: (F, C, kH, kW), b: (F,) or None. Returns (N, F, H_out, W_out)."""
    raise NotImplementedError
