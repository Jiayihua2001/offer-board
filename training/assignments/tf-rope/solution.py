import numpy as np


def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    d = x.shape[-1]
    inv_freq = base ** (-np.arange(0, d, 2) / d)                 # (d/2,)
    angles = np.asarray(positions)[:, None] * inv_freq[None, :]  # (T, d/2)
    cos, sin = np.cos(angles), np.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1).reshape(x.shape)
