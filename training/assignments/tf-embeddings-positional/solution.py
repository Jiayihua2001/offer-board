import numpy as np


def embed(tokens: np.ndarray, E: np.ndarray) -> np.ndarray:
    return E[tokens]


def sinusoidal_pe(T: int, d: int) -> np.ndarray:
    positions = np.arange(T)[:, None]
    inv_freq = 10000.0 ** (-np.arange(0, d, 2) / d)
    angles = positions * inv_freq[None, :]
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def embed_with_positions(tokens: np.ndarray, E: np.ndarray, P: np.ndarray) -> np.ndarray:
    T = tokens.shape[-1]
    return E[tokens] + P[:T]
