"""Token embeddings and positional encodings in NumPy."""
import numpy as np


def embed(tokens: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Rows of E (vocab, d) picked by the integer array `tokens`; output shape tokens.shape + (d,)."""
    raise NotImplementedError


def sinusoidal_pe(T: int, d: int) -> np.ndarray:
    """(T, d) table: PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(pos / 10000^(2i/d)). d is even."""
    raise NotImplementedError


def embed_with_positions(tokens: np.ndarray, E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """E[tokens] + P[:T] for tokens of shape (B, T) and a positional table P of shape (T_max, d)."""
    raise NotImplementedError
