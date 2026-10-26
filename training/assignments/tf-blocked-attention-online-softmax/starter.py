"""Blocked attention with an online softmax, in NumPy. A full-matrix reference is provided for your own checks."""
import numpy as np


def full_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    """Provided reference (materializes the T_q x T_k matrix). Returns (out, lse)."""
    s = q @ k.T / np.sqrt(q.shape[-1])
    m = s.max(axis=1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(axis=1, keepdims=True)
    return (p / l) @ v, (m + np.log(l))[:, 0]


def blocked_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, block_size: int):
    """Attention for one head without ever forming a (T_q, T_k) array.

    q: (T_q, d), k: (T_k, d), v: (T_k, d_v). Keys and values are processed in blocks of `block_size` rows
    with running max, running sum and running output per query. Returns (out (T_q, d_v), lse (T_q,)).
    """
    raise NotImplementedError
