"""Multi-head attention in NumPy. attention() and causal_mask() are provided; write the three functions below."""
import numpy as np


def causal_mask(T: int) -> np.ndarray:
    """Provided: boolean (T, T) with True where key j <= query i."""
    return np.tril(np.ones((T, T), dtype=bool))


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None):
    """Provided: scaled dot-product attention, broadcasting over leading dims. Returns (output, weights)."""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v, w


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    """(B, T, d_model) -> (B, H, T, d_head). Head h gets columns h*d_head:(h+1)*d_head of every token."""
    raise NotImplementedError


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B, H, T, d_head) -> (B, T, d_model); the inverse of split_heads."""
    raise NotImplementedError


def mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
        n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Multi-head self-attention on x (B, T, d_model). Weights are (d_model, d_model), no biases.

    mask: boolean, (T, T) or broadcastable to (B, H, T, T); True = may attend. Returns (B, T, d_model).
    """
    raise NotImplementedError
