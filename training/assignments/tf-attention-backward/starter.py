"""Backward pass of scaled dot-product attention. The forward pass is provided; write the two functions below."""
import numpy as np


def causal_mask(T: int) -> np.ndarray:
    """Provided."""
    return np.tril(np.ones((T, T), dtype=bool))


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None):
    """Provided forward pass. Returns (out, w) with w the attention weights (masked entries are exactly 0)."""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v, w


def softmax_backward(w: np.ndarray, d_w: np.ndarray) -> np.ndarray:
    """Gradient with respect to the softmax input, given its output w and d_w = dL/dw (softmax over the last axis)."""
    raise NotImplementedError


def attention_backward(q: np.ndarray, k: np.ndarray, v: np.ndarray, w: np.ndarray, d_out: np.ndarray):
    """Return (d_q, d_k, d_v) for out = w @ v, w = softmax(q @ k^T / sqrt(d_k)).

    q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v), w: (..., T_q, T_k), d_out: (..., T_q, d_v).
    """
    raise NotImplementedError
