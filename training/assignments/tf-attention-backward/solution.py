import numpy as np


def causal_mask(T: int) -> np.ndarray:
    return np.tril(np.ones((T, T), dtype=bool))


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None):
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v, w


def softmax_backward(w: np.ndarray, d_w: np.ndarray) -> np.ndarray:
    return w * (d_w - (w * d_w).sum(axis=-1, keepdims=True))


def attention_backward(q: np.ndarray, k: np.ndarray, v: np.ndarray, w: np.ndarray, d_out: np.ndarray):
    scale = 1.0 / np.sqrt(q.shape[-1])
    d_v = np.swapaxes(w, -1, -2) @ d_out
    d_w = d_out @ np.swapaxes(v, -1, -2)
    d_s = softmax_backward(w, d_w) * scale
    d_q = d_s @ k
    d_k = np.swapaxes(d_s, -1, -2) @ q
    return d_q, d_k, d_v
