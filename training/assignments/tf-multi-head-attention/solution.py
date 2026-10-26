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


def split_heads(x: np.ndarray, n_heads: int) -> np.ndarray:
    B, T, d = x.shape
    return x.reshape(B, T, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x: np.ndarray) -> np.ndarray:
    B, H, T, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * dh)


def mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
        n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    q, k, v = (split_heads(x @ W, n_heads) for W in (Wq, Wk, Wv))
    out, _ = attention(q, k, v, mask)
    return merge_heads(out) @ Wo
