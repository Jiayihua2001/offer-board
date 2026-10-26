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


def mha(x, Wq, Wk, Wv, Wo, n_heads, mask=None):
    q, k, v = (split_heads(x @ W, n_heads) for W in (Wq, Wk, Wv))
    out, _ = attention(q, k, v, mask)
    return merge_heads(out) @ Wo


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def ffn(x, W1, b1, W2, b2):
    return gelu(x @ W1 + b1) @ W2 + b2


def block(x: np.ndarray, p: dict, n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    h = x + mha(layer_norm(x, p["ln1_g"], p["ln1_b"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask)
    return h + ffn(layer_norm(h, p["ln2_g"], p["ln2_b"]), p["W1"], p["b1"], p["W2"], p["b2"])
