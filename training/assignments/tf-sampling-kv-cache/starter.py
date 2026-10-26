"""Sampling and KV-cached decoding for a NumPy GPT. The model is provided; write the three functions at the end."""
import numpy as np


def causal_mask(T: int) -> np.ndarray:
    return np.tril(np.ones((T, T), dtype=bool))


def attention(q, k, v, mask=None):
    """q: (..., T_q, d), k, v: (..., T_k, d); mask broadcastable to (..., T_q, T_k), True = may attend."""
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v, w


def split_heads(x, n_heads):
    B, T, d = x.shape
    return x.reshape(B, T, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x):
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


def block(x, p, n_heads, mask=None):
    h = x + mha(layer_norm(x, p["ln1_g"], p["ln1_b"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask)
    return h + ffn(layer_norm(h, p["ln2_g"], p["ln2_b"]), p["W1"], p["b1"], p["W2"], p["b2"])


def init_params(vocab, d_model, n_heads, n_layers, T, rng, scale=0.02):
    d_ff = 4 * d_model

    def blk():
        return {"ln1_g": np.ones(d_model), "ln1_b": np.zeros(d_model),
                "Wq": rng.normal(scale=scale, size=(d_model, d_model)), "Wk": rng.normal(scale=scale, size=(d_model, d_model)),
                "Wv": rng.normal(scale=scale, size=(d_model, d_model)), "Wo": rng.normal(scale=scale, size=(d_model, d_model)),
                "ln2_g": np.ones(d_model), "ln2_b": np.zeros(d_model),
                "W1": rng.normal(scale=scale, size=(d_model, d_ff)), "b1": np.zeros(d_ff),
                "W2": rng.normal(scale=scale, size=(d_ff, d_model)), "b2": np.zeros(d_model)}

    return {"wte": rng.normal(scale=scale, size=(vocab, d_model)), "wpe": rng.normal(scale=scale, size=(T, d_model)),
            "blocks": [blk() for _ in range(n_layers)], "lnf_g": np.ones(d_model), "lnf_b": np.zeros(d_model),
            "n_heads": n_heads}


def gpt_forward(tokens, params):
    """Full forward pass without a cache: logits (B, T, vocab) for tokens (B, T)."""
    T = tokens.shape[-1]
    x = params["wte"][tokens] + params["wpe"][:T]
    mask = causal_mask(T)
    for p in params["blocks"]:
        x = block(x, p, params["n_heads"], mask)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["wte"].T


def sample_next(logits: np.ndarray, rng: np.random.Generator | None, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None) -> int:
    """One token id from logits (vocab,). temperature 0 = argmax; else temperature, then top-k, then top-p, then draw."""
    raise NotImplementedError


def forward_with_cache(tokens: np.ndarray, params: dict, cache=None):
    """Logits (len(tokens), vocab) for the NEW tokens only, plus the updated cache (None = empty cache)."""
    raise NotImplementedError


def generate(tokens, params: dict, n_new: int, rng: np.random.Generator | None = None, use_cache: bool = True,
             temperature: float = 0.0, top_k: int | None = None, top_p: float | None = None) -> list[int]:
    """The prompt followed by n_new sampled tokens. The cached and uncached paths must agree."""
    raise NotImplementedError
