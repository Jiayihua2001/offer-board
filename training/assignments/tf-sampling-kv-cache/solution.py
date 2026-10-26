import numpy as np


def causal_mask(T: int) -> np.ndarray:
    return np.tril(np.ones((T, T), dtype=bool))


def attention(q, k, v, mask=None):
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
    T = tokens.shape[-1]
    x = params["wte"][tokens] + params["wpe"][:T]
    mask = causal_mask(T)
    for p in params["blocks"]:
        x = block(x, p, params["n_heads"], mask)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["wte"].T


def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None) -> int:
    logits = np.asarray(logits, dtype=float)
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits / temperature
    if top_k is not None and top_k < z.size:
        kth = np.sort(z)[-top_k]
        z = np.where(z >= kth, z, -np.inf)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p is not None:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, top_p) + 1  # smallest prefix with cumulative mass >= top_p
        keep = np.zeros(p.size, dtype=bool)
        keep[order[:cutoff]] = True
        p = np.where(keep, p, 0.0)
        p /= p.sum()
    return int(rng.choice(p.size, p=p))


def block_cached(x, p, n_heads, kv):
    """x: (t_new, d) for the new tokens; kv: None or (K, V) each (H, T_past, d_head). Returns (out, (K, V))."""
    a = layer_norm(x, p["ln1_g"], p["ln1_b"])
    q, k, v = (split_heads((a @ W)[None], n_heads)[0] for W in (p["Wq"], p["Wk"], p["Wv"]))  # (H, t_new, d_head)
    if kv is not None:
        k = np.concatenate([kv[0], k], axis=1)
        v = np.concatenate([kv[1], v], axis=1)
    t_new, T = q.shape[1], k.shape[1]
    mask = np.arange(T)[None, :] <= (T - t_new + np.arange(t_new))[:, None]  # key j visible to new query i iff j <= T_past + i
    out, _ = attention(q, k, v, mask)
    h = x + merge_heads(out[None])[0] @ p["Wo"]
    return h + ffn(layer_norm(h, p["ln2_g"], p["ln2_b"]), p["W1"], p["b1"], p["W2"], p["b2"]), (k, v)


def forward_with_cache(tokens, params, cache=None):
    tokens = np.asarray(tokens)
    T_past = 0 if cache is None else cache[0][0].shape[1]
    positions = T_past + np.arange(tokens.shape[0])
    x = params["wte"][tokens] + params["wpe"][positions]
    new_cache = []
    for i, p in enumerate(params["blocks"]):
        x, kv = block_cached(x, p, params["n_heads"], None if cache is None else cache[i])
        new_cache.append(kv)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["wte"].T, new_cache


def generate(tokens, params, n_new, rng=None, use_cache=True, temperature=0.0, top_k=None, top_p=None):
    out = [int(t) for t in tokens]
    if use_cache:
        logits, cache = forward_with_cache(np.array(out), params)
        for i in range(n_new):
            out.append(sample_next(logits[-1], rng, temperature, top_k, top_p))
            if i < n_new - 1:
                logits, cache = forward_with_cache(np.array(out[-1:]), params, cache)
    else:
        for _ in range(n_new):
            logits = gpt_forward(np.array([out]), params)[0]
            out.append(sample_next(logits[-1], rng, temperature, top_k, top_p))
    return out
