import numpy as np

from solution import block, causal_mask


def make_params(rng, d, d_ff, scale=0.3):
    return {
        "ln1_g": 1 + 0.1 * rng.normal(size=d), "ln1_b": 0.1 * rng.normal(size=d),
        "Wq": rng.normal(size=(d, d)) * scale, "Wk": rng.normal(size=(d, d)) * scale,
        "Wv": rng.normal(size=(d, d)) * scale, "Wo": rng.normal(size=(d, d)) * scale,
        "ln2_g": 1 + 0.1 * rng.normal(size=d), "ln2_b": 0.1 * rng.normal(size=d),
        "W1": rng.normal(size=(d, d_ff)) * scale, "b1": 0.1 * rng.normal(size=d_ff),
        "W2": rng.normal(size=(d_ff, d)) * scale, "b2": 0.1 * rng.normal(size=d),
    }


def ref_ln(x, g, b, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * g + b


def ref_block(x, p, H, mask=None):
    """Independent reference: explicit loops over batch and heads."""
    B, T, d = x.shape
    dh = d // H
    out = np.zeros_like(x)
    for i in range(B):
        a = ref_ln(x[i], p["ln1_g"], p["ln1_b"])
        q, k, v = a @ p["Wq"], a @ p["Wk"], a @ p["Wv"]
        heads = []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            s = q[:, sl] @ k[:, sl].T / np.sqrt(dh)
            if mask is not None:
                s = np.where(mask, s, -np.inf)
            w = np.exp(s - s.max(-1, keepdims=True))
            w /= w.sum(-1, keepdims=True)
            heads.append(w @ v[:, sl])
        hres = x[i] + np.concatenate(heads, -1) @ p["Wo"]
        z = ref_ln(hres, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"]
        z = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
        out[i] = hres + z @ p["W2"] + p["b2"]
    return out


def test_output_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 6, 16))
    p = make_params(rng, 16, 64)
    assert block(x, p, 4, causal_mask(6)).shape == (2, 6, 16)
    assert block(x, p, 2).shape == (2, 6, 16)


def test_zero_projections_leave_only_the_residual():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5, 8))
    p = make_params(rng, 8, 32)
    p["Wo"] = np.zeros((8, 8))
    p["W2"] = np.zeros((32, 8))
    p["b2"] = np.zeros(8)
    assert np.allclose(block(x, p, 2, causal_mask(5)), x, atol=1e-12)
    p["b2"] = np.full(8, 0.5)
    assert np.allclose(block(x, p, 2), x + 0.5, atol=1e-12)


def test_matches_reference():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 7, 16))
    p = make_params(rng, 16, 64)
    for H in (1, 2, 4):
        assert np.allclose(block(x, p, H), ref_block(x, p, H), atol=1e-9), H
        assert np.allclose(block(x, p, H, causal_mask(7)), ref_block(x, p, H, causal_mask(7)), atol=1e-9), H


def test_causal_mask_blocks_the_future():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 8, 16))
    p = make_params(rng, 16, 64)
    x2 = x.copy()
    x2[:, 5:] = rng.normal(size=(1, 3, 16))
    a, b = block(x, p, 4, causal_mask(8)), block(x2, p, 4, causal_mask(8))
    assert np.allclose(a[:, :5], b[:, :5], atol=1e-12), "a later token changed an earlier output"
    assert not np.allclose(a[:, 5:], b[:, 5:])
    assert not np.allclose(block(x, p, 4)[:, :5], block(x2, p, 4)[:, :5]), "without the mask the prefix must change"


def test_pre_norm_not_post_norm():
    # With both LayerNorm gains and biases at zero, each sub-layer sees a zero input:
    # attention outputs 0 and the FFN outputs the constant gelu(b1) @ W2 + b2,
    # so out - x is the same constant vector at every position. A post-norm block would output 0 instead.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 5, 8))
    p = make_params(rng, 8, 32)
    for key in ("ln1_g", "ln1_b", "ln2_g", "ln2_b"):
        p[key] = np.zeros(8)
    out = block(x, p, 2, causal_mask(5))
    delta = out - x
    const = 0.5 * p["b1"] * (1 + np.tanh(np.sqrt(2 / np.pi) * (p["b1"] + 0.044715 * p["b1"] ** 3))) @ p["W2"] + p["b2"]
    assert np.allclose(delta, const, atol=1e-10)


def test_batch_items_are_independent():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 4, 8))
    p = make_params(rng, 8, 32)
    full = block(x, p, 2, causal_mask(4))
    for i in range(3):
        assert np.allclose(block(x[i:i + 1], p, 2, causal_mask(4)), full[i:i + 1], atol=1e-12)
