import numpy as np

from solution import causal_mask, merge_heads, mha, split_heads


def ref_mha(x, Wq, Wk, Wv, Wo, H, mask=None):
    """Independent reference: an explicit loop over heads with column slices."""
    B, T, d = x.shape
    dh = d // H
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    heads = []
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        s = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / np.sqrt(dh)
        if mask is not None:
            s = np.where(mask, s, -np.inf)  # the reference only ever gets a (T, T) mask
        s = s - s.max(-1, keepdims=True)
        w = np.exp(s)
        w = w / w.sum(-1, keepdims=True)
        heads.append(w @ v[:, :, sl])
    return np.concatenate(heads, axis=-1) @ Wo


def weights(rng, d, scale=0.4):
    return [rng.normal(size=(d, d)) * scale for _ in range(4)]


def test_split_heads_layout():
    B, T, d, H = 2, 3, 8, 2
    x = np.arange(B * T * d, dtype=float).reshape(B, T, d)
    s = split_heads(x, H)
    assert s.shape == (B, H, T, d // H), s.shape
    for b in range(B):
        for h in range(H):
            for t in range(T):
                assert np.array_equal(s[b, h, t], x[b, t, h * 4:(h + 1) * 4]), (b, h, t)


def test_merge_heads_inverts_split_heads():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 7, 12))
    for H in (1, 2, 3, 4, 6, 12):
        assert np.array_equal(merge_heads(split_heads(x, H)), x), H
    s = split_heads(x, 3)
    assert np.array_equal(split_heads(merge_heads(s), 3), s)


def test_matches_reference_on_random_input():
    rng = np.random.default_rng(1)
    B, T, d, H = 2, 5, 8, 2
    x = rng.normal(size=(B, T, d))
    Wq, Wk, Wv, Wo = weights(rng, d)
    out = mha(x, Wq, Wk, Wv, Wo, H)
    assert out.shape == (B, T, d)
    assert np.allclose(out, ref_mha(x, Wq, Wk, Wv, Wo, H), atol=1e-10)
    out4 = mha(x, Wq, Wk, Wv, Wo, 4)
    assert np.allclose(out4, ref_mha(x, Wq, Wk, Wv, Wo, 4), atol=1e-10)
    assert not np.allclose(out, out4), "different head counts give different outputs"


def test_single_head_is_plain_attention():
    rng = np.random.default_rng(2)
    B, T, d = 1, 4, 6
    x = rng.normal(size=(B, T, d))
    Wq, Wk, Wv, Wo = weights(rng, d)
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    s = q @ k.transpose(0, 2, 1) / np.sqrt(d)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    assert np.allclose(mha(x, Wq, Wk, Wv, Wo, 1), (w @ v) @ Wo, atol=1e-10)


def test_causal_mask_blocks_the_future():
    rng = np.random.default_rng(3)
    B, T, d, H = 2, 6, 8, 4
    x = rng.normal(size=(B, T, d))
    Wq, Wk, Wv, Wo = weights(rng, d)
    x2 = x.copy()
    x2[:, 4:] = rng.normal(size=(B, 2, d))
    mask = causal_mask(T)
    a, b = mha(x, Wq, Wk, Wv, Wo, H, mask), mha(x2, Wq, Wk, Wv, Wo, H, mask)
    assert np.allclose(a[:, :4], b[:, :4], atol=1e-12), "earlier outputs changed when later tokens changed"
    assert not np.allclose(a[:, 4:], b[:, 4:])
    assert np.allclose(a, ref_mha(x, Wq, Wk, Wv, Wo, H, mask), atol=1e-10)
    assert not np.allclose(mha(x, Wq, Wk, Wv, Wo, H)[:, :4], mha(x2, Wq, Wk, Wv, Wo, H)[:, :4]), "without a mask the prefix must see the future"


def test_mask_shapes_broadcast():
    rng = np.random.default_rng(4)
    B, T, d, H = 2, 5, 8, 2
    x = rng.normal(size=(B, T, d))
    Wq, Wk, Wv, Wo = weights(rng, d)
    m = causal_mask(T)
    a = mha(x, Wq, Wk, Wv, Wo, H, m)
    assert np.allclose(a, mha(x, Wq, Wk, Wv, Wo, H, m[None, None]), atol=1e-12)
    assert np.allclose(a, mha(x, Wq, Wk, Wv, Wo, H, np.broadcast_to(m, (B, H, T, T))), atol=1e-12)


def test_permutation_equivariance_without_mask():
    rng = np.random.default_rng(5)
    B, T, d, H = 1, 7, 8, 2
    x = rng.normal(size=(B, T, d))
    Wq, Wk, Wv, Wo = weights(rng, d)
    perm = rng.permutation(T)
    out = mha(x, Wq, Wk, Wv, Wo, H)
    out_p = mha(x[:, perm], Wq, Wk, Wv, Wo, H)
    assert np.allclose(out_p, out[:, perm], atol=1e-10)
