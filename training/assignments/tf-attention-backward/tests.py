import numpy as np

from solution import attention_backward, causal_mask, softmax_backward


def ref_attention(q, k, v, mask=None):
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -1e9)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(-1, keepdims=True)
    return w @ v, w


def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        g[idx] = (f(xp) - f(xm)) / (2 * eps)
    return g


def check_finite_differences(q, k, v, d_out, mask=None, atol=1e-6):
    _, w = ref_attention(q, k, v, mask)
    d_q, d_k, d_v = attention_backward(q, k, v, w, d_out)
    assert d_q.shape == q.shape and d_k.shape == k.shape and d_v.shape == v.shape

    def loss(qq, kk, vv):
        return float((ref_attention(qq, kk, vv, mask)[0] * d_out).sum())

    assert np.allclose(d_q, numerical_grad(lambda x: loss(x, k, v), q), atol=atol), "d_q"
    assert np.allclose(d_k, numerical_grad(lambda x: loss(q, x, v), k), atol=atol), "d_k"
    assert np.allclose(d_v, numerical_grad(lambda x: loss(q, k, x), v), atol=atol), "d_v"
    return d_q, d_k, d_v


def test_softmax_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(3, 5))
    c = rng.normal(size=(3, 5))

    def softmax(z):
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    d_s = softmax_backward(softmax(s), c)
    assert d_s.shape == s.shape
    assert np.allclose(d_s, numerical_grad(lambda z: float((softmax(z) * c).sum()), s), atol=1e-6)
    assert np.allclose(d_s.sum(-1), 0.0, atol=1e-12), "rows of d_s must sum to zero"


def test_worked_example():
    q = np.array([[1.0, 0.0]])
    k = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, w = ref_attention(q, k, v)
    d_q, d_k, d_v = attention_backward(q, k, v, w, np.array([[1.0, 0.0]]))
    a = np.exp(1 / np.sqrt(2)) / (np.exp(1 / np.sqrt(2)) + 1)  # weight on the first key
    g = a * (1 - a) / np.sqrt(2)
    assert np.allclose(d_v, [[a, 0.0], [1 - a, 0.0]], atol=1e-12)
    assert np.allclose(d_q, [[g, -g]], atol=1e-12)
    assert np.allclose(d_k, [[g, 0.0], [-g, 0.0]], atol=1e-12)


def test_cross_attention_shapes_against_finite_differences():
    rng = np.random.default_rng(1)
    Tq, Tk, d, dv = 3, 4, 2, 3
    q, k, v = rng.normal(size=(Tq, d)), rng.normal(size=(Tk, d)), rng.normal(size=(Tk, dv))
    check_finite_differences(q, k, v, rng.normal(size=(Tq, dv)))


def test_self_attention_against_finite_differences():
    rng = np.random.default_rng(2)
    T, d = 5, 4
    q, k, v = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
    check_finite_differences(q, k, v, rng.normal(size=(T, d)))


def test_causal_mask_against_finite_differences():
    rng = np.random.default_rng(3)
    T, d = 4, 3
    q, k, v = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
    d_out = rng.normal(size=(T, d))
    d_q, d_k, d_v = check_finite_differences(q, k, v, d_out, mask=causal_mask(T))
    # only the last query sees the last key, so d_v[-1] comes from that single row
    _, w = ref_attention(q, k, v, causal_mask(T))
    assert np.allclose(d_v[-1], w[-1, -1] * d_out[-1], atol=1e-12)
    assert np.allclose(d_q[0], 0.0, atol=1e-12), "the first query attends to one key only, so its gradient is zero"


def test_batched_heads_match_per_example_results():
    rng = np.random.default_rng(4)
    B, H, T, d = 2, 3, 4, 5
    q, k, v = (rng.normal(size=(B, H, T, d)) for _ in range(3))
    d_out = rng.normal(size=(B, H, T, d))
    _, w = ref_attention(q, k, v)
    d_q, d_k, d_v = attention_backward(q, k, v, w, d_out)
    assert d_q.shape == d_k.shape == d_v.shape == (B, H, T, d)
    for b in range(B):
        for h in range(H):
            _, wbh = ref_attention(q[b, h], k[b, h], v[b, h])
            a, bb, c = attention_backward(q[b, h], k[b, h], v[b, h], wbh, d_out[b, h])
            assert np.allclose(d_q[b, h], a, atol=1e-12) and np.allclose(d_k[b, h], bb, atol=1e-12)
            assert np.allclose(d_v[b, h], c, atol=1e-12)


def test_dk_sums_to_zero_over_keys():
    rng = np.random.default_rng(5)
    T, d = 6, 4
    q, k, v = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
    _, w = ref_attention(q, k, v)
    _, d_k, _ = attention_backward(q, k, v, w, rng.normal(size=(T, d)))
    assert np.allclose(d_k.sum(axis=0), 0.0, atol=1e-10)


def test_saturated_softmax_kills_the_gradient():
    rng = np.random.default_rng(6)
    T, d = 5, 4
    q, k, v = rng.normal(size=(T, d)) * 60, rng.normal(size=(T, d)) * 60, rng.normal(size=(T, d))
    _, w = ref_attention(q, k, v)
    assert np.allclose(w.max(-1), 1.0), "scores this large make the weights one-hot"
    d_q, d_k, d_v = attention_backward(q, k, v, w, rng.normal(size=(T, d)))
    assert np.abs(d_q).max() < 1e-6 and np.abs(d_k).max() < 1e-6, "one-hot weights give (almost) no gradient to q and k"
    assert np.abs(d_v).max() > 0.1, "v still gets gradient through the hard attention"
