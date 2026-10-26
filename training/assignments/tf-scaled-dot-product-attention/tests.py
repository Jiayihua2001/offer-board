import numpy as np

from solution import attention, causal_mask


def test_causal_mask_shape_and_values():
    m = causal_mask(4)
    assert m.shape == (4, 4) and m.dtype == bool
    assert m[0].tolist() == [True, False, False, False]
    assert m[3].tolist() == [True, True, True, True]
    assert m[1, 2] is np.False_ or m[1, 2] == False  # noqa: E712


def test_identity_example():
    I = np.eye(3)
    out, w = attention(I, I, I)
    assert out.shape == (3, 3) and w.shape == (3, 3)
    assert np.allclose(w.sum(axis=-1), 1.0)
    s = np.exp(1 / np.sqrt(3))
    assert np.allclose(np.diag(w), s / (s + 2))
    assert np.allclose(out, w)  # v is the identity, so output == weights


def test_matches_reference_on_random_batched_input():
    rng = np.random.default_rng(0)
    B, H, Tq, Tk, d, dv = 2, 3, 5, 7, 8, 4
    q, k, v = rng.normal(size=(B, H, Tq, d)), rng.normal(size=(B, H, Tk, d)), rng.normal(size=(B, H, Tk, dv))
    out, w = attention(q, k, v)
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    ref_w = np.exp(scores - scores.max(-1, keepdims=True))
    ref_w /= ref_w.sum(-1, keepdims=True)
    assert out.shape == (B, H, Tq, dv)
    assert np.allclose(w, ref_w, atol=1e-8)
    assert np.allclose(out, ref_w @ v, atol=1e-8)


def test_causal_mask_zeroes_future_positions():
    rng = np.random.default_rng(1)
    T, d = 6, 4
    x = rng.normal(size=(T, d))
    out, w = attention(x, x, x, mask=causal_mask(T))
    assert np.allclose(w[0], [1, 0, 0, 0, 0, 0])
    assert np.all(w[np.triu_indices(T, k=1)] == 0)
    assert np.allclose(w.sum(axis=-1), 1.0)
    # The first output only sees the first value row.
    assert np.allclose(out[0], x[0])


def test_mask_broadcasts_over_batch_and_heads():
    rng = np.random.default_rng(2)
    B, H, T, d = 2, 2, 4, 3
    x = rng.normal(size=(B, H, T, d))
    out, w = attention(x, x, x, mask=causal_mask(T))
    assert w.shape == (B, H, T, T)
    assert np.all(w[..., 0, 1:] == 0)


def test_stable_for_large_scores():
    q = np.array([[1000.0, 0.0]])
    k = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    out, w = attention(q, k, v)
    assert np.all(np.isfinite(w))
    assert np.allclose(w, [[1.0, 0.0]])
    assert np.allclose(out, [[1.0, 0.0]])
