import tracemalloc

import numpy as np

from solution import blocked_attention


def ref_attention(q, k, v):
    s = q @ k.T / np.sqrt(q.shape[-1])
    w = np.exp(s - s.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return w @ v, np.log(np.exp(s).sum(1))


def test_identity_example():
    I = np.eye(3)
    out, lse = blocked_attention(I, I, I, 2)
    assert out.shape == (3, 3) and lse.shape == (3,)
    e = np.exp(1 / np.sqrt(3))
    assert np.allclose(np.diag(out), e / (e + 2), atol=1e-12)
    assert np.allclose(lse, np.log(e + 2), atol=1e-12)
    assert np.allclose(out, ref_attention(I, I, I)[0], atol=1e-12)


def test_matches_full_attention_for_every_block_size():
    rng = np.random.default_rng(0)
    T_q, T_k, d, d_v = 11, 13, 8, 5
    q, k, v = rng.normal(size=(T_q, d)), rng.normal(size=(T_k, d)), rng.normal(size=(T_k, d_v))
    ref_out, ref_lse = ref_attention(q, k, v)
    for b in (1, 2, 3, 5, 8, 13, 16, 100):
        out, lse = blocked_attention(q, k, v, b)
        assert out.shape == (T_q, d_v) and lse.shape == (T_q,), b
        assert np.allclose(out, ref_out, atol=1e-8), f"block_size={b}"
        assert np.allclose(lse, ref_lse, atol=1e-8), f"block_size={b}"


def test_result_does_not_depend_on_the_block_size():
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=(7, 4)), rng.normal(size=(20, 4)), rng.normal(size=(20, 6))
    a = blocked_attention(q, k, v, 4)
    b = blocked_attention(q, k, v, 7)
    assert np.allclose(a[0], b[0], atol=1e-12) and np.allclose(a[1], b[1], atol=1e-12)


def test_stable_for_huge_scores_and_all_negative_scores():
    rng = np.random.default_rng(2)
    q, k, v = rng.normal(size=(5, 4)) * 300, rng.normal(size=(9, 4)) * 300, rng.normal(size=(9, 3))
    out, lse = blocked_attention(q, k, v, 4)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(lse))
    # with scores this large the softmax is one-hot: each output row is one value row
    s = q @ k.T / 2
    assert np.allclose(out, v[s.argmax(1)], atol=1e-8)
    assert np.allclose(lse, s.max(1), atol=1e-8)
    # all scores negative: a running max initialised to 0 would be wrong here
    q2 = -np.abs(rng.normal(size=(4, 3))) * 5
    k2 = np.abs(rng.normal(size=(6, 3))) * 5
    v2 = rng.normal(size=(6, 2))
    out2, lse2 = blocked_attention(q2, k2, v2, 4)
    ref_out2, ref_lse2 = ref_attention(q2, k2, v2)
    assert np.allclose(out2, ref_out2, atol=1e-8) and np.allclose(lse2, ref_lse2, atol=1e-8)


def test_never_exponentiates_the_full_score_matrix():
    rng = np.random.default_rng(3)
    T, d, b = 1024, 16, 64
    q, k, v = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
    sizes = []
    real_exp = np.exp

    def spy(x, *args, **kwargs):
        sizes.append(np.size(x))
        return real_exp(x, *args, **kwargs)

    np.exp = spy
    try:
        out, lse = blocked_attention(q, k, v, b)
    finally:
        np.exp = real_exp
    assert sizes, "use np.exp for the exponentials"
    assert max(sizes) <= 2 * T * b, f"an array of {max(sizes)} elements went through np.exp; the full matrix has {T * T}"
    ref_out, ref_lse = ref_attention(q, k, v)
    assert np.allclose(out, ref_out, atol=1e-8) and np.allclose(lse, ref_lse, atol=1e-8)


def test_peak_memory_is_far_below_one_full_matrix():
    rng = np.random.default_rng(4)
    T, d, b = 2048, 16, 64  # one T x T float64 matrix is 33.5 MB; block-sized arrays are 1 MB each
    q, k, v = rng.normal(size=(T, d)), rng.normal(size=(T, d)), rng.normal(size=(T, d))
    tracemalloc.start()
    ref_attention(q, k, v)
    _, full_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    blocked_attention(q, k, v, b)
    _, blocked_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    one_matrix = T * T * 8
    if full_peak >= one_matrix:  # only judge when the tracer saw the reference allocate its T x T matrix
        assert blocked_peak < one_matrix / 4, f"peak {blocked_peak / 1e6:.1f} MB; a T x T float64 matrix is {one_matrix / 1e6:.1f} MB"
