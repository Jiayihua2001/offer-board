import numpy as np

from solution import rope


def ref_rope(x, positions, base=10000.0):
    """Independent reference with explicit loops over positions and pairs."""
    out = np.array(x, dtype=float, copy=True)
    d = x.shape[-1]
    for t, pos in enumerate(positions):
        for i in range(d // 2):
            theta = pos * base ** (-2 * i / d)
            a, b = x[..., t, 2 * i], x[..., t, 2 * i + 1]
            out[..., t, 2 * i] = a * np.cos(theta) - b * np.sin(theta)
            out[..., t, 2 * i + 1] = a * np.sin(theta) + b * np.cos(theta)
    return out


def test_worked_example():
    x = np.array([[1.0, 0.0, 1.0, 0.0]])
    y = rope(x, np.array([2]))
    assert np.allclose(y, [[np.cos(2), np.sin(2), np.cos(0.02), np.sin(0.02)]], atol=1e-12)
    assert np.allclose(rope(x, np.array([0])), x)
    assert np.allclose(x, [[1.0, 0.0, 1.0, 0.0]]), "rope must not modify its input"


def test_matches_reference_and_shapes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 7, 8))  # (B, H, T, d)
    positions = np.arange(7)
    y = rope(x, positions)
    assert y.shape == x.shape
    assert np.allclose(y, ref_rope(x, positions), atol=1e-12)
    positions2 = np.array([5, 0, 11, 3, 3, 100, 2])  # arbitrary positions, any order
    assert np.allclose(rope(x, positions2), ref_rope(x, positions2), atol=1e-12)
    assert np.allclose(rope(x, positions, base=500.0), ref_rope(x, positions, base=500.0), atol=1e-12)


def test_norms_are_preserved():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 10, 16))
    y = rope(x, np.arange(10) * 7)
    assert np.allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-12)
    assert not np.allclose(y[:, 1:], x[:, 1:]), "positions above 0 must change the vectors"


def test_dot_products_depend_only_on_the_relative_position():
    rng = np.random.default_rng(2)
    q, k = rng.normal(size=(1, 8)), rng.normal(size=(1, 8))
    for m, n in [(0, 0), (3, 1), (10, 4), (7, 7), (2, 9)]:
        base_score = rope(q, np.array([m]))[0] @ rope(k, np.array([n]))[0]
        for shift in (1, 5, 40, 1000):
            shifted = rope(q, np.array([m + shift]))[0] @ rope(k, np.array([n + shift]))[0]
            assert np.isclose(shifted, base_score, atol=1e-9), (m, n, shift)
    # and it does depend on the offset
    s1 = rope(q, np.array([5]))[0] @ rope(k, np.array([4]))[0]
    s2 = rope(q, np.array([5]))[0] @ rope(k, np.array([2]))[0]
    assert not np.isclose(s1, s2)


def test_attention_scores_are_shift_invariant_for_a_whole_sequence():
    rng = np.random.default_rng(3)
    T, d = 6, 8
    q, k = rng.normal(size=(T, d)), rng.normal(size=(T, d))
    scores = rope(q, np.arange(T)) @ rope(k, np.arange(T)).T
    scores_shifted = rope(q, np.arange(T) + 17) @ rope(k, np.arange(T) + 17).T
    assert np.allclose(scores, scores_shifted, atol=1e-9)
    assert not np.allclose(scores, q @ k.T), "rotation changes the scores (except on the diagonal)"
    assert np.allclose(np.diag(scores), np.diag(q @ k.T), atol=1e-9), "zero offset leaves the dot product unchanged"


def test_first_pair_rotates_one_radian_per_position_and_last_pair_slowly():
    d = 8
    e = np.zeros((1, d))
    e[0, 0] = 1.0
    e[0, d - 2] = 1.0
    for pos in (1, 2, 5):
        y = rope(e, np.array([pos]))[0]
        assert np.isclose(y[0], np.cos(pos)) and np.isclose(y[1], np.sin(pos))
        slow = pos * 10000 ** (-(d - 2) / d)
        assert np.isclose(y[d - 2], np.cos(slow)) and np.isclose(y[d - 1], np.sin(slow))
