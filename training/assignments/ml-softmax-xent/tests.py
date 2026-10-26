import numpy as np

from solution import cross_entropy, cross_entropy_grad, log_softmax, softmax


def test_rows_sum_to_one_and_match_reference():
    z = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]])
    p = softmax(z)
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[0], [0.65900114, 0.24243297, 0.09856589], atol=1e-6)
    assert np.allclose(p[1], [1 / 3] * 3)


def test_softmax_does_not_overflow_or_underflow():
    z = np.array([[1000.0, 0.0, -1000.0], [-1000.0, -1000.0, -1000.0]])
    p = softmax(z)
    assert np.all(np.isfinite(p))
    assert np.allclose(p[0], [1.0, 0.0, 0.0])
    assert np.allclose(p[1], [1 / 3] * 3)


def test_softmax_shift_invariance():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 7)) * 10
    shift = rng.normal(size=(5, 1)) * 100
    assert np.allclose(softmax(z), softmax(z + shift))


def test_log_softmax_is_stable_and_consistent():
    z = np.array([[1000.0, 0.0, -1000.0]])
    ls = log_softmax(z)
    assert np.all(np.isfinite(ls))
    assert np.allclose(ls[0, 0], 0.0)
    assert np.allclose(ls[0, 1], -1000.0)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(4, 6))
    assert np.allclose(np.exp(log_softmax(z)), softmax(z))


def test_cross_entropy_value():
    z = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, -1.0]])
    y = np.array([0, 1])
    expected = -(np.log(0.65900114) + np.log(np.exp(2.5) / (np.exp(0.5) + np.exp(2.5) + np.exp(-1.0)))) / 2
    assert np.isclose(cross_entropy(z, y), expected, atol=1e-6)
    assert np.isclose(cross_entropy(z * 1000, y), 0.0, atol=1e-6) or cross_entropy(z * 1000, y) < 1e-6


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(3, 4))
    y = np.array([1, 3, 0])
    g = cross_entropy_grad(z, y)
    assert g.shape == z.shape
    eps = 1e-5
    num = np.zeros_like(z)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            zp, zm = z.copy(), z.copy()
            zp[i, j] += eps
            zm[i, j] -= eps
            num[i, j] = (cross_entropy(zp, y) - cross_entropy(zm, y)) / (2 * eps)
    assert np.allclose(g, num, atol=1e-6)
    assert np.allclose(g.sum(axis=1), 0.0, atol=1e-9)
