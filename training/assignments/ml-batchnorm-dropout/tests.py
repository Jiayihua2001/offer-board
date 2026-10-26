import numpy as np

from solution import batchnorm_forward, dropout_forward


def test_training_output_is_standardized_then_scaled():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(64, 4))
    gamma = np.array([1.0, 2.0, 0.5, -1.0])
    beta = np.array([0.0, 1.0, -2.0, 3.0])
    out, rm, rv = batchnorm_forward(x, gamma, beta, np.zeros(4), np.ones(4), training=True)
    assert out.shape == (64, 4)
    assert np.allclose(out.mean(axis=0), beta, atol=1e-8)
    assert np.allclose(out.std(axis=0), np.abs(gamma), atol=1e-4), "eps makes the std slightly below |gamma|"


def test_running_stats_update_with_momentum_and_unbiased_variance():
    x = np.array([[1.0, 10.0], [3.0, 30.0]])
    rm0, rv0 = np.zeros(2), np.ones(2)
    out, rm, rv = batchnorm_forward(x, np.ones(2), np.zeros(2), rm0, rv0, training=True, momentum=0.1)
    assert np.allclose(out, [[-1.0, -1.0], [1.0, 1.0]], atol=1e-4)
    assert np.allclose(rm, [0.2, 2.0]), f"running mean {rm}"
    assert np.allclose(rv, [1.1, 20.9]), f"running var {rv} (unbiased batch var is 2 and 200)"
    assert np.array_equal(rm0, np.zeros(2)) and np.array_equal(rv0, np.ones(2)), "inputs must not be modified"
    out2, rm2, rv2 = batchnorm_forward(x, np.ones(2), np.zeros(2), rm, rv, training=True, momentum=0.5)
    assert np.allclose(rm2, [1.1, 11.0]) and np.allclose(rv2, [1.55, 110.45])


def test_eval_uses_running_stats_not_the_batch():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=100.0, scale=10.0, size=(5, 3))     # wildly different from the running stats
    gamma, beta = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.0, -0.5])
    rm, rv = np.array([1.0, 2.0, 3.0]), np.array([4.0, 9.0, 16.0])
    out, rm_out, rv_out = batchnorm_forward(x, gamma, beta, rm, rv, training=False, eps=1e-5)
    ref = gamma * (x - rm) / np.sqrt(rv + 1e-5) + beta
    assert np.allclose(out, ref, atol=1e-10)
    assert np.array_equal(rm_out, rm) and np.array_equal(rv_out, rv), "eval must not touch the running stats"
    out1, _, _ = batchnorm_forward(x[:1], gamma, beta, rm, rv, training=False)
    assert np.allclose(out1, ref[:1]), "a batch of one works in eval mode"


def test_running_stats_converge_to_population_values():
    rng = np.random.default_rng(2)
    rm, rv = np.zeros(3), np.ones(3)
    for _ in range(300):
        x = rng.normal(loc=3.0, scale=2.0, size=(32, 3))
        _, rm, rv = batchnorm_forward(x, np.ones(3), np.zeros(3), rm, rv, training=True, momentum=0.1)
    assert np.allclose(rm, 3.0, atol=0.3), f"running mean {rm} should approach 3"
    assert np.allclose(rv, 4.0, atol=0.8), f"running var {rv} should approach 4"


def test_constant_feature_is_finite_and_equals_beta():
    x = np.full((8, 2), 7.0)
    out, _, _ = batchnorm_forward(x, np.array([3.0, 3.0]), np.array([1.0, -1.0]), np.zeros(2), np.ones(2), training=True)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, [[1.0, -1.0]] * 8)


def test_dropout_training_drops_fraction_p_and_rescales():
    rng = np.random.default_rng(3)
    x = np.full((500, 200), 2.0)
    out, mask = dropout_forward(x, 0.3, training=True, rng=rng)
    assert out.shape == x.shape and mask.shape == x.shape
    frac_zero = np.mean(out == 0)
    assert abs(frac_zero - 0.3) < 0.01, f"fraction dropped {frac_zero:.3f} should be about 0.3"
    kept = out[out != 0]
    assert np.allclose(kept, 2.0 / 0.7), "kept values are scaled by 1 / (1 - p)"
    assert np.allclose(out, x * (mask != 0) / 0.7), "mask must mark the kept entries"
    assert abs(out.mean() - 2.0) < 0.02, "inverted dropout preserves the expectation"


def test_dropout_eval_is_identity_and_p_zero_keeps_everything():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 5))
    out, mask = dropout_forward(x, 0.5, training=False, rng=rng)
    assert np.array_equal(out, x) and np.all(mask == 1)
    out, mask = dropout_forward(x, 0.0, training=True, rng=rng)
    assert np.allclose(out, x) and np.all(mask == 1)


def test_dropout_is_deterministic_given_the_rng():
    x = np.ones((20, 20))
    a, _ = dropout_forward(x, 0.5, training=True, rng=np.random.default_rng(7))
    b, _ = dropout_forward(x, 0.5, training=True, rng=np.random.default_rng(7))
    c, _ = dropout_forward(x, 0.5, training=True, rng=np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
