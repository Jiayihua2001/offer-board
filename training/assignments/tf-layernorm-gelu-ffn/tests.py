import numpy as np

from solution import ffn, gelu, layer_norm


def ref_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def test_layer_norm_reference_values():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = layer_norm(x, np.ones(4), np.zeros(4), eps=0.0)
    assert np.allclose(y, [[-1.34164079, -0.4472136, 0.4472136, 1.34164079]], atol=1e-6)
    g, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(layer_norm(x, g, b, eps=0.0), y * g + b, atol=1e-12)
    # eps goes inside the square root: population variance of [1, 2, 3, 4] is 1.25
    y_eps = layer_norm(x, np.ones(4), np.zeros(4), eps=0.75)
    assert np.allclose(y_eps, (x - 2.5) / np.sqrt(1.25 + 0.75), atol=1e-12)


def test_layer_norm_statistics_and_shapes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 5, 16)) * 3 + 7
    y = layer_norm(x, np.ones(16), np.zeros(16))
    assert y.shape == x.shape
    assert np.allclose(y.mean(-1), 0.0, atol=1e-10)
    assert np.allclose(y.var(-1), 1.0, atol=1e-4)
    # every token is normalized on its own: changing one row leaves the others untouched
    x2 = x.copy()
    x2[1, 3, 0] += 100
    y2 = layer_norm(x2, np.ones(16), np.zeros(16))
    assert np.allclose(y2[0], y[0]) and np.allclose(y2[1, :3], y[1, :3]) and not np.allclose(y2[1, 3], y[1, 3])


def test_layer_norm_is_invariant_to_per_token_affine_input_changes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 8))
    g, b = rng.normal(size=8), rng.normal(size=8)
    y = layer_norm(x, g, b, eps=1e-12)
    assert np.allclose(layer_norm(10 * x + 5, g, b, eps=1e-12), y, atol=1e-6)


def test_layer_norm_of_a_constant_vector_is_finite():
    x = np.array([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]])
    b = np.array([0.1, 0.2, 0.3])
    y = layer_norm(x, np.ones(3), b)
    assert np.all(np.isfinite(y))
    assert np.allclose(y, [b, b], atol=1e-12)


def test_gelu_values():
    x = np.array([-10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0])
    y = gelu(x)
    assert np.allclose(y, ref_gelu(x), atol=1e-6)
    assert np.isclose(y[4], 0.0) and np.isclose(y[6], 0.84119199, atol=1e-6) and np.isclose(y[2], -0.15880801, atol=1e-6)
    assert np.isclose(y[-1], 10.0, atol=1e-6) and np.isclose(y[0], 0.0, atol=1e-6)
    assert y[3] < 0, "GELU is negative for small negative inputs (unlike ReLU)"
    assert gelu(np.zeros((2, 3, 4))).shape == (2, 3, 4)


def test_ffn_matches_reference_and_is_position_wise():
    rng = np.random.default_rng(2)
    B, T, d, d_ff = 2, 3, 8, 32
    x = rng.normal(size=(B, T, d))
    W1, b1 = rng.normal(size=(d, d_ff)) * 0.3, rng.normal(size=d_ff) * 0.1
    W2, b2 = rng.normal(size=(d_ff, d)) * 0.3, rng.normal(size=d) * 0.1
    y = ffn(x, W1, b1, W2, b2)
    assert y.shape == (B, T, d)
    ref = np.stack([np.stack([ref_gelu(x[i, t] @ W1 + b1) @ W2 + b2 for t in range(T)]) for i in range(B)])
    assert np.allclose(y, ref, atol=1e-10)
    # zero weights: the output is just the final bias
    assert np.allclose(ffn(x, np.zeros((d, d_ff)), b1, np.zeros((d_ff, d)), b2), b2)
    # position-wise: permuting tokens permutes outputs
    assert np.allclose(ffn(x[:, ::-1], W1, b1, W2, b2), y[:, ::-1])
