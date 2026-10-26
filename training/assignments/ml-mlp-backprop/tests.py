import numpy as np

from solution import forward, init_params, loss_and_grads, predict, train

KEYS = ("W1", "b1", "W2", "b2")


def ring_data(n=300, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = ((X ** 2).sum(axis=1) > 1.2).astype(int)
    return X, y


def test_forward_shapes_and_manual_values():
    params = {
        "W1": np.array([[1.0, -1.0], [0.5, 2.0]]),
        "b1": np.array([0.0, -1.0]),
        "W2": np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]),
        "b2": np.array([0.1, 0.2, 0.3]),
    }
    X = np.array([[1.0, 2.0], [-1.0, 0.0]])
    logits, cache = forward(X, params)
    assert logits.shape == (2, 3)
    # row 0: h_pre = [2, 2] -> h = [2, 2] -> logits = [2 + 4 + .1, 2 + .2, -2 + .3]
    # row 1: h_pre = [-1, 0] -> h = [0, 0] -> logits = b2
    assert np.allclose(logits, [[6.1, 2.2, -1.7], [0.1, 0.2, 0.3]])


def test_zero_output_weights_give_log_c_loss():
    params = init_params(3, 5, 4, np.random.default_rng(0))
    params["W2"] = np.zeros_like(params["W2"])
    params["b2"] = np.zeros_like(params["b2"])
    X = np.random.default_rng(1).normal(size=(8, 3))
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])          # balanced over the 4 classes
    loss, grads = loss_and_grads(X, y, params)
    assert np.isclose(loss, np.log(4.0), atol=1e-12)
    assert np.allclose(grads["b2"], 0.0, atol=1e-12), "balanced labels + uniform prediction: db2 = mean(p - onehot) = 0"
    assert set(grads) == set(KEYS)
    for k in KEYS:
        assert grads[k].shape == params[k].shape, f"grad {k} has shape {grads[k].shape}"


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    params = init_params(3, 4, 3, rng)
    X = rng.normal(size=(5, 3))
    y = np.array([0, 2, 1, 2, 0])
    loss, grads = loss_and_grads(X, y, params)
    assert isinstance(float(loss), float) and loss > 0
    eps = 1e-5
    for k in KEYS:
        num = np.zeros_like(params[k])
        for idx in np.ndindex(params[k].shape):
            pp = {kk: v.copy() for kk, v in params.items()}
            pm = {kk: v.copy() for kk, v in params.items()}
            pp[k][idx] += eps
            pm[k][idx] -= eps
            num[idx] = (loss_and_grads(X, y, pp)[0] - loss_and_grads(X, y, pm)[0]) / (2 * eps)
        assert np.allclose(grads[k], num, atol=1e-6), f"gradient of {k} is off: max err {np.abs(grads[k] - num).max()}"


def test_loss_matches_independent_cross_entropy():
    rng = np.random.default_rng(4)
    params = init_params(2, 6, 3, rng)
    X = rng.normal(size=(8, 2))
    y = rng.integers(0, 3, size=8)
    logits, _ = forward(X, params)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ref = -np.mean(np.log(p[np.arange(8), y]))
    assert np.isclose(loss_and_grads(X, y, params)[0], ref, atol=1e-10)


def test_dead_relu_unit_gets_zero_gradient():
    rng = np.random.default_rng(5)
    params = init_params(3, 4, 2, rng)
    params["b1"][1] = -100.0                       # unit 1 is never active
    X = rng.normal(size=(10, 3))
    y = rng.integers(0, 2, size=10)
    _, grads = loss_and_grads(X, y, params)
    assert np.allclose(grads["W1"][:, 1], 0.0) and np.isclose(grads["b1"][1], 0.0)
    assert np.allclose(grads["W2"][1, :], 0.0), "a unit that is always 0 contributes nothing to W2's row"
    assert not np.allclose(grads["W1"][:, 0], 0.0), "live units must still get a gradient"


def test_training_fits_nonlinear_data():
    X, y = ring_data()
    params0 = init_params(2, 16, 2, np.random.default_rng(0))
    snapshot = {k: v.copy() for k, v in params0.items()}
    params, losses = train(X, y, params0, lr=0.5, n_iter=300)
    assert all(np.array_equal(snapshot[k], params0[k]) for k in KEYS), "train must not mutate its input dict"
    losses = np.asarray(losses, dtype=float)
    assert losses.shape == (300,)
    assert losses[-1] < 0.3 * losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    acc = np.mean(predict(X, params) == y)
    assert acc >= 0.95, f"training accuracy {acc:.3f} < 0.95"
    assert predict(X, params).shape == (300,)
