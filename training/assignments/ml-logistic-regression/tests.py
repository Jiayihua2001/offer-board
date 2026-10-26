import numpy as np

from solution import fit, grad, log_loss, predict_proba, sigmoid


def make_data(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    z = X @ np.array([1.0, -1.5]) + 0.3
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    return X, y


def test_sigmoid_values_and_stability():
    assert np.isclose(sigmoid(np.array(0.0)), 0.5)
    assert np.allclose(sigmoid(np.array([2.0, -2.0])), [0.88079708, 0.11920292], atol=1e-7)
    with np.errstate(over="raise", invalid="raise"):
        p = sigmoid(np.array([1000.0, -1000.0, 0.0]))
    assert np.allclose(p, [1.0, 0.0, 0.5])
    assert sigmoid(np.zeros((3, 4))).shape == (3, 4)


def test_log_loss_matches_naive_formula_on_moderate_logits():
    X, y = make_data(seed=1)
    w, b = np.array([0.4, -0.7]), 0.2
    p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert np.isclose(log_loss(X, y, w, b), naive, atol=1e-9)
    assert np.isclose(log_loss(X, y, w, b, l2=0.5), naive + 0.25 * (w @ w), atol=1e-9)
    assert np.isclose(log_loss(X, y, np.zeros(2), 0.0), np.log(2.0), atol=1e-12)


def test_log_loss_is_finite_for_extreme_logits():
    X = np.array([[1.0], [-1.0]])
    w = np.array([1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        right = log_loss(X, np.array([1.0, 0.0]), w, 0.0)
        wrong = log_loss(X, np.array([0.0, 1.0]), w, 0.0)
        g_w, g_b = grad(X, np.array([0.0, 1.0]), w, 0.0)
    assert np.isfinite(right) and right < 1e-6
    assert np.isclose(wrong, 1000.0, atol=1e-6), "a confidently wrong logit z costs exactly |z|"
    assert np.allclose(g_w, [1.0]) and np.isclose(g_b, 0.0)


def test_bias_is_not_regularized():
    X, y = make_data(seed=2)
    assert np.isclose(log_loss(X, y, np.zeros(2), 5.0, l2=100.0), log_loss(X, y, np.zeros(2), 5.0, l2=0.0))
    _, db_reg = grad(X, y, np.array([0.3, 0.1]), 5.0, l2=100.0)
    _, db = grad(X, y, np.array([0.3, 0.1]), 5.0, l2=0.0)
    assert np.isclose(db_reg, db)


def test_gradient_matches_finite_differences():
    X, y = make_data(seed=3, n=50)
    w, b, l2 = np.array([0.6, -0.2]), -0.4, 0.3
    dw, db = grad(X, y, w, b, l2)
    assert dw.shape == (2,)
    eps = 1e-6
    for j in range(2):
        wp, wm = w.copy(), w.copy()
        wp[j] += eps
        wm[j] -= eps
        num = (log_loss(X, y, wp, b, l2) - log_loss(X, y, wm, b, l2)) / (2 * eps)
        assert np.isclose(dw[j], num, atol=1e-6), f"dw[{j}] = {dw[j]} vs numeric {num}"
    num_b = (log_loss(X, y, w, b + eps, l2) - log_loss(X, y, w, b - eps, l2)) / (2 * eps)
    assert np.isclose(db, num_b, atol=1e-6), f"db = {db} vs numeric {num_b}"


def test_fit_reaches_a_stationary_point():
    X, y = make_data(seed=4)
    w, b = fit(X, y, lr=0.1, n_iter=1000, l2=0.01)
    assert w.shape == (2,)
    dw, db = grad(X, y, w, b, l2=0.01)
    assert np.sqrt(dw @ dw + db ** 2) < 1e-3, "gradient should be ~0 at the optimum"
    assert w[0] > 0 and w[1] < 0, "signs should follow the generating weights [1, -1.5]"
    p = predict_proba(X, w, b)
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1))
    assert np.mean((p > 0.5) == (y == 1)) > 0.7
    assert log_loss(X, y, w, b) < np.log(2.0), "must beat the constant predictor"


def test_l2_shrinks_weights_on_separable_data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)         # perfectly separable
    w0, b0 = fit(X, y, lr=0.5, n_iter=500, l2=0.0)
    w1, _ = fit(X, y, lr=0.5, n_iter=500, l2=0.1)
    w2, _ = fit(X, y, lr=0.5, n_iter=500, l2=1.0)
    assert np.linalg.norm(w0) > np.linalg.norm(w1) > np.linalg.norm(w2)
    assert np.all(np.isfinite(w0))
    assert np.mean((predict_proba(X, w0, b0) > 0.5) == (y == 1)) >= 0.95


def test_predict_proba_is_half_at_zero_weights():
    X = np.random.default_rng(6).normal(size=(7, 3))
    assert np.allclose(predict_proba(X, np.zeros(3), 0.0), 0.5)
