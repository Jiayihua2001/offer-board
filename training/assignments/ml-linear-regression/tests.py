import numpy as np

from solution import fit_closed_form, fit_gd, predict


def make_data(seed=0, n=200, d=3, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5])[:d]
    y = 0.7 + X @ w_true + noise * rng.normal(size=n)
    return X, y, w_true


def test_predict_uses_intercept_first():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    w = np.array([1.0, 10.0, 100.0])
    assert np.allclose(predict(X, w), [101.0, 321.0])
    assert predict(X, w).shape == (2,)


def test_closed_form_recovers_exact_line():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    w = fit_closed_form(X, y)
    assert w.shape == (2,), f"expected shape (2,), got {w.shape}"
    assert np.allclose(w, [1.0, 2.0], atol=1e-8)
    assert np.allclose(predict(np.array([[4.0]]), w), [9.0])


def test_closed_form_matches_lstsq_with_noise():
    X, y, _ = make_data(seed=1)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    ref = np.linalg.lstsq(Xb, y, rcond=None)[0]
    w = fit_closed_form(X, y)
    assert w.shape == (4,)
    assert np.allclose(w, ref, atol=1e-8), f"{w} vs {ref}"


def test_closed_form_survives_collinear_features():
    X, y, _ = make_data(seed=2)
    X2 = np.hstack([X, X[:, :1]])        # last column duplicates the first: X^T X is singular
    w = fit_closed_form(X2, y)
    assert w.shape == (5,) and np.all(np.isfinite(w)), "weights must be finite for a rank-deficient X"
    Xb = np.hstack([np.ones((len(X2), 1)), X2])
    ref_pred = Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0]
    assert np.allclose(predict(X2, w), ref_pred, atol=1e-6), "fitted values are unique even when w is not"


def test_gd_converges_to_closed_form():
    X, y, _ = make_data(seed=3)
    w_cf = fit_closed_form(X, y)
    w, losses = fit_gd(X, y, lr=0.1, n_iter=500)
    losses = np.asarray(losses, dtype=float)
    assert w.shape == (4,)
    assert losses.shape == (500,), f"expected one loss per iteration, got {losses.shape}"
    assert np.allclose(w, w_cf, atol=1e-4), f"GD {w} should match closed form {w_cf}"
    mse_cf = np.mean((predict(X, w_cf) - y) ** 2)
    assert abs(losses[-1] - mse_cf) < 1e-6
    assert losses[0] > losses[-1]


def test_gd_loss_never_increases():
    X, y, _ = make_data(seed=4)
    _, losses = fit_gd(X, y, lr=0.1, n_iter=300)
    losses = np.asarray(losses, dtype=float)
    assert np.all(np.diff(losses) <= 1e-10), "full-batch GD on a convex quadratic must not go uphill"
    assert np.isclose(losses[0], np.mean(y ** 2)), "starting from w = 0 the first loss is mean(y^2)"


def test_single_feature_shapes():
    X, y, w_true = make_data(seed=5, d=1, noise=0.0)
    w = fit_closed_form(X, y)
    assert w.shape == (2,)
    assert np.allclose(w, [0.7, w_true[0]], atol=1e-8)
    w_gd, losses = fit_gd(X, y, lr=0.1, n_iter=400)
    assert w_gd.shape == (2,) and len(losses) == 400
    assert np.allclose(w_gd, w, atol=1e-5)
