import numpy as np

from solution import calibration_bins, ece, fit_temperature, nll, softmax


def calibrated_logits(seed=5, n=4000, c=5, scale=2.0):
    """Logits z and labels drawn from softmax(z), so z is perfectly calibrated by construction."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, c)) * scale
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    y = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    return z, y


def ref_nll(z, y):
    m = z.max(1, keepdims=True)
    ls = z - m - np.log(np.exp(z - m).sum(1, keepdims=True))
    return -ls[np.arange(len(y)), y].mean()


def test_nll_matches_reference_and_is_stable():
    z, y = calibrated_logits(seed=0, n=50)
    assert np.isclose(nll(z, y, 1.0), ref_nll(z, y), atol=1e-12)
    assert np.isclose(nll(z, y, 2.5), ref_nll(z / 2.5, y), atol=1e-12)
    huge = np.array([[1000.0, 0.0], [-1000.0, 0.0]])
    v = nll(huge, np.array([1, 1]), 1.0)
    assert np.isfinite(v) and np.isclose(v, 500.0), "one row costs 1000, the other 0"


def test_fit_recovers_the_sharpening_factor():
    z, y = calibrated_logits()
    for scale, expected in [(3.0, 3.0), (1.0, 1.0), (1 / 3, 1 / 3)]:
        T = fit_temperature(z * scale, y)
        assert abs(T - expected) < 0.2 * expected, f"scale {scale}: fitted T = {T}, expected about {expected}"


def test_fitted_T_is_a_minimizer_and_argmax_is_unchanged():
    z, y = calibrated_logits(seed=6)
    sharp = 3.0 * z
    T = fit_temperature(sharp, y)
    best = nll(sharp, y, T)
    for other in [T * 0.8, T * 1.2, 1.0, 5.0]:
        assert best <= nll(sharp, y, other) + 1e-9, f"nll at T={T} should beat T={other}"
    assert nll(sharp, y, T) < nll(sharp, y, 1.0)
    assert np.array_equal(softmax(sharp / T).argmax(1), sharp.argmax(1)), "scaling must not change predictions"


def test_fit_is_precise_on_a_smooth_objective():
    z, y = calibrated_logits(seed=7, n=500)
    T = fit_temperature(2.0 * z, y, tol=1e-5)
    grid = np.linspace(T - 0.01, T + 0.01, 201)
    vals = np.array([nll(2.0 * z, y, t) for t in grid])
    assert abs(grid[vals.argmin()] - T) <= 2e-3, "the returned T should be at the bottom of the valley"


def test_calibration_bins_hand_example():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    y = np.array([0, 1, 0, 0])
    acc, conf, count = calibration_bins(probs, y, n_bins=4)     # bins (0, .25], (.25, .5], (.5, .75], (.75, 1]
    assert acc.shape == conf.shape == count.shape == (4,)
    assert count.tolist() == [0, 0, 2, 2]
    assert np.allclose(acc, [0.0, 0.0, 1.0, 0.5])
    assert np.allclose(conf, [0.0, 0.0, 0.65, 0.85])
    assert np.isclose(ece(probs, y, n_bins=4), 0.5 * 0.35 + 0.5 * 0.35)
    acc, conf, count = calibration_bins(probs, y, n_bins=10)
    assert count.sum() == 4 and acc[count == 0].sum() == 0 and conf[count == 0].sum() == 0


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.tile([[0.7, 0.3]], (10, 1))
    y = np.array([0] * 7 + [1] * 3)             # 70% correct at 70% confidence
    assert np.isclose(ece(probs, y, n_bins=10), 0.0, atol=1e-12)
    y_all_right = np.zeros(10, int)
    assert np.isclose(ece(probs, y_all_right, n_bins=10), 0.3)


def test_temperature_scaling_reduces_ece_and_nll():
    z, y = calibrated_logits(seed=8)
    sharp = 3.0 * z
    T = fit_temperature(sharp, y)
    before, after = ece(softmax(sharp), y), ece(softmax(sharp / T), y)
    assert before > 0.15, f"overconfident logits should have a large ECE, got {before}"
    assert after < 0.05, f"after scaling ECE should be small, got {after}"
    assert nll(sharp, y, T) < nll(sharp, y, 1.0)
    flat = z / 3.0
    T_flat = fit_temperature(flat, y)
    assert T_flat < 1.0 and ece(softmax(flat / T_flat), y) < ece(softmax(flat), y)
