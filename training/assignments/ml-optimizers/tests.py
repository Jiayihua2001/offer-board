import numpy as np

from solution import SGD, Adam, warmup_cosine


def test_sgd_momentum_two_steps_by_hand():
    opt = SGD(lr=0.1, momentum=0.9)
    p = np.array([1.0])
    p = opt.step(p, np.array([2.0]))
    assert np.allclose(p, [0.8]), f"after step 1 expected [0.8], got {p}"
    p = opt.step(p, np.array([2.0]))
    assert np.allclose(p, [0.42]), f"after step 2 expected [0.42] (v = 3.8), got {p}"


def test_sgd_without_momentum_is_plain_gd_and_keeps_shapes():
    opt = SGD(lr=0.5)
    p = np.arange(6.0).reshape(2, 3)
    g = np.ones((2, 3))
    p0 = p.copy()
    out = opt.step(p, g)
    assert out.shape == (2, 3)
    assert np.allclose(out, p0 - 0.5)
    assert np.array_equal(p, p0), "step must not modify its input in place"
    out = opt.step(out, g)
    assert np.allclose(out, p0 - 1.0)


def test_adam_first_step_is_a_sign_step():
    opt = Adam(lr=0.1)
    p = opt.step(np.array([1.0, 1.0, 1.0]), np.array([3.0, -0.001, 0.0]))
    assert np.allclose(p, [0.9, 1.1, 1.0], atol=1e-5), f"expected lr * sign(g), got {p}"


def test_adam_matches_reference_over_several_steps():
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05
    rng = np.random.default_rng(0)
    p = rng.normal(size=(4,))
    opt = Adam(lr=lr, beta1=b1, beta2=b2, eps=eps)
    m = np.zeros(4)
    v = np.zeros(4)
    p_ref = p.copy()
    for t in range(1, 8):
        g = rng.normal(size=(4,)) * 3
        p = opt.step(p, g)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        p_ref = p_ref - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        assert np.allclose(p, p_ref, atol=1e-12), f"step {t}: {p} vs {p_ref}"


def test_adam_is_invariant_to_gradient_scale():
    a = np.array([1.0, 10.0])
    p1, p2 = np.array([3.0, -2.0]), np.array([3.0, -2.0])
    o1, o2 = Adam(lr=0.1), Adam(lr=0.1)
    for _ in range(20):
        p1 = o1.step(p1, a * p1)
        p2 = o2.step(p2, 1000.0 * a * p2)
    assert np.allclose(p1, p2, atol=1e-5), "Adam's update depends on the direction of g, not its scale"


def test_both_optimizers_minimize_an_ill_conditioned_quadratic():
    a = np.array([1.0, 10.0])             # f(p) = 0.5 * sum(a * p^2), gradient a * p
    p = np.array([3.0, -2.0])
    opt = SGD(lr=0.05, momentum=0.9)
    for _ in range(300):
        p = opt.step(p, a * p)
    assert np.abs(p).max() < 1e-4, f"SGD+momentum did not converge: {p}"

    p = np.array([3.0, -2.0])
    opt = Adam(lr=0.2)
    for t in range(500):
        opt.lr = warmup_cosine(t, 500, 50, 0.2)
        p = opt.step(p, a * p)
    assert np.abs(p).max() < 1e-5, f"Adam with a cosine schedule did not converge: {p}"


def test_warmup_cosine_shape():
    assert np.isclose(warmup_cosine(0, 100, 10, 1.0), 0.0)
    assert np.isclose(warmup_cosine(5, 100, 10, 1.0), 0.5)
    assert np.isclose(warmup_cosine(10, 100, 10, 1.0), 1.0)
    assert np.isclose(warmup_cosine(55, 100, 10, 1.0), 0.5)
    assert np.isclose(warmup_cosine(100, 100, 10, 1.0), 0.0)
    assert np.isclose(warmup_cosine(150, 100, 10, 1.0), 0.0), "stay at lr_min after total_steps"
    assert np.isclose(warmup_cosine(100, 100, 10, 1.0, lr_min=0.1), 0.1)
    assert np.isclose(warmup_cosine(55, 100, 10, 1.0, lr_min=0.1), 0.55)
    lrs = np.array([warmup_cosine(s, 100, 10, 1.0) for s in range(101)])
    assert np.all(np.diff(lrs[:11]) > 0), "warmup must increase"
    assert np.all(np.diff(lrs[10:]) <= 1e-12), "decay must not increase"
    assert np.all((lrs >= 0) & (lrs <= 1.0))
