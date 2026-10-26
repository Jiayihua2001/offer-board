import numpy as np

from solution import pca_fit, pca_inverse_transform, pca_transform


def correlated_data(seed=0, n=200, d=5):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, d)) * np.array([5.0, 3.0, 1.5, 0.7, 0.2])[:d]
    mix = np.linalg.qr(rng.normal(size=(d, d)))[0]
    return latent @ mix + rng.normal(size=d) * 10          # rotated, with a big offset


def eig_reference(X):
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order].T


def test_shapes_and_orthonormal_components():
    X = correlated_data()
    m = pca_fit(X, 3)
    assert m["mean"].shape == (5,) and np.allclose(m["mean"], X.mean(axis=0))
    assert m["components"].shape == (3, 5)
    assert m["explained_variance"].shape == (3,) and m["explained_variance_ratio"].shape == (3,)
    assert np.allclose(m["components"] @ m["components"].T, np.eye(3), atol=1e-10)
    assert pca_transform(X, m).shape == (200, 3)
    assert pca_inverse_transform(pca_transform(X, m), m).shape == (200, 5)


def test_matches_eigendecomposition_up_to_sign():
    X = correlated_data(seed=1)
    vals, vecs = eig_reference(X)
    m = pca_fit(X, 5)
    assert np.allclose(m["explained_variance"], vals, rtol=1e-8)
    assert np.all(np.diff(m["explained_variance"]) <= 1e-12), "components must be sorted by decreasing variance"
    for i in range(5):
        assert np.isclose(abs(m["components"][i] @ vecs[i]), 1.0, atol=1e-8), f"component {i} differs from the eigenvector"


def test_explained_variance_ratio():
    X = correlated_data(seed=2)
    total = ((X - X.mean(axis=0)) ** 2).sum() / (len(X) - 1)
    full = pca_fit(X, 5)
    assert np.isclose(full["explained_variance_ratio"].sum(), 1.0, atol=1e-10)
    two = pca_fit(X, 2)
    assert np.allclose(two["explained_variance_ratio"], two["explained_variance"] / total)
    assert two["explained_variance_ratio"].sum() < 1.0, "the ratio is over the total variance, not over the kept components"
    assert np.allclose(two["explained_variance_ratio"], full["explained_variance_ratio"][:2])


def test_transform_is_centered_projection_with_uncorrelated_scores():
    X = correlated_data(seed=3)
    m = pca_fit(X, 3)
    Z = pca_transform(X, m)
    assert np.allclose(Z, (X - m["mean"]) @ m["components"].T, atol=1e-10)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-8)
    cov_z = Z.T @ Z / (len(X) - 1)
    assert np.allclose(cov_z, np.diag(m["explained_variance"]), atol=1e-8), "scores are uncorrelated with variance = eigenvalue"


def test_reconstruction_error_decreases_and_vanishes_at_full_rank():
    X = correlated_data(seed=4)
    vals, _ = eig_reference(X)
    errs = []
    for k in range(1, 6):
        m = pca_fit(X, k)
        X_hat = pca_inverse_transform(pca_transform(X, m), m)
        errs.append(((X - X_hat) ** 2).sum() / (len(X) - 1))
        assert np.isclose(errs[-1], vals[k:].sum(), rtol=1e-6), f"error with {k} components should be the dropped variance"
    assert all(a > b for a, b in zip(errs, errs[1:]))
    assert errs[-1] < 1e-8


def test_exactly_low_rank_data():
    X = np.array([[-2, -1], [-1, -0.5], [1, 0.5], [2, 1]], float)
    m = pca_fit(X, 1)
    assert np.isclose(abs(m["components"][0] @ np.array([2, 1]) / np.sqrt(5)), 1.0)
    assert np.isclose(m["explained_variance_ratio"][0], 1.0)
    assert np.allclose(pca_inverse_transform(pca_transform(X, m), m), X, atol=1e-10)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 6))       # rank 2 in 6-D
    m = pca_fit(X, 2)
    assert np.isclose(m["explained_variance_ratio"].sum(), 1.0, atol=1e-10)
    assert np.allclose(pca_inverse_transform(pca_transform(X, m), m), X, atol=1e-8)
