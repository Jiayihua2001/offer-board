import numpy as np

from solution import assign, inertia, kmeans, kmeans_pp_init

CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])


def blobs(seed=0, n_per=50, scale=0.5):
    rng = np.random.default_rng(seed)
    X = np.vstack([c + scale * rng.normal(size=(n_per, 2)) for c in CENTERS])
    truth = np.repeat(np.arange(len(CENTERS)), n_per)
    return X, truth


def true_cluster_of(X, rows, truth):
    """Map each row of `rows` (which must be a row of X) to its true cluster id."""
    out = []
    for r in rows:
        matches = np.where(np.all(X == r, axis=1))[0]
        assert len(matches) == 1, "every k-means++ centroid must be an actual data point"
        out.append(truth[matches[0]])
    return out


def test_assign_and_inertia_hand_example():
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], float)
    C = np.array([[0, 0], [10, 10]], float)
    labels = assign(X, C)
    assert labels.shape == (4,)
    assert labels.tolist() == [0, 0, 1, 1]
    assert np.isclose(inertia(X, C, labels), 2.0)
    assert np.isclose(inertia(X, np.array([[0, 0.5], [10, 10.5]]), labels), 1.0)


def test_kmeanspp_seeds_land_in_distinct_clusters():
    X, truth = blobs()
    for seed in range(5):
        C = kmeans_pp_init(X, 4, np.random.default_rng(seed))
        assert C.shape == (4, 2)
        owners = true_cluster_of(X, C, truth)
        assert len(set(owners)) == 4, f"seed {seed}: seeds fell in clusters {owners}; D^2 sampling should spread them"


def test_recovers_well_separated_clusters_up_to_permutation():
    X, truth = blobs(seed=1)
    centroids, labels, history = kmeans(X, 4, np.random.default_rng(1))
    assert centroids.shape == (4, 2) and labels.shape == (200,)
    table = np.zeros((4, 4), int)
    np.add.at(table, (truth, labels), 1)
    assert np.all((table > 0).sum(axis=1) == 1), f"a true cluster was split:\n{table}"
    assert np.all((table > 0).sum(axis=0) == 1), f"a predicted cluster mixes true clusters:\n{table}"
    for c in CENTERS:
        assert np.min(np.linalg.norm(centroids - c, axis=1)) < 0.3, f"no centroid near {c}"


def test_inertia_history_never_increases_and_matches_final_state():
    X, _ = blobs(seed=2, scale=2.0)          # overlapping clusters: more iterations before convergence
    centroids, labels, history = kmeans(X, 4, np.random.default_rng(2), n_iter=100)
    history = np.asarray(history, dtype=float)
    assert 1 <= len(history) <= 100
    assert np.all(np.diff(history) <= 1e-8), f"inertia went up: {history}"
    assert np.isclose(history[-1], inertia(X, centroids, labels), rtol=1e-6)
    assert np.array_equal(labels, assign(X, centroids)), "returned labels must be the assignment to the returned centroids"


def test_centroids_are_the_means_of_their_clusters():
    X, _ = blobs(seed=3)
    centroids, labels, _ = kmeans(X, 4, np.random.default_rng(3))
    for j in range(4):
        assert np.any(labels == j), "no cluster should end up empty on this data"
        assert np.allclose(centroids[j], X[labels == j].mean(axis=0), atol=1e-6)


def test_converges_early_on_easy_data():
    X, _ = blobs(seed=4)
    _, _, history = kmeans(X, 4, np.random.default_rng(4), n_iter=100)
    assert len(history) < 20, f"should stop once centroids stop moving, ran {len(history)} iterations"


def test_k_equals_n_gives_zero_inertia():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 3))
    centroids, labels, history = kmeans(X, 6, rng)
    assert sorted(labels.tolist()) == list(range(6)), "each point gets its own cluster"
    assert np.isclose(history[-1], 0.0, atol=1e-12)
