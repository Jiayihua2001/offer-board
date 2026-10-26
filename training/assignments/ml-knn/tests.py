import numpy as np

from solution import knn_predict, pairwise_sq_dists


def test_pairwise_matches_broadcast_reference_and_is_nonnegative():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
    D = pairwise_sq_dists(A, B)
    ref = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    assert D.shape == (7, 5)
    assert np.allclose(D, ref, atol=1e-10)
    big = rng.normal(size=(50, 8)) * 1e4 + 1e6           # large norms: cancellation territory
    DD = pairwise_sq_dists(big, big)
    assert np.all(DD >= 0), "clip at zero: the expansion can produce tiny negative values"
    assert np.allclose(np.diag(pairwise_sq_dists(A, A)), 0.0, atol=1e-9)


def test_k1_returns_label_of_the_nearest_point():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([5, 5, 2, 2])
    pred = knn_predict(X_train, y_train, np.array([[0.9], [2.4], [-10.0]]), k=1)
    assert pred.shape == (3,)
    assert pred.tolist() == [5, 2, 5]


def test_majority_vote_and_smallest_label_tie_break():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([5, 5, 2, 2, 2])
    assert knn_predict(X_train, y_train, np.array([[1.4]]), k=2).tolist() == [2], "5 vs 2 tie -> 2"
    assert knn_predict(X_train, y_train, np.array([[1.4]]), k=3).tolist() == [5], "neighbors 1, 2, 0 -> labels 5, 2, 5"
    assert knn_predict(X_train, y_train, np.array([[1.4]]), k=5).tolist() == [2]
    y_neg = np.array([-1, -1, 7, 7, 7])
    assert knn_predict(X_train, y_neg, np.array([[1.4]]), k=2).tolist() == [-1], "labels may be negative"
    assert knn_predict(X_train, y_neg, np.array([[1.4]]), k=4).tolist() == [-1], "neighbors 1, 2, 0, 3: two of each label"
    assert knn_predict(X_train, y_neg, np.array([[1.4]]), k=5).tolist() == [7]


def test_predicts_two_blobs_correctly():
    rng = np.random.default_rng(1)
    X_train = np.vstack([rng.normal(size=(40, 3)), rng.normal(size=(40, 3)) + 6.0])
    y_train = np.array([10] * 40 + [20] * 40)
    X_test = np.vstack([rng.normal(size=(10, 3)), rng.normal(size=(10, 3)) + 6.0])
    pred = knn_predict(X_train, y_train, X_test, k=5)
    assert pred.tolist() == [10] * 10 + [20] * 10


def test_matches_bruteforce_on_random_data():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(60, 5))
    y_train = rng.integers(0, 3, size=60)
    X_test = rng.normal(size=(15, 5))
    k = 7
    pred = knn_predict(X_train, y_train, X_test, k)
    for i in range(15):
        d = ((X_train - X_test[i]) ** 2).sum(axis=1)
        nn = np.argsort(d)[:k]
        counts = np.bincount(y_train[nn], minlength=3)
        best = np.flatnonzero(counts == counts.max()).min()
        assert pred[i] == best, f"test row {i}: got {pred[i]}, expected {best}"


def test_larger_input_runs_vectorized():
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(3000, 16))
    y_train = rng.integers(0, 4, size=3000)
    X_test = rng.normal(size=(400, 16))
    pred = knn_predict(X_train, y_train, X_test, k=9)
    assert pred.shape == (400,)
    assert set(np.unique(pred)) <= {0, 1, 2, 3}
