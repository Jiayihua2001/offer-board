import numpy as np


def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    return np.maximum(a2 - 2.0 * A @ B.T + b2, 0.0)


def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    classes, y_idx = np.unique(y_train, return_inverse=True)         # sorted classes -> ties favor the smallest
    D = pairwise_sq_dists(X_test, X_train)                           # (m, n)
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]                    # (m, k) unordered nearest neighbors
    votes = np.zeros((X_test.shape[0], len(classes)), dtype=int)
    np.add.at(votes, (np.arange(X_test.shape[0])[:, None], y_idx[nn]), 1)
    return classes[votes.argmax(axis=1)]                             # argmax takes the first max
