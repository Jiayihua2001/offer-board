import numpy as np


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)       # (N, k)


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    idx = [int(rng.integers(n))]
    d2 = ((X - X[idx[0]]) ** 2).sum(axis=1)
    for _ in range(1, k):
        i = int(rng.choice(n, p=d2 / d2.sum()))                    # far points are likely, chosen points impossible
        idx.append(i)
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(axis=1))
    return X[idx].copy()


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return _sq_dists(X, centroids).argmin(axis=1)


def inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    return float(((X - centroids[labels]) ** 2).sum())


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 100, tol: float = 1e-6):
    centroids = kmeans_pp_init(X, k, rng)
    history = []
    labels = assign(X, centroids)
    for _ in range(n_iter):
        labels = assign(X, centroids)
        history.append(inertia(X, centroids, labels))
        new = centroids.copy()
        for j in range(k):
            members = labels == j
            if members.any():
                new[j] = X[members].mean(axis=0)
        shift = np.abs(new - centroids).max()
        centroids = new
        if shift <= tol:
            break
    labels = assign(X, centroids)
    return centroids, labels, np.asarray(history)
