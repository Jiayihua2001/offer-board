"""k-means (Lloyd's algorithm) with k-means++ seeding. Vectorized NumPy only."""
import numpy as np


def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return k rows of X chosen by D^2 sampling, shape (k, D)."""
    raise NotImplementedError


def assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for each row of X, shape (N,)."""
    raise NotImplementedError


def inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    raise NotImplementedError


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 100, tol: float = 1e-6):
    """Run Lloyd's algorithm from a k-means++ init.

    Returns (centroids (k, D), labels (N,), history) where history holds the inertia at each iteration.
    """
    raise NotImplementedError
