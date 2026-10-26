import numpy as np


def pca_fit(X: np.ndarray, n_components: int) -> dict:
    mean = X.mean(axis=0)
    Xc = X - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)      # rows of Vt: eigenvectors of the covariance, sorted
    var = S ** 2 / (X.shape[0] - 1)                         # eigenvalues of Xc.T Xc / (N - 1)
    return {
        "mean": mean,
        "components": Vt[:n_components],
        "explained_variance": var[:n_components],
        "explained_variance_ratio": var[:n_components] / var.sum(),
    }


def pca_transform(X: np.ndarray, model: dict) -> np.ndarray:
    return (X - model["mean"]) @ model["components"].T


def pca_inverse_transform(Z: np.ndarray, model: dict) -> np.ndarray:
    return Z @ model["components"] + model["mean"]
