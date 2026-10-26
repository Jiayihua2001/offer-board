"""Principal component analysis through the SVD of the centered data. Vectorized NumPy only."""
import numpy as np


def pca_fit(X: np.ndarray, n_components: int) -> dict:
    """Return {"mean": (D,), "components": (n_components, D), "explained_variance": (n_components,),
    "explained_variance_ratio": (n_components,)}. Components are unit rows sorted by decreasing variance."""
    raise NotImplementedError


def pca_transform(X: np.ndarray, model: dict) -> np.ndarray:
    """Project centered X onto the components, shape (N, n_components)."""
    raise NotImplementedError


def pca_inverse_transform(Z: np.ndarray, model: dict) -> np.ndarray:
    """Map scores back to the original space, shape (N, D)."""
    raise NotImplementedError
