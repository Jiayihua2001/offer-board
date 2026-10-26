"""Cosine top-k retrieval and maximal marginal relevance re-ranking over dense embeddings. Vectorized NumPy only."""
import numpy as np


def normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Unit-length rows; an all-zero row stays zero."""
    raise NotImplementedError


def cosine_topk(query: np.ndarray, docs: np.ndarray, k: int):
    """(indices, scores) of the k most cosine-similar rows of docs (n, d) to query (d,), most similar first."""
    raise NotImplementedError


def mmr(query: np.ndarray, docs: np.ndarray, k: int, lam: float = 0.5) -> list:
    """Greedy maximal marginal relevance: k document indices in selection order."""
    raise NotImplementedError
