"""A two-layer graph convolutional network (Kipf and Welling) with mean-pool readout. Vectorized NumPy only."""
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Provided."""
    return np.maximum(x, 0.0)


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2 for a symmetric 0/1 adjacency A (N, N) with zero diagonal."""
    raise NotImplementedError


def gcn_layer(X: np.ndarray, A_hat: np.ndarray, W: np.ndarray, b: np.ndarray, activation: bool = True) -> np.ndarray:
    """relu(A_hat @ X @ W + b), or without the relu when activation is False. Shape (N, d_out)."""
    raise NotImplementedError


def gcn_forward(X: np.ndarray, A: np.ndarray, params: dict) -> np.ndarray:
    """Two GCN layers (ReLU after each) using params W1, b1, W2, b2. Returns node embeddings (N, d2)."""
    raise NotImplementedError


def graph_readout(H: np.ndarray, params: dict) -> np.ndarray:
    """Mean over nodes, then H_mean @ W_out + b_out. Shape (d_out,)."""
    raise NotImplementedError
