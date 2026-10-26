import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A_tilde = A + np.eye(A.shape[0])                      # self-loops keep each node's own features in the sum
    d_inv_sqrt = 1.0 / np.sqrt(A_tilde.sum(axis=1))       # degree >= 1 thanks to the self-loop
    return A_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def gcn_layer(X: np.ndarray, A_hat: np.ndarray, W: np.ndarray, b: np.ndarray, activation: bool = True) -> np.ndarray:
    out = A_hat @ (X @ W) + b                             # transform, then aggregate over the neighborhood
    return relu(out) if activation else out


def gcn_forward(X: np.ndarray, A: np.ndarray, params: dict) -> np.ndarray:
    A_hat = normalize_adjacency(A)
    H = gcn_layer(X, A_hat, params["W1"], params["b1"])
    return gcn_layer(H, A_hat, params["W2"], params["b2"])


def graph_readout(H: np.ndarray, params: dict) -> np.ndarray:
    return H.mean(axis=0) @ params["W_out"] + params["b_out"]
