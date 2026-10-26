"""A two-layer MLP with ReLU and softmax cross-entropy, forward and backward by hand. Vectorized NumPy only."""
import numpy as np


def init_params(d_in: int, d_hidden: int, d_out: int, rng: np.random.Generator) -> dict:
    """He-initialized parameters. Provided; the tests rely on it."""
    return {
        "W1": rng.normal(size=(d_in, d_hidden)) * np.sqrt(2.0 / d_in),
        "b1": np.zeros(d_hidden),
        "W2": rng.normal(size=(d_hidden, d_out)) * np.sqrt(2.0 / d_hidden),
        "b2": np.zeros(d_out),
    }


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax, stable. Provided."""
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def forward(X: np.ndarray, params: dict):
    """Return (logits, cache) with logits of shape (N, C). cache is whatever backward needs."""
    raise NotImplementedError


def loss_and_grads(X: np.ndarray, y: np.ndarray, params: dict):
    """Mean cross-entropy of integer labels y (shape (N,)) and a dict of gradients with the keys of params."""
    raise NotImplementedError


def train(X: np.ndarray, y: np.ndarray, params: dict, lr: float = 0.5, n_iter: int = 300):
    """Full-batch gradient descent. Returns (new_params, losses) with one loss per iteration."""
    raise NotImplementedError


def predict(X: np.ndarray, params: dict) -> np.ndarray:
    """Integer class predictions of shape (N,)."""
    raise NotImplementedError
