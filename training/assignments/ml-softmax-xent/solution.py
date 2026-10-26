import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def cross_entropy(z: np.ndarray, y: np.ndarray) -> float:
    ls = log_softmax(z)
    return float(-ls[np.arange(z.shape[0]), y].mean())


def cross_entropy_grad(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = softmax(z)
    p[np.arange(z.shape[0]), y] -= 1.0
    return p / z.shape[0]
