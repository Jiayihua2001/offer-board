"""Optimizers with state and a learning-rate schedule. NumPy only; follow the PyTorch conventions."""
import numpy as np


class SGD:
    """Gradient descent with (optional) momentum: v = momentum * v + g; p = p - lr * v."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        # Keep the velocity here; create it on the first step, shaped like p.

    def step(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Return the updated parameter array; do not modify p in place."""
        raise NotImplementedError


class Adam:
    """Adam with bias correction: p = p - lr * m_hat / (sqrt(v_hat) + eps)."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        # Keep m, v and the step counter here.

    def step(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Return the updated parameter array; do not modify p in place."""
        raise NotImplementedError


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Linear warmup from 0 to lr_max over warmup_steps, then cosine decay to lr_min at total_steps."""
    raise NotImplementedError
