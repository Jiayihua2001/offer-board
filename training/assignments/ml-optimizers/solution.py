import numpy as np


class SGD:
    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(p, dtype=float)
        self.v = self.momentum * self.v + g
        return p - self.lr * self.v


class Adam:
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(p, dtype=float)
            self.v = np.zeros_like(p, dtype=float)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        m_hat = self.m / (1 - self.beta1 ** self.t)      # undo the pull toward 0 of the zero-initialized EMA
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float = 0.0) -> float:
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + np.cos(np.pi * progress))
