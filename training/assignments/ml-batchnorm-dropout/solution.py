import numpy as np


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, running_mean: np.ndarray,
                      running_var: np.ndarray, training: bool, momentum: float = 0.1, eps: float = 1e-5):
    if training:
        n = x.shape[0]
        mu = x.mean(axis=0)
        var = x.var(axis=0)                                   # biased: this is what normalizes the batch
        x_hat = (x - mu) / np.sqrt(var + eps)
        unbiased = var * n / max(n - 1, 1)                    # the running estimate tracks the population variance
        running_mean = (1 - momentum) * running_mean + momentum * mu
        running_var = (1 - momentum) * running_var + momentum * unbiased
    else:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    return gamma * x_hat + beta, running_mean, running_var


def dropout_forward(x: np.ndarray, p: float, training: bool, rng: np.random.Generator):
    if not training or p == 0.0:
        return x.copy(), np.ones_like(x)
    mask = (rng.random(x.shape) >= p).astype(x.dtype)
    return x * mask / (1.0 - p), mask
