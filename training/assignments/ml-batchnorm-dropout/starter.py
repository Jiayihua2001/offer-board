"""Batch normalization and inverted dropout forward passes with train/eval modes. Vectorized NumPy only."""
import numpy as np


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, running_mean: np.ndarray,
                      running_var: np.ndarray, training: bool, momentum: float = 0.1, eps: float = 1e-5):
    """Normalize x (N, D) per feature.

    Training: batch mean and biased batch variance; return running stats updated with momentum
    (unbiased batch variance for the running estimate). Eval: use the running stats, return them unchanged.
    Returns (out, running_mean, running_var) without modifying the inputs.
    """
    raise NotImplementedError


def dropout_forward(x: np.ndarray, p: float, training: bool, rng: np.random.Generator):
    """Inverted dropout with drop probability p in [0, 1). Returns (out, mask) with mask shaped like x."""
    raise NotImplementedError
