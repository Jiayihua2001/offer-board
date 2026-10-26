"""Cross-validation splitters that preserve class proportions or keep groups intact. NumPy only."""
import numpy as np


def stratified_kfold(y: np.ndarray, k: int, rng: np.random.Generator):
    """k (train_idx, test_idx) pairs whose test sets each hold floor(n_c / k) or ceil(n_c / k) samples of every class c.

    Shuffle with rng. Raise ValueError if some class has fewer than k samples.
    """
    raise NotImplementedError


def group_kfold(groups: np.ndarray, k: int):
    """k (train_idx, test_idx) pairs with every group in exactly one test fold, balanced greedily by size.

    Raise ValueError if there are fewer than k distinct groups.
    """
    raise NotImplementedError
