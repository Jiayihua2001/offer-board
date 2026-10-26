import numpy as np


def _folds_from_assignment(fold_of: np.ndarray, k: int):
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def stratified_kfold(y: np.ndarray, k: int, rng: np.random.Generator):
    y = np.asarray(y)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) < k:
            raise ValueError(f"class {c} has {len(idx)} samples, fewer than k={k}")
        idx = rng.permutation(idx)
        fold_of[idx] = np.arange(len(idx)) % k            # deal round-robin: counts differ by at most one
    return _folds_from_assignment(fold_of, k)


def group_kfold(groups: np.ndarray, k: int):
    groups = np.asarray(groups)
    ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(ids) < k:
        raise ValueError(f"{len(ids)} groups, fewer than k={k}")
    fold_of_group = np.empty(len(ids), dtype=int)
    load = np.zeros(k, dtype=int)
    for g in np.argsort(-counts, kind="stable"):            # largest groups first, into the emptiest fold
        f = int(np.argmin(load))
        fold_of_group[g] = f
        load[f] += counts[g]
    return _folds_from_assignment(fold_of_group[inverse], k)
