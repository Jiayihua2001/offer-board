import numpy as np

from solution import group_kfold, stratified_kfold


def check_partition(folds, n):
    seen = np.zeros(n, int)
    for train_idx, test_idx in folds:
        train_idx, test_idx = np.asarray(train_idx), np.asarray(test_idx)
        assert len(np.intersect1d(train_idx, test_idx)) == 0, "train and test overlap"
        assert len(train_idx) + len(test_idx) == n, "train must be the complement of test"
        assert len(np.unique(test_idx)) == len(test_idx), "duplicate test indices"
        seen[test_idx] += 1
    assert np.all(seen == 1), "every index must be in exactly one test fold"


def test_stratified_partition_and_class_proportions():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1, 2], size=203, p=[0.6, 0.3, 0.1])
    k = 5
    folds = list(stratified_kfold(y, k, np.random.default_rng(1)))
    assert len(folds) == k
    check_partition(folds, len(y))
    counts = np.bincount(y, minlength=3)
    for _, test_idx in folds:
        c = np.bincount(y[test_idx], minlength=3)
        assert np.all((c == counts // k) | (c == counts // k + 1)), f"fold class counts {c} vs totals {counts}"


def test_stratified_small_example():
    y = np.array([0, 0, 0, 0, 1, 1])
    folds = list(stratified_kfold(y, 2, np.random.default_rng(0)))
    check_partition(folds, 6)
    for _, test_idx in folds:
        assert sorted(y[test_idx].tolist()) == [0, 0, 1]


def test_stratified_is_shuffled_and_reproducible():
    y = np.repeat([0, 1], 50)                 # class 0 is indices 0..49: unshuffled folds would be contiguous
    a = list(stratified_kfold(y, 5, np.random.default_rng(3)))
    b = list(stratified_kfold(y, 5, np.random.default_rng(3)))
    c = list(stratified_kfold(y, 5, np.random.default_rng(4)))
    for (_, ta), (_, tb) in zip(a, b):
        assert np.array_equal(np.sort(ta), np.sort(tb)), "same seed must give the same folds"
    assert any(not np.array_equal(np.sort(ta), np.sort(tc)) for (_, ta), (_, tc) in zip(a, c)), "different seeds should differ"
    contiguous = all(np.ptp(np.sort(t)[:10]) == 9 for _, t in a)
    assert not contiguous, "folds should be shuffled, not contiguous blocks"


def test_stratified_rejects_classes_smaller_than_k():
    y = np.array([0, 0, 0, 0, 1, 1])
    try:
        list(stratified_kfold(y, 3, np.random.default_rng(0)))
    except ValueError:
        return
    raise AssertionError("expected ValueError when a class has fewer than k samples")


def test_group_kfold_never_splits_a_group():
    rng = np.random.default_rng(5)
    groups = rng.integers(0, 15, size=200)
    k = 4
    folds = list(group_kfold(groups, k))
    assert len(folds) == k
    check_partition(folds, len(groups))
    for train_idx, test_idx in folds:
        assert len(np.intersect1d(groups[train_idx], groups[test_idx])) == 0, "a group leaked across the split"
    tested = np.concatenate([np.unique(groups[t]) for _, t in folds])
    assert np.array_equal(np.sort(tested), np.unique(groups)), "every group is in exactly one test fold"


def test_group_kfold_is_balanced():
    rng = np.random.default_rng(6)
    sizes = rng.integers(1, 20, size=30)
    groups = np.repeat(np.arange(30), sizes)
    groups = groups[rng.permutation(len(groups))]
    k = 5
    folds = list(group_kfold(groups, k))
    check_partition(folds, len(groups))
    bound = len(groups) / k + sizes.max()
    for _, test_idx in folds:
        assert len(test_idx) <= bound, f"fold of size {len(test_idx)} exceeds the greedy bound {bound}"
    small = group_kfold(np.array([0, 0, 0, 1, 1, 2, 2, 3]), 2)
    test_sizes = sorted(len(t) for _, t in small)
    assert test_sizes == [4, 4]


def test_group_kfold_rejects_too_few_groups():
    try:
        list(group_kfold(np.array([0, 0, 1, 1]), 3))
    except ValueError:
        return
    raise AssertionError("expected ValueError when there are fewer than k groups")
