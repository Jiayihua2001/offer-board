# Stratified and grouped cross-validation splits

Writing a splitter sounds trivial until the interviewer asks why plain k-fold overestimates a model whose data has 10 samples per molecule or per patient, or why a 2% positive class needs stratification. This is the "do you think about leakage" question, and the materials-discovery version of it (same composition or prototype in train and test) is one you have lived through.

Implement two splitters that return a list of `k` `(train_idx, test_idx)` pairs of integer index arrays:

- `stratified_kfold(y, k, rng)`: `y` is an integer label array of length `N`. Every fold's test set contains either `floor(n_c / k)` or `ceil(n_c / k)` samples of each class `c`. Samples are shuffled with `rng` before being dealt, so the folds are random but reproducible. Raise `ValueError` if any class has fewer than `k` samples.
- `group_kfold(groups, k)`: `groups` is an array of length `N` with a group id per sample. Every group appears in exactly one test fold, so no group is ever split between train and test. Balance the folds greedily: assign groups from largest to smallest, each to the fold with the fewest samples so far. Raise `ValueError` if there are fewer than `k` distinct groups.

In both cases the test sets partition `0..N-1`, and each train set is the complement of its test set.

## Example

```
y = np.array([0, 0, 0, 0, 1, 1])
for train_idx, test_idx in stratified_kfold(y, k=2, rng=np.random.default_rng(0)):
    print(sorted(y[test_idx]))        # [0, 0, 1] twice: each fold has 2 of class 0 and 1 of class 1

groups = np.array([0, 0, 0, 1, 1, 2, 2, 3])
group_kfold(groups, k=2)    # test folds: groups {0, 3} (4 samples) and {1, 2} (4 samples)
```

## Constraints

- No loops over samples; loop over classes, groups or folds only (`np.unique` and boolean masks do the work).
- Test sets are disjoint and cover every index exactly once; train sets are exact complements.
- For `group_kfold` the greedy balancing guarantees every fold has at most `N / k + max_group_size` samples; a test checks it.

## What the interviewer listens for

- Why stratify: with a rare class, a random fold can hold zero positives, which makes per-fold metrics undefined and the variance across folds huge.
- Leakage: if samples sharing a group are correlated (same structure, same patient, same prompt template), random splits let the model memorize the group and inflate the validation score. Group splitting estimates performance on new groups, which is usually the question you care about.
- The remaining limitations: group k-fold cannot also stratify perfectly; time-series data needs forward-chaining splits instead; and the choice of `k` trades bias (small `k`, small train sets) against variance and compute.
- Follow-up: how you would evaluate a materials model on new compositions versus new polymorphs of known compositions, and why those two splits give very different numbers.
