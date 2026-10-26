# A CART decision tree classifier

Trees are the "can you write a recursive algorithm with a clear stopping rule and a vectorized inner loop" question, and the base learner of the gradient-boosting models you used for quality estimation. Implement a CART classifier with Gini impurity, axis-aligned thresholds at the midpoints between consecutive distinct feature values, a depth limit and a minimum leaf size.

- `gini(y) -> float`: `1 - sum_c p_c^2` for integer labels `y`.
- `best_split(X, y, min_samples_leaf=1) -> (feature, threshold, gain) or None`: search every feature and every midpoint between consecutive distinct sorted values; a split sends `x[:, feature] <= threshold` left. Only splits that leave at least `min_samples_leaf` samples on each side count. Return the split with the largest Gini gain, or `None` if there is no valid split with positive gain.
- `class DecisionTree(max_depth=None, min_samples_leaf=1)` with `fit(X, y) -> self` and `predict(X) -> labels` of shape `(N,)`. After `fit`, `self.root` holds the tree as nested dicts: a leaf is `{"value": label}` and an internal node is `{"feature": f, "threshold": t, "left": node, "right": node}`. A leaf predicts the majority label (ties to the smallest label). A node becomes a leaf when it is pure, when `depth == max_depth` (the root has depth 0), when it has fewer than `2 * min_samples_leaf` samples, or when `best_split` returns `None`.

## Example

```
X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
y = np.array([0, 0, 0, 1, 1, 1])
gini(y)                      -> 0.5
best_split(X, y)             -> (0, 6.5, 0.5)
best_split(X, y, min_samples_leaf=4)  -> None

tree = DecisionTree(max_depth=2).fit(X2, y2)     # y2 = (x0 > 0) & (x1 > 0)
tree.root   -> {"feature": 0, "threshold": ..., "left": {"value": 0}, "right": {"feature": 1, ...}}
```

## Constraints

- The split search for one feature must be vectorized over thresholds (sort once, cumulative class counts): `O(D N log N)` per node, not `O(D N^2)`.
- Loops over features and the recursion over nodes are fine; no loop over samples inside the split search.
- Thresholds must be the midpoints (a test checks the exact value), and the depth limit and leaf size must be respected exactly (tests walk `root`).

## What the interviewer listens for

- Why Gini (or entropy) instead of misclassification error: the impurity must be strictly concave so that a split that does not change the majority can still have positive gain.
- Greedy top-down search is not optimal: XOR has zero gain at the root for every split, so a greedy tree needs luck or depth to find it. That motivates ensembles.
- The complexity argument for the sorted cumulative-count scan, and why trees need no feature scaling and are invariant to monotone transforms.
- Follow-up: how bagging (random forests, feature subsampling) and boosting (fit the next tree to the gradient of the loss, shrink with a learning rate) change the bias-variance trade-off of this learner, and which hyperparameters here matter most for XGBoost.
