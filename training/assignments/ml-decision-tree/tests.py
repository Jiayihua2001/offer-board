import numpy as np

from solution import DecisionTree, best_split, gini


def depth(node):
    if "value" in node:
        return 0
    return 1 + max(depth(node["left"]), depth(node["right"]))


def leaf_sizes(node, X):
    """Number of training rows routed to each leaf."""
    if "value" in node:
        return [len(X)]
    m = X[:, node["feature"]] <= node["threshold"]
    return leaf_sizes(node["left"], X[m]) + leaf_sizes(node["right"], X[~m])


def rectangle_data(seed=1, n=300):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    return X, y


def test_gini_values():
    assert np.isclose(gini(np.array([1, 1, 1, 1])), 0.0)
    assert np.isclose(gini(np.array([0, 1, 0, 1])), 0.5)
    assert np.isclose(gini(np.array([0, 1, 2])), 2.0 / 3.0)
    assert np.isclose(gini(np.array([7, 7, 7, 3])), 0.375)


def test_best_split_finds_midpoint_and_respects_min_leaf():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    f, t, gain = best_split(X, y)
    assert f == 0 and np.isclose(t, 6.5) and np.isclose(gain, 0.5)
    assert best_split(X, y, min_samples_leaf=4) is None, "no split can leave 4 samples on both sides"
    assert best_split(X, np.zeros(6, int)) is None, "a pure node has no split with positive gain"
    assert best_split(np.ones((6, 1)), y) is None, "a constant feature has no threshold"


def test_best_split_ignores_a_noise_feature():
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1, 1, size=200)
    X = np.column_stack([x0, rng.uniform(-1, 1, size=200)])
    y = (x0 > 0.3).astype(int)
    f, t, gain = best_split(X, y)
    below, above = x0[x0 <= 0.3].max(), x0[x0 > 0.3].min()
    assert f == 0
    assert np.isclose(t, (below + above) / 2), f"threshold {t} should be the midpoint between {below} and {above}"
    assert np.isclose(gain, gini(y)), "a perfect split removes all impurity"


def test_fits_rectangle_rule_exactly_with_depth_two():
    X, y = rectangle_data()
    tree = DecisionTree(max_depth=2).fit(X, y)
    assert isinstance(tree.root, dict) and "feature" in tree.root
    assert depth(tree.root) == 2
    pred = tree.predict(X)
    assert pred.shape == (300,)
    assert np.mean(pred == y) == 1.0
    X_test, y_test = rectangle_data(seed=2, n=1000)
    assert np.mean(tree.predict(X_test) == y_test) >= 0.98


def test_depth_limit_is_respected():
    X, y = rectangle_data(seed=3)
    for d in [0, 1, 2, 3]:
        tree = DecisionTree(max_depth=d).fit(X, y)
        assert depth(tree.root) <= d, f"max_depth={d} but tree has depth {depth(tree.root)}"
    stump = DecisionTree(max_depth=0).fit(X, y)
    assert stump.root == {"value": 0}, "depth 0 is a single leaf predicting the majority class"
    acc1 = np.mean(DecisionTree(max_depth=1).fit(X, y).predict(X) == y)
    assert 0.7 < acc1 < 1.0, "one split cannot express an AND of two features"


def test_min_samples_leaf_is_respected():
    X, y = rectangle_data(seed=4)
    y = y.copy()
    y[::7] = 1 - y[::7]                       # label noise: an unconstrained tree grows tiny leaves
    free = DecisionTree().fit(X, y)
    assert min(leaf_sizes(free.root, X)) < 10
    assert np.mean(free.predict(X) == y) == 1.0, "an unconstrained tree fits the training set"
    tree = DecisionTree(min_samples_leaf=10).fit(X, y)
    sizes = leaf_sizes(tree.root, X)
    assert min(sizes) >= 10, f"smallest leaf has {min(sizes)} samples"
    assert sum(sizes) == 300


def test_three_blobs_multiclass():
    rng = np.random.default_rng(5)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
    X = np.vstack([c + rng.normal(size=(60, 2)) for c in centers])
    y = np.repeat([3, 1, 2], 60)                  # labels not 0..C-1 on purpose
    tree = DecisionTree(max_depth=6, min_samples_leaf=2).fit(X, y)
    assert np.mean(tree.predict(X) == y) >= 0.98
    X_test = np.vstack([c + rng.normal(size=(200, 2)) for c in centers])
    y_test = np.repeat([3, 1, 2], 200)
    assert np.mean(tree.predict(X_test) == y_test) >= 0.95
    assert set(np.unique(tree.predict(X_test))) <= {1, 2, 3}
