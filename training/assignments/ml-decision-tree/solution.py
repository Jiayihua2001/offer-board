import numpy as np


def gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - (p ** 2).sum())


def best_split(X: np.ndarray, y: np.ndarray, min_samples_leaf: int = 1):
    n, d = X.shape
    classes = np.unique(y)
    if len(classes) < 2 or n < 2 * min_samples_leaf:
        return None
    parent = gini(y)
    best = None
    for f in range(d):
        order = np.argsort(X[:, f], kind="stable")
        xs, ys = X[order, f], y[order]
        onehot = (ys[:, None] == classes[None, :]).astype(float)
        left = onehot.cumsum(axis=0)[:-1]                      # class counts of the first i + 1 points
        right = onehot.sum(axis=0) - left
        n_left = np.arange(1, n)
        n_right = n - n_left
        g_left = 1.0 - ((left / n_left[:, None]) ** 2).sum(axis=1)
        g_right = 1.0 - ((right / n_right[:, None]) ** 2).sum(axis=1)
        gain = parent - (n_left * g_left + n_right * g_right) / n
        valid = (xs[1:] != xs[:-1]) & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not valid.any():
            continue
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > 0 and (best is None or gain[i] > best[2]):
            best = (f, float((xs[i] + xs[i + 1]) / 2), float(gain[i]))
    return best


class DecisionTree:
    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _build(self, X, y, depth):
        values, counts = np.unique(y, return_counts=True)
        leaf = {"value": int(values[np.argmax(counts)])}        # first max -> smallest label on ties
        if len(values) == 1 or (self.max_depth is not None and depth >= self.max_depth) \
                or len(y) < 2 * self.min_samples_leaf:
            return leaf
        split = best_split(X, y, self.min_samples_leaf)
        if split is None:
            return leaf
        f, t, _ = split
        m = X[:, f] <= t
        return {"feature": f, "threshold": t,
                "left": self._build(X[m], y[m], depth + 1),
                "right": self._build(X[~m], y[~m], depth + 1)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.root = self._build(np.asarray(X, dtype=float), np.asarray(y), 0)
        return self

    def _predict_node(self, node, X, out, idx):
        if "value" in node:
            out[idx] = node["value"]
            return
        m = X[idx, node["feature"]] <= node["threshold"]
        self._predict_node(node["left"], X, out, idx[m])
        self._predict_node(node["right"], X, out, idx[~m])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape[0], dtype=int)
        self._predict_node(self.root, X, out, np.arange(X.shape[0]))
        return out
