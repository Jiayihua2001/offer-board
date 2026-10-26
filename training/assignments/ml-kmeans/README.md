# k-means with k-means++ initialization

k-means is the clustering question because it has a crisp objective, a two-step algorithm that provably never gets worse, and an initialization story (k-means++) that separates people who have used `sklearn` from people who understand why random init fails. Implement Lloyd's algorithm in NumPy with k-means++ seeding and return the objective per iteration so the monotonicity can be checked.

- `kmeans_pp_init(X, k, rng) -> centroids` of shape `(k, D)`: the first centroid is a uniformly random row of `X`; each next one is a row drawn with probability proportional to its squared distance to the nearest centroid already chosen. Every centroid is an actual row of `X`.
- `assign(X, centroids) -> labels` of shape `(N,)`: index of the nearest centroid (squared Euclidean distance).
- `inertia(X, centroids, labels) -> float`: sum over points of the squared distance to their assigned centroid.
- `kmeans(X, k, rng, n_iter=100, tol=1e-6) -> (centroids, labels, history)`: k-means++ init, then repeat: assign, append the current inertia to `history`, move each centroid to the mean of its points (a cluster that lost all its points keeps its centroid). Stop early when no centroid moves by more than `tol`. `history` is a list or array with at most `n_iter` entries; the returned `labels` and `centroids` are consistent with `history[-1]`.

## Example

```
X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], float)
C = np.array([[0, 0], [10, 10]], float)
assign(X, C)          -> [0, 0, 1, 1]
inertia(X, C, assign(X, C))   -> 2.0

centroids, labels, history = kmeans(X, 2, np.random.default_rng(0))
centroids   -> [[0, 0.5], [10, 10.5]] in some order
history[-1] -> 1.0
```

## Constraints

- No loops over points: distances are one broadcasted expression or the `||a||^2 - 2 a.b + ||b||^2` expansion; the means are one `np.add.at` or a boolean-mask sum per cluster (a loop over `k` is fine).
- `history` must be non-increasing to within floating-point error.
- Each iteration is `O(N k D)`.

## What the interviewer listens for

- Why the objective never increases: assignment minimizes it over labels for fixed centroids, the mean minimizes it over centroids for fixed labels (the mean is the minimizer of the sum of squared distances).
- Why k-means++: random init can put two seeds in one true cluster and leave another cluster split or empty; `D^2` sampling spreads the seeds and bounds the expected objective within `O(log k)` of optimal.
- The failure modes: only finds a local optimum (so restart and keep the lowest inertia), assumes spherical clusters of similar size, and `k` has to come from somewhere (elbow, silhouette, or the application).
- Follow-up: how you would adapt this to a 1 M-point dataset (mini-batch k-means) or to a kernel or non-Euclidean distance (k-medoids: the mean is no longer the minimizer).
