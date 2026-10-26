# k-nearest neighbors, vectorized

k-NN has no training step, so the whole question is about computing pairwise distances without a Python loop and voting without one either. It is the warm-up that shows whether you reach for broadcasting and the `||a||^2 - 2 a.b + ||b||^2` expansion, which is also how every retrieval system computes similarities at scale.

- `pairwise_sq_dists(A, B)`: squared Euclidean distances between every row of `A` (shape `(n, d)`) and every row of `B` (shape `(m, d)`), returned as `(n, m)`. Use the expansion with a single matrix product. Clip at 0 so no entry is negative.
- `knn_predict(X_train, y_train, X_test, k)`: for each test row, take the `k` nearest training rows and return the majority label, shape `(m,)`. Labels are arbitrary integers (they may be negative or non-contiguous). Ties between labels go to the smallest label.

## Example

```
X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
y_train = np.array([5, 5, 2, 2])
knn_predict(X_train, y_train, np.array([[0.9], [2.4]]), k=1)   -> [5, 2]
knn_predict(X_train, y_train, np.array([[1.4]]), k=2)          -> [2]   (neighbors 1.0 and 2.0: labels 5 and 2 tie, smallest wins)
knn_predict(X_train, y_train, np.array([[1.4]]), k=4)          -> [2]
```

## Constraints

- No loops over samples anywhere: distances are one expression, neighbor selection is one `argpartition` or `argsort` call, voting is one-hot sums.
- `pairwise_sq_dists(A, A)` must have an exactly zero (not slightly negative) diagonal.
- `O(n m d)` time and `O(n m)` memory for the distance matrix; be ready to say what you would do when that matrix does not fit.

## What the interviewer listens for

- The expansion and why it is faster than broadcasting `(A[:, None] - B[None]) ** 2` (one BLAS matmul versus materializing an `(n, m, d)` tensor), and its numerical cost (cancellation when `||a||^2` and `||b||^2` are large and close, which is where the clipping comes from).
- `argpartition` versus `argsort`: `O(n)` versus `O(n log n)` per query when you only need the top `k`.
- Why the vote must be deterministic under ties and how distance-weighted voting changes the picture.
- Follow-up: the curse of dimensionality (distances concentrate as `d` grows), why you standardize features first, and how an approximate index (LSH, HNSW, FAISS IVF) trades exactness for `O(log n)` queries.
