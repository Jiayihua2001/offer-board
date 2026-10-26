# PCA via SVD

PCA is asked to see whether you connect three views of the same thing: maximum-variance directions, eigenvectors of the covariance, and the singular vectors of the centered data. The SVD route is the one to implement (it never forms the covariance matrix and is numerically the best behaved); the eigendecomposition is what the tests use as an independent reference.

Implement, with `X` of shape `(N, D)`:

- `pca_fit(X, n_components) -> model`: a dict with `"mean"` (shape `(D,)`), `"components"` (shape `(n_components, D)`, rows are unit-length principal directions sorted by decreasing variance), `"explained_variance"` (shape `(n_components,)`, the variance of the data along each component, with the `N - 1` denominator) and `"explained_variance_ratio"` (each component's variance divided by the total variance of `X` across all `D` dimensions).
- `pca_transform(X, model) -> Z` of shape `(N, n_components)`: center with the stored mean and project.
- `pca_inverse_transform(Z, model) -> X_hat` of shape `(N, D)`: map scores back to the original space.

Component signs are arbitrary (an eigenvector times -1 is still an eigenvector); the tests compare up to sign.

## Example

```
X = np.array([[-2, -1], [-1, -0.5], [1, 0.5], [2, 1]], float)   # exactly on a line of slope 1/2
m = pca_fit(X, 1)
m["components"]                -> [[0.894, 0.447]] (or its negative)
m["explained_variance_ratio"]  -> [1.0]
pca_inverse_transform(pca_transform(X, m), m)  -> X (exact reconstruction)
```

## Constraints

- No loops; `np.linalg.svd` with `full_matrices=False` and matrix products only.
- Do not form `X.T @ X` for the fit (that squares the condition number); the `(N - 1)` convention must match the covariance exactly.
- SVD cost is `O(N D min(N, D))`; know when a randomized or truncated SVD is the right call.

## What the interviewer listens for

- The link between the three views: `X_c = U S V^T` gives `X_c^T X_c / (N - 1) = V (S^2 / (N - 1)) V^T`, so the right singular vectors are the covariance eigenvectors and the eigenvalues are `S^2 / (N - 1)`.
- Why you must center (otherwise the first component points at the mean) and when you should also standardize (features on different scales, otherwise the biggest-variance feature wins by units alone).
- Reconstruction error with `k` components equals the sum of the dropped eigenvalues; this is the Eckart-Young statement that PCA is the best rank-`k` approximation in the Frobenius norm.
- Follow-up: how you would run PCA on 10 M rows that do not fit in memory (incremental or randomized SVD, or the covariance of streamed mini-batches), and how it relates to the autoencoder with a linear decoder.
