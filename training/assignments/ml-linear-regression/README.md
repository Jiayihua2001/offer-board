# Linear regression: closed form and gradient descent

The first thing an ML coding round checks is whether you can go from "minimize a squared error" to working code two different ways: solve it exactly with linear algebra, and solve it iteratively with gradient descent. The exact route tests that you know the normal equations and their failure mode; the iterative route is the template every later model in this track follows (loss, gradient, update, loss history).

Implement three functions in NumPy. Weights are a single vector `w` of shape `(D + 1,)` whose first entry is the intercept, so `predict(X, w) = w[0] + X @ w[1:]`.

- `fit_closed_form(X, y)`: `X` has shape `(N, D)`, `y` has shape `(N,)`. Build the design matrix with a bias column and solve the least-squares problem. Return `w` of shape `(D + 1,)`. One test duplicates a feature column, so the rank-deficient case must not crash: `np.linalg.lstsq` handles it, `np.linalg.solve` on the normal equations needs a small ridge term.
- `fit_gd(X, y, lr=0.1, n_iter=500)`: full-batch gradient descent on the mean squared error `mean((pred - y)^2)` starting from `w = 0`. Return `(w, losses)` where `losses` is a list or array of length `n_iter` with the MSE recorded once per iteration.
- `predict(X, w)`: predictions of shape `(N,)`.

## Example

```
X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])          # y = 1 + 2 x
fit_closed_form(X, y)   -> [1.0, 2.0]
predict(np.array([[4.0]]), [1.0, 2.0])  -> [9.0]

w, losses = fit_gd(X, y, lr=0.1, n_iter=500)
w            -> close to [1.0, 2.0]
losses[0]    -> 21.0   (the loss at w = 0 is mean(y^2))
```

## Constraints

- No Python loops over samples or features; the only loop is over GD iterations.
- The loss history must be non-increasing when `lr` is small enough (full-batch GD on a convex quadratic never goes up).
- Closed form is `O(N D^2 + D^3)`; one GD step is `O(N D)`. Be ready to say when each wins.

## What the interviewer listens for

- The derivation: set the gradient of `||Xw - y||^2` to zero and you get `X^T X w = X^T y`. Why `lstsq` (a QR or SVD solve) is preferred over forming `X^T X`, which squares the condition number.
- What happens with collinear features: `X^T X` is singular, the solution is not unique, and either the minimum-norm solution (`lstsq`) or a ridge penalty picks one.
- The GD trade-off: no `D^3` term and it streams over data, but you need a learning rate below `2 / lambda_max` of the Hessian `(2 / N) X^T X`, which is why features get standardized first.
- Follow-up: how do the gradient and the closed form change when you add an L2 penalty `lam * ||w[1:]||^2`, and why the bias is usually left out of it.
