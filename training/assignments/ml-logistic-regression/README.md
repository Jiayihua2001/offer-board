# Logistic regression with L2, stable and from scratch

Logistic regression is the model interviewers use to check three things at once: that you can derive the gradient of a likelihood, that you know why naive `log(sigmoid(z))` blows up, and that you understand what regularization does to a separable dataset. The model has weights `w` of shape `(D,)` and a scalar bias `b`; the loss is the mean binary cross-entropy plus `(l2 / 2) * ||w||^2`. The bias is not regularized.

Implement, vectorized over the batch:

- `sigmoid(z)`: elementwise, for any array `z`. Must not overflow: the tests run it under `np.errstate(over="raise")` with `z = +-1000`, so a two-branch `np.where` that still evaluates `np.exp(-z)` on both branches does not pass.
- `log_loss(X, y, w, b, l2=0.0)`: mean over `N` samples of `-(y log p + (1 - y) log(1 - p))` with `p = sigmoid(X @ w + b)`, plus `(l2 / 2) * ||w||^2`. Computed without ever taking `log` of a probability (use the softplus form). `X` has shape `(N, D)`, `y` is 0/1 of shape `(N,)`.
- `grad(X, y, w, b, l2=0.0)`: returns `(dw, db)`, the gradient of `log_loss` with respect to `w` (shape `(D,)`) and `b` (a float).
- `fit(X, y, lr=0.1, n_iter=1000, l2=0.0)`: full-batch gradient descent from `w = 0, b = 0`. Returns `(w, b)`.
- `predict_proba(X, w, b)`: `P(y = 1)` of shape `(N,)`.

## Example

```
X = np.array([[1.0], [-1.0]]);  y = np.array([1, 0])
log_loss(X, y, w=np.array([0.0]), b=0.0)      -> 0.693   (log 2: a coin flip)
log_loss(X, y, w=np.array([1000.0]), b=0.0)   -> ~0.0    (finite, no warning)
log_loss(X, 1 - y, w=np.array([1000.0]), b=0.0) -> 1000.0 (confidently wrong costs |z|)
grad(X, y, w=np.array([0.0]), b=0.0)          -> (array([-0.5]), 0.0)
```

## Constraints

- No loops over samples. The only loop is over GD iterations.
- `log_loss` and `grad` must stay finite for logits of magnitude 1000 in either direction.
- The gradient is checked against central finite differences of your `log_loss`, including the L2 term.

## What the interviewer listens for

- The derivation: `d/dz [softplus(z) - y z] = sigmoid(z) - y`, so the gradient is `X^T (p - y) / N`, the same "prediction minus target" form as linear regression and softmax.
- Why `log(1 + exp(z))` needs `logaddexp` and why `sigmoid` needs the `exp(-|z|)` trick: it is not about accuracy, it is about `inf` and `nan` propagating into every weight.
- What happens on linearly separable data without L2 (the weights grow without bound and the loss approaches 0 but never reaches it) and how L2 fixes it. A test checks that larger `l2` gives a smaller `||w||`.
- Follow-up: why the bias is left unregularized, and how you would pick `l2` (cross-validation on log-loss, not accuracy).
