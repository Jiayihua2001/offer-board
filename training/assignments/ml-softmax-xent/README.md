# Stable softmax and cross-entropy, with gradients

This is the most common "write it from scratch" warm-up in ML coding rounds, and the place where numerical stability questions get asked. Implement four functions in NumPy, vectorized over a batch.

- `softmax(z)`: `z` has shape `(N, C)`. Return probabilities of shape `(N, C)` that sum to 1 per row. Must not overflow for logits like `1000.0` or underflow to all zeros for `-1000.0`.
- `log_softmax(z)`: the log of the above, computed without taking `log` of a tiny number.
- `cross_entropy(z, y)`: mean negative log-likelihood of integer labels `y` of shape `(N,)` under the logits `z`.
- `cross_entropy_grad(z, y)`: gradient of `cross_entropy` with respect to `z`, shape `(N, C)`.

## Example

```
z = np.array([[2.0, 1.0, 0.1]])
softmax(z)        -> [[0.659, 0.242, 0.099]]
cross_entropy(z, np.array([0]))  -> 0.417
```

## Constraints

- No loops over rows; use broadcasting.
- `softmax(z + c)` must equal `softmax(z)` for any per-row constant `c`.
- The gradient is checked against finite differences, so get the `1/N` factor right.

## What the interviewer listens for

- Why the max-subtraction trick is exact rather than an approximation.
- Why `log_softmax` should be `z - logsumexp(z)` and not `np.log(softmax(z))`.
- The clean form of the gradient, `(p - onehot) / N`, and why that makes softmax plus cross-entropy such a well-behaved output layer (the gradient never saturates the way a sigmoid plus squared error does).
- Follow-up: how label smoothing changes the target distribution and the gradient.
