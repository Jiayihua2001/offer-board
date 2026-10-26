# Two-layer MLP: forward and backward by hand

Manual backprop through one hidden layer is the canonical "do you actually understand autograd" question. The forward pass is three lines; the backward pass is where people lose the `1/N`, forget to sum the bias gradient over the batch, or multiply by the ReLU mask in the wrong place. The tests catch all three with a finite-difference check.

The network is `X (N, d_in) -> W1 (d_in, H), b1 (H,) -> ReLU -> W2 (H, C), b2 (C,) -> softmax cross-entropy` on integer labels `y` of shape `(N,)`. Parameters live in a dict with keys `W1, b1, W2, b2`. The starter provides `init_params` and a stable `log_softmax`; do not change them, the tests depend on the initialization being deterministic.

Implement:

- `forward(X, params) -> (logits, cache)`: `logits` has shape `(N, C)`; `cache` is whatever your backward pass needs.
- `loss_and_grads(X, y, params) -> (loss, grads)`: the mean cross-entropy as a float, and a dict with the same keys and shapes as `params`.
- `train(X, y, params, lr=0.5, n_iter=300) -> (params, losses)`: full-batch gradient descent; `losses` has one entry per iteration. Return a new dict (do not mutate the input in place).
- `predict(X, params) -> labels` of shape `(N,)`.

## Example

```
params = init_params(2, 4, 3, np.random.default_rng(0))
logits, cache = forward(X, params)          # X: (5, 2)  ->  logits: (5, 3)

zero = {k: np.zeros_like(v) for k, v in params.items()}
loss_and_grads(X, y, zero)[0]               -> log(3) = 1.0986  (uniform prediction)
```

## Constraints

- No loops over samples; every gradient is one matrix product or one sum over the batch axis.
- Gradients are checked against central differences with `atol=1e-6`, so use the stable `log_softmax` for the loss and keep the `1/N`.
- A test trains on a nonlinear 2-D dataset (points inside versus outside a circle) and expects at least 95% training accuracy, which a correct gradient reaches in 300 steps.

## What the interviewer listens for

- The derivation in the order the chain rule applies: `dlogits = (p - onehot) / N`, then each linear layer gives `dW = inp.T @ dout`, `db = dout.sum(0)`, `dinp = dout @ W.T`, and ReLU just masks.
- Why the bias gradient is a sum over the batch (the same `b` is added to every row, so every row contributes) and why `dW` is `inp.T @ dout` and not the other way around (match the shapes).
- Dead ReLUs: a unit whose pre-activation is negative on every sample gets a zero gradient for its whole input column and never recovers. A test pins this.
- Follow-up: what changes with a second hidden layer, and how you would write the backward pass as a loop over layers once you have the per-layer rule.
