# Batch norm and inverted dropout, train and eval modes

Two layers whose entire difficulty is the difference between training and inference. Interviewers ask for them because the bugs are famous: forgetting to switch to running statistics at eval time, updating the running variance with the wrong estimator, or scaling dropout at test time instead of training time. Implement the forward passes only; the backward pass of batch norm is a separate (and harder) question.

- `batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5) -> (out, running_mean, running_var)`. `x` has shape `(N, D)`; `gamma`, `beta`, `running_mean`, `running_var` have shape `(D,)`. In training, normalize each feature with the batch mean and the biased batch variance, then return updated running statistics `running = (1 - momentum) * running + momentum * batch_stat`, where the variance fed into the running estimate is the unbiased one (`N / (N - 1)` times the biased one), as PyTorch does. In eval, normalize with the running statistics and return them unchanged. Never modify the input arrays in place.
- `dropout_forward(x, p, training, rng) -> (out, mask)`. Inverted dropout with drop probability `0 <= p < 1`: in training each element is kept with probability `1 - p` (draw the mask with `rng.random(x.shape)`), and the kept elements are scaled by `1 / (1 - p)`. In eval `out` is `x` and `mask` is all ones. `mask` has the shape of `x`.

## Example

```
x = np.array([[1.0, 10.0], [3.0, 30.0]])
out, rm, rv = batchnorm_forward(x, np.ones(2), np.zeros(2), np.zeros(2), np.ones(2), training=True)
out  -> [[-1, -1], [1, 1]]        (each column standardized; eps makes it 0.99999...)
rm   -> [0.2, 2.0]                (0.9 * 0 + 0.1 * mean)
rv   -> [1.1, 20.9]               (0.9 * 1 + 0.1 * unbiased var: 2 and 200)

out, mask = dropout_forward(np.ones((4, 4)), p=0.5, training=True, rng=np.random.default_rng(0))
out  -> entries are 0 or 2, roughly half each
```

## Constraints

- Vectorized over the batch and the features; no loops.
- `eps` must sit inside the square root so a constant feature (variance 0) gives a finite output equal to `beta`.
- The returned running statistics must be new arrays: a test checks the inputs are untouched.

## What the interviewer listens for

- Why batch norm needs two modes at all: at inference there may be no batch (or a batch of one), so the layer must become a fixed affine map using population estimates gathered during training.
- Why the running estimate uses the unbiased variance while the forward pass uses the biased one, and why neither choice matters much for large batches but both matter for a batch of 2.
- Why dropout scales by `1 / (1 - p)` at training time: it keeps the expected pre-activation the same in both modes, so the eval path is a no-op and the network sees the same scale it trained on.
- Follow-up: what breaks with batch norm at batch size 1 or with strongly correlated samples in a batch, and why layer norm replaced it in transformers.
