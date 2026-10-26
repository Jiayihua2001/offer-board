# Backprop through attention

Autograd hides this, which is exactly why interviewers ask for it: derive and implement the backward pass of scaled dot-product attention by hand. This is also the computation FlashAttention has to recompute block by block, and the reason it stores the row-wise logsumexp instead of the weights.

The forward pass (provided in the starter as `attention(q, k, v, mask)`, which returns `(out, w)`) is

```
s   = q @ k^T / sqrt(d_k)        (..., T_q, T_k)
w   = softmax(s, axis=-1)        (..., T_q, T_k), masked entries exactly 0
out = w @ v                      (..., T_q, d_v)
```

Write two functions:

- `softmax_backward(w, d_w) -> d_s`: given the softmax output `w` and the gradient `d_w` of the loss with respect to it, return the gradient with respect to the softmax input, same shape. Row-wise: `d_s = w * (d_w - sum(w * d_w))`.
- `attention_backward(q, k, v, w, d_out) -> (d_q, d_k, d_v)`: given the inputs, the forward weights `w` and `d_out = dL/d out`, return the gradients with the shapes of `q`, `k`, `v`. Leading batch and head dims must broadcast as in the forward pass.

## Example

```
q = [[1, 0]], k = [[1, 0], [0, 1]], v = [[1, 0], [0, 1]]        (T_q = 1, T_k = 2, d = 2)
w = softmax([1/sqrt2, 0]) = [0.6698, 0.3302]
d_out = [[1, 0]]
d_v = w^T @ d_out = [[0.6698, 0], [0.3302, 0]]
d_w = d_out @ v^T = [[1, 0]]
d_s = w * (d_w - 0.6698) = [[0.2212, -0.2212]]
d_q = d_s @ k / sqrt2 = [[0.1564, -0.1564]]
d_k = d_s^T @ q / sqrt2 = [[0.1564, 0], [-0.1564, 0]]
```

## Constraints

- No loops over positions or batch entries; everything is `matmul`, `swapaxes` and broadcasting.
- The tests compare against central finite differences with tolerance `1e-6`, on masked and unmasked inputs and on batched `(B, H, T, d)` shapes.
- Do not recompute the forward pass inside the backward: use the `w` you are given.

## What the interviewer listens for

- The softmax Jacobian `diag(w) - w w^T` and why the vector-Jacobian product reduces to one elementwise line; the rows of `d_s` sum to zero, which is why `d_k` summed over keys is zero.
- The cost: the backward is about twice the forward (`d_q`, `d_k`, `d_v` are three `(T, T, d)`-sized matmuls) and it needs the `T x T` weights, so memory is `O(T^2)` per head unless you recompute them, which is FlashAttention's trade.
- The failure mode: saturated softmax. If the scores are large, `w` is one-hot, `d_s` is zero and nothing upstream learns; that is the `1/sqrt(d_k)` scaling story told from the gradient side.
- Follow-up: how the masked positions behave in the backward (zero weight means zero gradient, no special casing) and what changes for dropout on the attention weights.
