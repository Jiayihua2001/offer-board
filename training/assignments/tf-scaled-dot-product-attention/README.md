# Scaled dot-product attention with a causal mask

The operation at the heart of every transformer. Implement it in NumPy so that it works for a single head and for batched, multi-head inputs through broadcasting.

```
attention(q, k, v, mask=None) -> (output, weights)
```

- `q` has shape `(..., T_q, d_k)`, `k` has shape `(..., T_k, d_k)`, `v` has shape `(..., T_k, d_v)`. The leading dimensions (batch, heads) must broadcast.
- Scores are `q @ k^T / sqrt(d_k)`.
- `mask`, if given, is a boolean array broadcastable to `(..., T_q, T_k)`; `True` means "may attend". Masked positions get weight exactly 0.
- Return the output `(..., T_q, d_v)` and the attention weights `(..., T_q, T_k)`.

Also implement `causal_mask(T)`, returning a boolean `(T, T)` lower-triangular array, so that position `i` can only look at positions `0..i`.

## Example

With `q = k = v = I_3` (the 3x3 identity) and no mask, every query attends most to its own key: the weights have `e^(1/sqrt 3)` on the diagonal, and the output is a blend that still favors the matching row.

## Constraints

- Stable softmax (subtract the row max before `exp`).
- No Python loops over sequence positions.
- With a causal mask, the first row of the weights must be exactly `[1, 0, 0, ...]`.

## What the interviewer listens for

- Why the `1/sqrt(d_k)` scaling exists: without it, dot products grow with `d_k`, softmax saturates, and gradients vanish.
- Why masking happens before the softmax, and why `-inf`-style masking is the right way to get exact zeros.
- The cost: `O(T^2 d)` time and `O(T^2)` memory for the weights, which is what FlashAttention and KV caches address.
- How this generalizes to cross-attention (queries from one sequence, keys and values from another), which is the layer you used in your cross-attention quality estimators.
