# Multi-head attention

One attention head computes a single weighted average per query. Multi-head attention runs `H` of them in parallel on `H` slices of the model dimension, so different heads can look for different things (the previous token, the matching bracket, the subject of the verb), then mixes them back with an output projection. This is the layer in every block of GPT, and the one whose reshapes people get wrong on a whiteboard.

`attention(q, k, v, mask)` and `causal_mask(T)` from the previous assignment are provided in the starter, fully implemented. Write three functions:

- `split_heads(x, n_heads)`: `(B, T, d_model) -> (B, H, T, d_head)` with `d_head = d_model / H`. Head `h` gets columns `h*d_head` to `(h+1)*d_head` of every token.
- `merge_heads(x)`: the exact inverse, `(B, H, T, d_head) -> (B, T, d_model)`.
- `mha(x, Wq, Wk, Wv, Wo, n_heads, mask=None) -> (B, T, d_model)`: project `x` with `Wq`, `Wk`, `Wv` (each `(d_model, d_model)`, no biases), split into heads, run scaled dot-product attention per head with the scale `1/sqrt(d_head)`, merge, and apply `Wo` `(d_model, d_model)`. `mask` is boolean, `(T, T)` or anything broadcastable to `(B, H, T, T)`, `True` = may attend.

## Example

```
x: (2, 5, 8), n_heads = 2            -> d_head = 4
split_heads(x, 2)[1, 0, 3]           == x[1, 3, 0:4]
split_heads(x, 2)[1, 1, 3]           == x[1, 3, 4:8]
merge_heads(split_heads(x, 2))       == x
mha(x, Wq, Wk, Wv, Wo, 2).shape      == (2, 5, 8)
```

With `n_heads = 1`, `mha` is exactly `attention(x @ Wq, x @ Wk, x @ Wv)[0] @ Wo`.

## Constraints

- No loop over heads: one reshape and transpose each way, and one call to `attention`.
- Every head attends with the scale of its own dimension `d_head`, not `d_model`.
- Without a mask, permuting the positions of `x` must permute the rows of the output in the same way.

## What the interviewer listens for

- Why the split is `reshape` then `transpose` and not a direct reshape to `(B, H, T, d_head)`: the memory layout is tokens-major, so the direct reshape hands each head a mix of different tokens.
- The cost: the projections are `O(T d^2)`, attention is `O(T^2 d)` in total (each head is `O(T^2 d_head)`), and `H` heads cost the same as one wide head. Why heads at all then: `H` separate softmaxes can attend to `H` different places; one softmax averages them.
- The failure mode: scaling by `sqrt(d_model)` instead of `sqrt(d_head)` makes the attention too flat, and forgetting `Wo` leaves the heads unable to talk to each other.
- Follow-up: multi-query and grouped-query attention share `K` and `V` across heads to shrink the KV cache; what changes in `split_heads` to support that.
