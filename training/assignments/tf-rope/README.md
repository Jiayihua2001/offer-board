# Rotary position embeddings

Llama, Mistral, Qwen and most models since 2022 do not add a position vector to the input. Instead they rotate each query and key inside the attention layer by an angle proportional to its position, so the dot product between a query at position `m` and a key at position `n` depends on `m - n` only. That is RoPE: relative position encoding obtained with an absolute operation, no extra parameters, and it composes with a KV cache because each key is rotated once and never touched again.

```
rope(x, positions, base=10000.0) -> np.ndarray, same shape as x
```

- `x` has shape `(..., T, d)` with `d` even; `positions` is an integer array of shape `(T,)`.
- Dimension pairs `(2i, 2i+1)` for `i = 0 .. d/2 - 1` are rotated by the angle `positions[t] * theta_i` with `theta_i = base^(-2i/d)`: `x'[2i] = x[2i] cos - x[2i+1] sin` and `x'[2i+1] = x[2i] sin + x[2i+1] cos`.
- Leading dims (batch, heads) broadcast; `positions` applies along the `T` axis.

The relative-position property: for rotated `q_m = rope(q, m)` and `k_n = rope(k, n)`, `q_m . k_n` depends only on `m - n` (and on `q`, `k`), and rotations preserve norms.

## Example

```
x = [[1, 0, 1, 0]], positions = [2], d = 4      (pairs (0, 1) and (2, 3); theta = [1, 0.01])
rope(x, [2]) -> [[cos 2, sin 2, cos 0.02, sin 0.02]] = [[-0.4161, 0.9093, 0.9998, 0.0200]]
rope(x, [0]) -> x
```

## Constraints

- No loops over positions or pairs; one `cos`/`sin` table of shape `(T, d/2)` and broadcasting.
- `rope` must not modify `x` in place.
- Interleaved pairs `(2i, 2i+1)` as above (the original formulation), not the "rotate-half" layout Hugging Face uses; the tests check the exact values.

## What the interviewer listens for

- Why a rotation: `R(m)^T R(n) = R(n - m)`, so the score `(R(m) q) . (R(n) k) = q . R(n - m) k` is a function of the offset, with no learned parameters and no change to `V`.
- Where it is applied and the cost: to `q` and `k` after the projections and before the scores, `O(T d)` per layer, negligible next to attention; the `cos`/`sin` tables are precomputed once per sequence length.
- The failure mode: positions past the training length put the low-frequency pairs at angles never seen, which is why context extension methods (position interpolation, NTK-aware scaling, YaRN) rescale `base` or the positions rather than retrain.
- Follow-up: the role of `base` (larger means slower rotation of the low-frequency pairs and longer usable context) and how the "rotate-half" layout is just a permutation of the same pairs.
