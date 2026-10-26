# Token embeddings and sinusoidal positional encoding

The input side of the transformer. Token ids become vectors through a lookup table, and because attention is permutation-equivariant, a position signal has to be added before the first block or the model cannot tell "dog bites man" from "man bites dog". Implement the lookup, the fixed sinusoidal encoding from the original paper, and the combination with a positional table (learned, as in GPT-2, or sinusoidal).

- `embed(tokens, E) -> np.ndarray`: `tokens` is an integer array of any shape, `E` is `(vocab, d)`. Return `tokens.shape + (d,)`, the rows of `E` picked by the ids.
- `sinusoidal_pe(T, d) -> np.ndarray` of shape `(T, d)` with `d` even, the Vaswani et al. formula: `PE[pos, 2i] = sin(pos / 10000^(2i/d))` and `PE[pos, 2i+1] = cos(pos / 10000^(2i/d))`.
- `embed_with_positions(tokens, E, P) -> np.ndarray`: `tokens` is `(B, T)`, `P` is a positional table `(T_max, d)` with `T_max >= T`. Return `E[tokens] + P[:T]`, shape `(B, T, d)`.

## Example

```
sinusoidal_pe(3, 4) ->
  [[0.        1.        0.        1.       ]
   [0.841471  0.540302  0.01      0.99995  ]
   [0.909297 -0.416147  0.019999  0.9998   ]]
```

Column pair 0 rotates by 1 radian per position; column pair 1 rotates by `1/100` radian per position (`10000^(-2/4)`).

## Constraints

- No Python loops over positions or dimensions; build the angle table with broadcasting.
- `embed` must work for 1-D and 2-D `tokens`.
- `d` is even; you may assume it.

## What the interviewer listens for

- Why positions have to be injected at all: attention and the feed-forward block treat the sequence as a set.
- Why sinusoids: every frequency pair is a 2-D rotation, so `PE(pos + k)` is a fixed linear function of `PE(pos)` for each `k`, which gives the model a handle on relative offsets. This is the idea RoPE later applies to the queries and keys instead of the input.
- The cost of the alternatives: a learned table is simple but fixed to `T_max`; sinusoids extrapolate in principle but models still degrade past the trained length.
- Follow-up: why the original paper scales the embedding by `sqrt(d)` before adding `PE`, and why GPT-2 does not need to (its learned table and embedding are on the same scale).
