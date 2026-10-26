# A pre-norm decoder block

One GPT block: attention, then a feed-forward network, each wrapped in a residual connection with a LayerNorm in front of it. Stack `L` of these and you have the body of the model. The arrangement (norm before the sub-layer, "pre-norm") is what GPT-2, Llama and nearly every model since use, because it keeps the residual stream an identity path from the embedding to the logits and trains without warm-up tricks.

Everything below the block is provided in the starter, fully implemented: `causal_mask`, `attention`, `split_heads`, `merge_heads`, `mha`, `layer_norm`, `gelu`, `ffn`. Write one function:

```
block(x, p, n_heads, mask=None) -> (B, T, d_model)
    h   = x + mha(layer_norm(x, p["ln1_g"], p["ln1_b"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask)
    out = h + ffn(layer_norm(h, p["ln2_g"], p["ln2_b"]), p["W1"], p["b1"], p["W2"], p["b2"])
```

`p` is a dict with exactly those keys: `ln1_g`, `ln1_b`, `ln2_g`, `ln2_b` of shape `(d,)`; `Wq`, `Wk`, `Wv`, `Wo` of shape `(d, d)`; `W1` `(d, d_ff)`, `b1` `(d_ff,)`, `W2` `(d_ff, d)`, `b2` `(d,)`.

## Example

```
x: (2, 6, 16), n_heads = 4, mask = causal_mask(6)
block(x, p, 4, mask).shape -> (2, 6, 16)
with p["Wo"] = 0, p["W2"] = 0, p["b2"] = 0:  block(x, p, 4, mask) == x      (only the residual path is left)
```

## Constraints

- The order matters and is tested: norm, sub-layer, add; twice.
- The causal mask goes to attention only; with it, changing token `t` must not change any output before `t`.
- No loops over batch or positions beyond what the provided helpers do.

## What the interviewer listens for

- Pre-norm versus post-norm: in post-norm (`LN(x + f(x))`) the residual stream is renormalized every block, so gradients to early layers shrink with depth and training needs careful warm-up; pre-norm keeps a clean skip path and is the default now, at the price of a final LayerNorm before the logits.
- Why attention is the only place positions interact: the FFN and the norms act per token, so causality and permutation equivariance are decided entirely by the mask.
- The failure mode: initializing `Wo` and `W2` large makes each block perturb the residual stream a lot; GPT-2 scales them by `1/sqrt(2L)` so the sum over `L` blocks stays `O(1)`.
- Follow-up: where dropout goes (after attention weights, after each sub-layer output) and what parallel blocks (attention and FFN side by side, as in GPT-J) change.
