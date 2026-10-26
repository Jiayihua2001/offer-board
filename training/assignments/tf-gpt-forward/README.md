# The full GPT forward pass and loss

Put the pieces together: token ids in, next-token logits out, and the training objective. This is the whole model of GPT-2 in about fifteen lines once the block exists, and it is the function every later assignment (training, sampling, the KV cache) builds on.

The starter provides, fully implemented: `causal_mask`, `attention`, `split_heads`, `merge_heads`, `mha`, `layer_norm`, `gelu`, `ffn`, `block`, and `init_params(vocab, d_model, n_heads, n_layers, T, rng)`, which returns

```
params = {"wte": (vocab, d_model), "wpe": (T_max, d_model), "blocks": [block params, ...],
          "lnf_g": (d_model,), "lnf_b": (d_model,), "n_heads": int}
```

Write two functions:

- `gpt_forward(tokens, params) -> logits` of shape `(B, T, vocab)` for integer `tokens` of shape `(B, T)`: `x = wte[tokens] + wpe[:T]`, then every block with a causal mask, then the final LayerNorm, then the tied output projection `x @ wte.T` (the same matrix as the input embedding, GPT-2 style).
- `lm_loss(logits, targets) -> float`: mean cross-entropy over all `B * T` positions of integer `targets` `(B, T)` under `logits`, computed with a numerically stable log-softmax.

## Example

```
params = init_params(vocab=50, d_model=32, n_heads=4, n_layers=2, T=16, rng)
tokens = rng.integers(0, 50, size=(3, 10))
gpt_forward(tokens, params).shape       -> (3, 10, 50)
lm_loss(logits, targets)                -> about log(50) = 3.91 at initialization
```

The targets for next-token prediction are the inputs shifted left by one: `targets = seq[:, 1:]` for `tokens = seq[:, :-1]`.

## Constraints

- Causal: the logits at position `t` depend only on tokens `0..t`.
- `lm_loss` must not overflow for logits of size `1e4` and must return a Python `float`.
- No loops except over the layers.

## What the interviewer listens for

- Why the loss at init is `log(vocab)`: a small-scale init gives near-uniform logits, and that is the first number you look at in a training log to catch bugs (a loss far above `log(vocab)` means the logits are too large at the start).
- Weight tying: it saves `vocab * d_model` parameters and couples the input and output spaces; it also means the final LayerNorm's scale directly sets the logit temperature.
- The cost per token: about `2 * n_params` FLOPs for the forward (the `6 N D` rule for training), plus the attention term `O(T^2 d)` per layer, which dominates only when `T` is large.
- Follow-up: why the softmax cross-entropy is computed from log-softmax and never from `log(softmax)`, and how the same code gives a per-token loss for perplexity reporting.
