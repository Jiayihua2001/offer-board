# LayerNorm, GELU and the feed-forward block

The other half of a transformer block. Attention moves information between positions; the position-wise feed-forward network transforms it, and LayerNorm keeps every token's activations on a fixed scale so dozens of residual blocks can be stacked. These three are short, and they are the parts of GPT-2 that people mis-remember on a whiteboard (biased or unbiased variance, where `eps` goes, which GELU).

- `layer_norm(x, g, b, eps=1e-5)`: `x` is `(..., d)`; normalize each vector over the last axis to mean 0 and variance 1 (population variance, `eps` inside the square root), then scale by `g` and shift by `b`, both `(d,)`.
- `gelu(x)`: the tanh approximation used by GPT-2, `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, elementwise.
- `ffn(x, W1, b1, W2, b2)`: `gelu(x @ W1 + b1) @ W2 + b2` with `W1` of shape `(d, d_ff)`, `W2` of shape `(d_ff, d)`. Applied independently to every position.

## Example

```
layer_norm([[1, 2, 3, 4]], g=1, b=0, eps=0) -> [[-1.3416, -0.4472, 0.4472, 1.3416]]
gelu([-2, -1, 0, 1, 2])                     -> [-0.0454, -0.1588, 0, 0.8412, 1.9546]
ffn(x (2, 3, 8), W1 (8, 32), b1 (32,), W2 (32, 8), b2 (8,)).shape -> (2, 3, 8)
```

## Constraints

- Everything is vectorized over the leading dims; no loops over tokens.
- LayerNorm of a constant vector must be finite (that is what `eps` is for) and equal to `b`.
- The tests check `gelu` against the tanh form to `1e-6`.

## What the interviewer listens for

- Why normalize over the feature axis and not the batch: LayerNorm has no batch statistics, so it behaves the same at train and test time and for a sequence of length 1 during decoding, which BatchNorm cannot do.
- The cost is negligible (`O(d)` per token) next to the FFN, which is about two thirds of the parameters of a GPT block (`8 d^2` of `12 d^2`) and the same share of the FLOPs.
- The failure mode: `eps` outside the square root, or the sample (`n-1`) variance, which both pass a casual look but do not match a trained checkpoint.
- Follow-up: RMSNorm drops the mean subtraction and the bias (used in Llama), and SwiGLU replaces GELU with a gated hidden layer with `2/3` the width to keep the parameter count.
