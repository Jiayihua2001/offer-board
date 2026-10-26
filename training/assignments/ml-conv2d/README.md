# 2-D convolution from scratch

A convolution layer is asked for because it checks three things: that you can write down the output-size formula, that you know PyTorch's `conv2d` is actually cross-correlation (no kernel flip), and that you can turn a six-deep nested loop into a handful of array operations. The usual trajectory in the room is "naive loops, now make it fast", so build it so that only the loop over output positions remains, or go straight to im2col.

Implement:

- `out_size(n, k, stride, padding) -> int`: the output length of one spatial axis, `(n + 2 * padding - k) // stride + 1`.
- `conv2d(x, w, b=None, stride=1, padding=0) -> out`: `x` has shape `(N, C, H, W)`, `w` has shape `(F, C, kH, kW)`, `b` is `None` or shape `(F,)`. Zero-pad `x` by `padding` on each spatial side, then `out[n, f, i, j] = b[f] + sum over c, u, v of w[f, c, u, v] * x_pad[n, c, i * stride + u, j * stride + v]`. Output shape `(N, F, H_out, W_out)`.

## Example

```
x = np.array([[1, 2], [3, 4]], float).reshape(1, 1, 2, 2)
w = np.ones((1, 1, 2, 2))
conv2d(x, w)                     -> [[[[10.]]]]

impulse = np.zeros((1, 1, 3, 3)); impulse[0, 0, 1, 1] = 1
k = np.arange(9.).reshape(1, 1, 3, 3)
conv2d(impulse, k, padding=1)    -> k flipped in both axes (that is cross-correlation)

out_size(32, 3, stride=2, padding=1) -> 16
```

## Constraints

- No Python loops over `N`, `C` or `F`. Loops over the `H_out x W_out` output positions are acceptable; im2col with a single matrix product is better.
- Must match a brute-force reference exactly (`atol=1e-10`) for several `(stride, padding)` combinations, including non-square kernels.
- A larger input (`N=16, C=8, F=16, 32x32`) is in the tests; the fully naive version is too slow for it.

## What the interviewer listens for

- The output-size formula and where the floor comes from (the last window that still fits), plus how "same" padding is `(k - 1) / 2` for odd `k` and stride 1.
- Cross-correlation versus convolution: the learned kernel absorbs the flip, so frameworks skip it; the impulse-response test makes the convention visible.
- The im2col trade-off: it turns the op into one GEMM (fast, uses BLAS) at the cost of materializing every patch, `C * kH * kW` times the input memory.
- Follow-up: the cost in multiply-adds (`N * F * C * kH * kW * H_out * W_out`), how a depthwise-separable convolution cuts it, and what the backward pass looks like (the gradient with respect to the input is a transposed convolution).
