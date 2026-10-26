import numpy as np


def out_size(n: int, k: int, stride: int = 1, padding: int = 0) -> int:
    return (n + 2 * padding - k) // stride + 1


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None, stride: int = 1, padding: int = 0) -> np.ndarray:
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    ho, wo = out_size(h, kh, stride, padding), out_size(wd, kw, stride, padding)
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    # im2col: cols[n, i, j, c, u, v] = xp[n, c, i*s + u, j*s + v], gathered with fancy indexing.
    ii = (stride * np.arange(ho))[:, None] + np.arange(kh)[None, :]      # (ho, kh)
    jj = (stride * np.arange(wo))[:, None] + np.arange(kw)[None, :]      # (wo, kw)
    cols = xp[:, :, ii[:, None, :, None], jj[None, :, None, :]]           # (n, c, ho, wo, kh, kw)
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * kh * kw)

    out = cols @ w.reshape(f, -1).T                                       # one GEMM: (n*ho*wo, f)
    out = out.reshape(n, ho, wo, f).transpose(0, 3, 1, 2)
    if b is not None:
        out = out + b[None, :, None, None]
    return out
