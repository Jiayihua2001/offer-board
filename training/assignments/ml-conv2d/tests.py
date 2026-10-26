import numpy as np

from solution import conv2d, out_size


def conv2d_reference(x, w, b=None, stride=1, padding=0):
    """Deliberately naive: loops over everything."""
    n_, c_, h_, w_ = x.shape
    f_, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    ho = (h_ + 2 * padding - kh) // stride + 1
    wo = (w_ + 2 * padding - kw) // stride + 1
    out = np.zeros((n_, f_, ho, wo))
    for n in range(n_):
        for f in range(f_):
            for i in range(ho):
                for j in range(wo):
                    s = 0.0 if b is None else b[f]
                    for c in range(c_):
                        for u in range(kh):
                            for v in range(kw):
                                s += w[f, c, u, v] * xp[n, c, i * stride + u, j * stride + v]
                    out[n, f, i, j] = s
    return out


def test_out_size_formula():
    assert out_size(5, 3) == 3
    assert out_size(5, 3, stride=1, padding=1) == 5
    assert out_size(32, 3, stride=2, padding=1) == 16
    assert out_size(7, 2, stride=2, padding=0) == 3
    assert out_size(28, 5, stride=1, padding=2) == 28


def test_box_filter_and_output_shape():
    x = np.array([[1.0, 2.0], [3.0, 4.0]]).reshape(1, 1, 2, 2)
    w = np.ones((1, 1, 2, 2))
    out = conv2d(x, w)
    assert out.shape == (1, 1, 1, 1)
    assert np.isclose(out[0, 0, 0, 0], 10.0)
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    out = conv2d(x, w)
    assert out.shape == (1, 1, 3, 3)
    assert np.allclose(out[0, 0], [[10, 14, 18], [26, 30, 34], [42, 46, 50]])


def test_impulse_response_is_the_flipped_kernel():
    impulse = np.zeros((1, 1, 3, 3))
    impulse[0, 0, 1, 1] = 1.0
    k = np.arange(9.0).reshape(1, 1, 3, 3)
    out = conv2d(impulse, k, padding=1)
    assert out.shape == (1, 1, 3, 3)
    assert np.allclose(out[0, 0], k[0, 0, ::-1, ::-1]), "cross-correlation of an impulse returns the kernel flipped"


def test_matches_reference_no_padding_stride_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 6, 5))
    w = rng.normal(size=(4, 3, 3, 2))          # non-square kernel
    b = rng.normal(size=4)
    out = conv2d(x, w, b)
    ref = conv2d_reference(x, w, b)
    assert out.shape == ref.shape == (2, 4, 4, 4)
    assert np.allclose(out, ref, atol=1e-10)


def test_matches_reference_with_stride_and_padding():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 2, 7, 8))
    w = rng.normal(size=(3, 2, 3, 3))
    for stride, padding in [(2, 1), (3, 0), (2, 2), (1, 2)]:
        out = conv2d(x, w, None, stride=stride, padding=padding)
        ref = conv2d_reference(x, w, None, stride=stride, padding=padding)
        assert out.shape == ref.shape, f"stride {stride}, padding {padding}: {out.shape} vs {ref.shape}"
        assert np.allclose(out, ref, atol=1e-10), f"mismatch at stride {stride}, padding {padding}"


def test_bias_is_added_per_filter():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 2, 4, 4))
    w = rng.normal(size=(3, 2, 2, 2))
    b = np.array([10.0, -20.0, 30.0])
    diff = conv2d(x, w, b) - conv2d(x, w)
    assert np.allclose(diff[0, 0], 10.0) and np.allclose(diff[0, 1], -20.0) and np.allclose(diff[0, 2], 30.0)


def test_identity_kernel_with_same_padding():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 1, 5, 5))
    w = np.zeros((1, 1, 3, 3))
    w[0, 0, 1, 1] = 1.0
    assert np.allclose(conv2d(x, w, padding=1), x)


def test_larger_input_is_fast_enough():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(16, 8, 32, 32))
    w = rng.normal(size=(16, 8, 3, 3))
    b = rng.normal(size=16)
    out = conv2d(x, w, b, stride=1, padding=1)
    assert out.shape == (16, 16, 32, 32)
    # spot-check a few output positions against the naive formula
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    for n, f, i, j in [(0, 0, 0, 0), (5, 7, 13, 2), (15, 15, 31, 31), (3, 9, 16, 30)]:
        ref = b[f] + np.sum(w[f] * xp[n, :, i:i + 3, j:j + 3])
        assert np.isclose(out[n, f, i, j], ref, atol=1e-9)
