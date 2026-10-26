import numpy as np

from solution import embed, embed_with_positions, sinusoidal_pe


def ref_pe(T, d):
    pe = np.zeros((T, d))
    for pos in range(T):
        for j in range(d):
            angle = pos / (10000 ** (2 * (j // 2) / d))
            pe[pos, j] = np.sin(angle) if j % 2 == 0 else np.cos(angle)
    return pe


def test_embed_is_a_lookup():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(10, 4))
    tokens = np.array([[3, 1, 3], [0, 9, 2]])
    x = embed(tokens, E)
    assert x.shape == (2, 3, 4)
    assert np.allclose(x[0, 0], E[3]) and np.allclose(x[0, 2], E[3]) and np.allclose(x[1, 1], E[9])
    assert embed(np.array([7, 7]), E).shape == (2, 4)
    assert np.allclose(embed(np.array([7, 7]), E)[0], E[7])


def test_pe_reference_values():
    pe = sinusoidal_pe(6, 8)
    assert pe.shape == (6, 8)
    assert np.allclose(pe[0], [0, 1, 0, 1, 0, 1, 0, 1])
    assert np.isclose(pe[1, 0], np.sin(1.0)) and np.isclose(pe[1, 1], np.cos(1.0))
    assert np.isclose(pe[3, 2], np.sin(3 / 10000 ** (2 / 8)))
    assert np.isclose(pe[5, 7], np.cos(5 / 10000 ** (6 / 8)))
    assert np.allclose(pe, ref_pe(6, 8), atol=1e-12)
    assert np.allclose(sinusoidal_pe(50, 16), ref_pe(50, 16), atol=1e-12)


def test_pe_is_bounded_and_rows_are_distinct():
    pe = sinusoidal_pe(100, 32)
    assert np.all(np.abs(pe) <= 1 + 1e-12)
    dist = np.linalg.norm(pe[:, None, :] - pe[None, :, :], axis=-1)
    dist[np.arange(100), np.arange(100)] = np.inf
    assert dist.min() > 1e-3, "two positions got (nearly) the same encoding"


def test_shift_is_a_fixed_linear_map_per_frequency():
    T, d, k, i = 40, 16, 7, 2  # frequency pair i lives in columns 2i and 2i+1
    pe = sinusoidal_pe(T, d)
    w = 1 / 10000 ** (2 * i / d)
    M = np.array([[np.cos(k * w), np.sin(k * w)], [-np.sin(k * w), np.cos(k * w)]])
    before = pe[: T - k, 2 * i: 2 * i + 2]
    after = pe[k:, 2 * i: 2 * i + 2]
    assert np.allclose(after, before @ M.T, atol=1e-10)
    # the same matrix does not describe a different shift
    assert not np.allclose(pe[k + 1:, 2 * i: 2 * i + 2], pe[: T - k - 1, 2 * i: 2 * i + 2] @ M.T)


def test_wavelengths_grow_geometrically():
    pe = sinusoidal_pe(8, 4)
    assert np.isclose(pe[1, 0], np.sin(1.0))  # first pair: 1 radian per position
    assert np.isclose(pe[1, 2], np.sin(0.01))  # last pair of d=4: 10000^(-2/4) = 1/100 radian per position
    assert np.isclose(pe[2, 1], np.cos(2.0))


def test_embed_with_positions():
    rng = np.random.default_rng(1)
    E = rng.normal(size=(12, 8))
    P = sinusoidal_pe(20, 8)
    tokens = np.array([[4, 4, 4], [1, 2, 3]])
    x = embed_with_positions(tokens, E, P)
    assert x.shape == (2, 3, 8)
    assert np.allclose(x[0, 1], E[4] + P[1])
    assert np.allclose(x[1, 2], E[3] + P[2])
    assert not np.allclose(x[0, 0], x[0, 2]), "the same token at two positions must differ"
    P2 = rng.normal(size=(20, 8))  # a learned table works the same way
    assert np.allclose(embed_with_positions(tokens, E, P2), E[tokens] + P2[:3])
