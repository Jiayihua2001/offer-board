import numpy as np

from solution import forward_with_cache, generate, gpt_forward, init_params, sample_next

LOGITS = np.log([0.5, 0.3, 0.15, 0.05])


def freq(draws, V=4):
    return np.bincount(np.asarray(draws), minlength=V) / len(draws)


def model(seed=0):
    rng = np.random.default_rng(seed)
    return init_params(vocab=16, d_model=16, n_heads=2, n_layers=2, T=32, rng=rng, scale=0.1)


def test_temperature_zero_is_argmax():
    rng = np.random.default_rng(0)
    logits = np.array([0.1, 2.0, -1.0, 1.9])
    draws = [sample_next(logits, rng, temperature=0.0) for _ in range(20)]
    assert draws == [1] * 20
    assert isinstance(draws[0], int)
    assert sample_next(logits, rng, temperature=0.0, top_k=2, top_p=0.5) == 1
    assert sample_next(LOGITS, None, temperature=0.0) == 0


def test_plain_sampling_follows_the_softmax():
    rng = np.random.default_rng(1)
    draws = [sample_next(LOGITS, rng) for _ in range(4000)]
    assert np.allclose(freq(draws), [0.5, 0.3, 0.15, 0.05], atol=0.04), freq(draws)
    # logits are shift-invariant
    draws2 = [sample_next(LOGITS + 100.0, rng) for _ in range(4000)]
    assert np.allclose(freq(draws2), [0.5, 0.3, 0.15, 0.05], atol=0.04)


def test_temperature_sharpens_and_flattens():
    rng = np.random.default_rng(2)
    cold = [sample_next(LOGITS, rng, temperature=0.1) for _ in range(500)]
    assert freq(cold)[0] > 0.98, freq(cold)
    hot = [sample_next(LOGITS, rng, temperature=100.0) for _ in range(4000)]
    assert np.all(np.abs(freq(hot) - 0.25) < 0.05), freq(hot)


def test_top_k_keeps_only_the_k_largest():
    rng = np.random.default_rng(3)
    draws = [sample_next(LOGITS, rng, top_k=2) for _ in range(3000)]
    assert set(draws) == {0, 1}, set(draws)
    assert np.allclose(freq(draws)[:2], [0.625, 0.375], atol=0.04), freq(draws)
    assert all(sample_next(LOGITS, rng, top_k=1) == 0 for _ in range(20))
    draws = [sample_next(LOGITS, rng, top_k=10) for _ in range(3000)]  # larger than the vocabulary: no effect
    assert set(draws) == {0, 1, 2, 3}
    shuffled = LOGITS[[2, 0, 3, 1]]  # the k largest, wherever they sit
    assert set(sample_next(shuffled, rng, top_k=2) for _ in range(500)) == {1, 3}


def test_top_p_keeps_the_smallest_nucleus():
    rng = np.random.default_rng(4)
    for p, support in [(0.45, {0}), (0.75, {0, 1}), (0.9, {0, 1, 2}), (0.99, {0, 1, 2, 3})]:
        draws = [sample_next(LOGITS, rng, top_p=p) for _ in range(2000)]
        assert set(draws) == support, (p, set(draws))
    draws = [sample_next(LOGITS, rng, top_p=0.75) for _ in range(3000)]
    assert np.allclose(freq(draws)[:2], [0.625, 0.375], atol=0.04), freq(draws)
    shuffled = np.log([0.05, 0.5, 0.15, 0.3])
    assert set(sample_next(shuffled, rng, top_p=0.75) for _ in range(500)) == {1, 3}
    # temperature is applied before the nucleus is chosen: a cold distribution has a smaller nucleus
    assert set(sample_next(LOGITS, rng, temperature=0.3, top_p=0.9) for _ in range(500)) == {0, 1}


def test_prefill_matches_the_full_forward():
    params = model()
    prompt = np.array([3, 1, 4, 1, 5])
    logits, cache = forward_with_cache(prompt, params)
    assert logits.shape == (5, 16)
    assert np.allclose(logits, gpt_forward(prompt[None], params)[0], atol=1e-8)


def test_incremental_decoding_matches_the_full_forward():
    params = model(seed=1)
    seq = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    full = gpt_forward(seq[None], params)[0]
    cache, rows = None, []
    for t in seq:  # one token at a time
        logits, cache = forward_with_cache(np.array([t]), params, cache)
        assert logits.shape == (1, 16)
        rows.append(logits[0])
    assert np.allclose(np.stack(rows), full, atol=1e-8), "token-by-token logits differ from the full forward"
    cache, rows = None, []
    for a, b in [(0, 3), (3, 4), (4, 6), (6, 10)]:  # chunked prefill
        logits, cache = forward_with_cache(seq[a:b], params, cache)
        rows.append(logits)
    assert np.allclose(np.concatenate(rows), full, atol=1e-8), "chunked prefill differs from the full forward"


def test_greedy_generation_with_and_without_cache():
    params = model(seed=2)
    prompt = [3, 1, 4, 1, 5]
    a = generate(prompt, params, 8, use_cache=False)
    b = generate(prompt, params, 8, use_cache=True)
    assert a == b, (a, b)
    assert len(a) == 13 and a[:5] == prompt and all(isinstance(t, int) and 0 <= t < 16 for t in a)
    for i in range(5, 13):  # greedy means each token is the argmax of the full-forward logits at that step
        assert a[i] == int(np.argmax(gpt_forward(np.array([a[:i]]), params)[0, -1]))


def test_sampled_generation_matches_with_the_same_rng():
    params = model(seed=3)
    prompt = [7, 7, 2]
    a = generate(prompt, params, 10, rng=np.random.default_rng(5), use_cache=False, temperature=1.0, top_k=5)
    b = generate(prompt, params, 10, rng=np.random.default_rng(5), use_cache=True, temperature=1.0, top_k=5)
    assert a == b and len(a) == 13, (a, b)
    c = generate(prompt, params, 10, rng=np.random.default_rng(6), use_cache=True, temperature=1.0, top_k=5)
    assert c != a, "a different rng seed should give a different sample"
    d = generate(prompt, params, 10, rng=np.random.default_rng(7), use_cache=True, temperature=1.0, top_p=0.5)
    assert len(d) == 13 and d[:3] == prompt
