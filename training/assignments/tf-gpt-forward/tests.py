import numpy as np

from solution import gpt_forward, init_params, lm_loss


def ref_ln(x, g, b, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * g + b


def ref_forward(tokens, params):
    """Independent reference: one sequence at a time, explicit head loop."""
    H = params["n_heads"]
    out = []
    for seq in tokens:
        T = len(seq)
        x = params["wte"][seq] + params["wpe"][:T]
        d = x.shape[-1]
        dh = d // H
        mask = np.tril(np.ones((T, T), dtype=bool))
        for p in params["blocks"]:
            a = ref_ln(x, p["ln1_g"], p["ln1_b"])
            q, k, v = a @ p["Wq"], a @ p["Wk"], a @ p["Wv"]
            heads = []
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                s = np.where(mask, q[:, sl] @ k[:, sl].T / np.sqrt(dh), -np.inf)
                w = np.exp(s - s.max(-1, keepdims=True))
                w /= w.sum(-1, keepdims=True)
                heads.append(w @ v[:, sl])
            x = x + np.concatenate(heads, -1) @ p["Wo"]
            z = ref_ln(x, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"]
            z = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))
            x = x + z @ p["W2"] + p["b2"]
        x = ref_ln(x, params["lnf_g"], params["lnf_b"])
        out.append(x @ params["wte"].T)
    return np.stack(out)


def small_model(seed=0, scale=0.3):
    rng = np.random.default_rng(seed)
    params = init_params(vocab=20, d_model=16, n_heads=4, n_layers=2, T=12, rng=rng, scale=scale)
    tokens = rng.integers(0, 20, size=(3, 9))
    return params, tokens, rng


def test_logits_shape():
    params, tokens, _ = small_model()
    logits = gpt_forward(tokens, params)
    assert logits.shape == (3, 9, 20), logits.shape
    assert np.all(np.isfinite(logits))
    assert gpt_forward(tokens[:1, :4], params).shape == (1, 4, 20)


def test_matches_reference():
    params, tokens, _ = small_model(seed=1)
    assert np.allclose(gpt_forward(tokens, params), ref_forward(tokens, params), atol=1e-8)


def test_causality():
    params, tokens, rng = small_model(seed=2)
    tokens2 = tokens.copy()
    tokens2[:, 6:] = (tokens[:, 6:] + 1) % 20
    a, b = gpt_forward(tokens, params), gpt_forward(tokens2, params)
    assert np.allclose(a[:, :6], b[:, :6], atol=1e-10), "logits before position 6 changed when later tokens changed"
    assert not np.allclose(a[:, 6:], b[:, 6:])


def test_positions_matter_and_batch_items_are_independent():
    params, tokens, _ = small_model(seed=3)
    same = np.full((1, 6), 7)
    logits = gpt_forward(same, params)
    assert not np.allclose(logits[0, 0], logits[0, 5]), "the same token at two positions must get different logits"
    full = gpt_forward(tokens, params)
    for i in range(tokens.shape[0]):
        assert np.allclose(gpt_forward(tokens[i:i + 1], params), full[i:i + 1], atol=1e-12)


def test_loss_is_about_log_vocab_at_init():
    rng = np.random.default_rng(4)
    params = init_params(vocab=50, d_model=32, n_heads=4, n_layers=2, T=16, rng=rng)  # default scale 0.02
    seq = rng.integers(0, 50, size=(4, 11))
    loss = lm_loss(gpt_forward(seq[:, :-1], params), seq[:, 1:])
    assert isinstance(loss, float)
    assert abs(loss - np.log(50)) < 0.05, loss


def test_lm_loss_values_and_stability():
    logits = np.array([[[2.0, 1.0, 0.1], [0.0, 0.0, 0.0]]])
    targets = np.array([[0, 2]])
    p0 = np.exp(2.0) / (np.exp(2.0) + np.exp(1.0) + np.exp(0.1))
    expected = -(np.log(p0) + np.log(1 / 3)) / 2
    assert np.isclose(lm_loss(logits, targets), expected, atol=1e-10)
    big = np.array([[[1e4, 0.0, -1e4]]])
    assert np.isfinite(lm_loss(big, np.array([[0]]))) and np.isclose(lm_loss(big, np.array([[0]])), 0.0, atol=1e-6)
    assert np.isclose(lm_loss(big, np.array([[1]])), 1e4, rtol=1e-6)
    # a perfect prediction at every position gives (almost) zero loss
    rng = np.random.default_rng(5)
    targets = rng.integers(0, 6, size=(2, 5))
    sharp = np.full((2, 5, 6), -50.0)
    np.put_along_axis(sharp, targets[..., None], 50.0, axis=-1)
    assert lm_loss(sharp, targets) < 1e-6
