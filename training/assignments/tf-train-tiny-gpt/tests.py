import torch

from solution import MAX_LEN, N_DIGITS, SEP, SEQ, VOCAB, TinyGPT, make_batch, train

CFG = dict(vocab_size=VOCAB, d_model=64, n_heads=4, n_layers=2, max_len=MAX_LEN)
STEPS, LR = 600, 1e-3
_cache = {}


def fresh_model(seed=0):
    torch.manual_seed(seed)
    return TinyGPT(**CFG)


def trained_model():
    """Train once; the later tests share the result."""
    if "model" not in _cache:
        torch.manual_seed(0)
        model = TinyGPT(**CFG)
        gen = torch.Generator().manual_seed(1)
        losses = train(model, lambda: make_batch(64, gen), STEPS, LR)
        _cache["model"], _cache["losses"] = model, [float(l) for l in losses]
    return _cache["model"], _cache["losses"]


@torch.no_grad()
def solve(model, digits):
    """Greedy autoregressive decoding of the answer, given the digits and the separator."""
    model.eval()
    seq = torch.cat([digits, torch.full((digits.shape[0], 1), SEP)], dim=1)
    for _ in range(SEQ):
        nxt = model(seq)[:, -1].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    return seq[:, SEQ + 1:]


def test_make_batch_layout():
    x, y = make_batch(8, torch.Generator().manual_seed(0))
    assert x.shape == (8, 2 * SEQ) and y.shape == (8, 2 * SEQ)
    assert torch.all(x[:, SEQ] == SEP)
    assert torch.all(y[:, :SEQ] == -100) and torch.all(y[:, SEQ:] >= 0)
    assert torch.equal(y[:, SEQ:], x[:, :SEQ].sort(dim=1).values)


def test_forward_shape_and_parameter_count():
    model = fresh_model()
    x, _ = make_batch(4, torch.Generator().manual_seed(0))
    logits = model(x)
    assert logits.shape == (4, 2 * SEQ, VOCAB), logits.shape
    assert torch.isfinite(logits).all()
    assert model(x[:, :5]).shape == (4, 5, VOCAB), "any T <= max_len must work"
    n = sum(p.numel() for p in model.parameters())
    assert 50_000 <= n <= 200_000, f"{n} parameters; expected about 100k for d_model=64, 2 layers"


def test_causal_and_batch_independent():
    model = fresh_model(1).eval()
    x, _ = make_batch(3, torch.Generator().manual_seed(2))
    x2 = x.clone()
    x2[:, 8:] = (x2[:, 8:] + 1) % N_DIGITS
    with torch.no_grad():
        a, b = model(x), model(x2)
        assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5), "logits before position 8 changed when later tokens changed"
        assert not torch.allclose(a[:, 8:], b[:, 8:])
        assert torch.allclose(model(x[:1]), a[:1], atol=1e-5), "batch items must not interact"


def test_training_reduces_the_loss():
    _, losses = trained_model()
    assert len(losses) == STEPS
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    assert last < 0.25 * first, f"loss went from {first:.3f} to {last:.3f}"
    assert last < 0.3, f"final loss {last:.3f}"


def test_exact_match_on_held_out_problems():
    model, _ = trained_model()
    digits = torch.randint(0, N_DIGITS, (300, SEQ), generator=torch.Generator().manual_seed(12345))
    pred = solve(model, digits)
    acc = (pred == digits.sort(dim=1).values).all(dim=1).float().mean().item()
    assert acc >= 0.9, f"exact match {acc:.3f} after {STEPS} steps; the reference reaches about 0.99"
    example = solve(model, torch.tensor([[3, 1, 4, 1, 5, 9]]))
    assert example.tolist() == [[1, 1, 3, 4, 5, 9]], example.tolist()
