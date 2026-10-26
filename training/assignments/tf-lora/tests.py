import torch
import torch.nn as nn

from solution import LoRALinear


def make(seed=0, in_f=32, out_f=16, r=4, alpha=8.0):
    torch.manual_seed(seed)
    base = nn.Linear(in_f, out_f)
    return LoRALinear(base, r=r, alpha=alpha), base


def test_output_equals_base_at_init():
    lora, base = make()
    x = torch.randn(5, 32)
    assert lora.A.shape == (4, 32) and lora.B.shape == (16, 4)
    assert torch.all(lora.B == 0), "B must start at zero so the update starts at zero"
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    assert lora(torch.randn(2, 3, 32)).shape == (2, 3, 16), "leading dims must pass through like nn.Linear"


def test_only_A_and_B_are_trainable():
    lora, base = make()
    names = sorted(n for n, p in lora.named_parameters() if p.requires_grad)
    assert names == ["A", "B"], names
    assert not base.weight.requires_grad and not base.bias.requires_grad
    n_train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert n_train == 4 * (32 + 16), n_train
    assert sum(p.numel() for p in lora.parameters()) == 4 * (32 + 16) + 32 * 16 + 16


def test_factored_forward_matches_the_low_rank_formula():
    lora, base = make(seed=1)
    with torch.no_grad():
        lora.B.copy_(torch.randn(16, 4))
        lora.A.copy_(torch.randn(4, 32))
    x = torch.randn(7, 32)
    expected = base(x) + (x @ (lora.B @ lora.A).t()) * (8.0 / 4)
    assert torch.allclose(lora(x), expected, atol=1e-5)


def test_merge_and_unmerge():
    lora, base = make(seed=2)
    with torch.no_grad():
        lora.B.copy_(torch.randn(16, 4) * 0.3)
    W0 = base.weight.detach().clone()
    x = torch.randn(6, 32)
    before = lora(x).detach().clone()
    lora.merge()
    assert torch.allclose(base.weight, W0 + 2.0 * (lora.B @ lora.A), atol=1e-6), "merged weight must be W0 + (alpha/r) B A"
    assert torch.allclose(lora(x), before, atol=1e-5), "merging must not change the output"
    assert torch.allclose(base(x), before, atol=1e-5), "after merging, the base layer alone gives the adapted output"
    lora.merge()
    assert torch.allclose(base.weight, W0 + 2.0 * (lora.B @ lora.A), atol=1e-6), "merge must be idempotent"
    lora.unmerge()
    assert torch.allclose(base.weight, W0, atol=1e-6)
    assert torch.allclose(lora(x), before, atol=1e-5)
    lora.unmerge()
    assert torch.allclose(base.weight, W0, atol=1e-6), "unmerge must be idempotent"


def test_gradients_flow_to_A_and_B_but_not_the_base():
    lora, base = make(seed=3)
    x, target = torch.randn(8, 32), torch.randn(8, 16)
    loss = ((lora(x) - target) ** 2).mean()
    loss.backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert lora.B.grad is not None and lora.B.grad.abs().sum() > 0, "B gets a gradient on the first step"
    assert lora.A.grad is not None and torch.all(lora.A.grad == 0), "with B = 0 the gradient to A is exactly zero at first"
    with torch.no_grad():
        lora.B -= 0.1 * lora.B.grad
    lora.zero_grad()
    ((lora(x) - target) ** 2).mean().backward()
    assert lora.A.grad.abs().sum() > 0, "once B has moved, A gets a gradient too"


def test_lora_learns_a_low_rank_update():
    torch.manual_seed(4)
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=2, alpha=4.0)
    U, V = torch.randn(16, 2), torch.randn(2, 32) / 4
    target_layer = nn.Linear(32, 16)
    with torch.no_grad():
        target_layer.weight.copy_(base.weight + U @ V)
        target_layer.bias.copy_(base.bias)
    x = torch.randn(256, 32)
    y = target_layer(x).detach()
    W0 = base.weight.detach().clone()
    opt = torch.optim.Adam([p for p in lora.parameters() if p.requires_grad], lr=2e-2)
    first = ((lora(x) - y) ** 2).mean().item()
    for _ in range(300):
        opt.zero_grad()
        loss = ((lora(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    last = ((lora(x) - y) ** 2).mean().item()
    assert last < 0.02 * first, f"loss only went from {first:.4f} to {last:.4f}"
    assert torch.equal(base.weight, W0), "the base weight must be untouched by training"
    assert torch.allclose(base.weight + 2.0 * (lora.B @ lora.A), target_layer.weight, atol=0.15), "the learned update should approximate U @ V"
