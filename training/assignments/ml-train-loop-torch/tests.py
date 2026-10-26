import math

import torch
import torch.nn as nn

from solution import MLP, evaluate, fit, make_moons, train_epoch

torch.set_num_threads(1)


class Probe(nn.Module):
    """Records the mode, the grad state and the batch size of every forward call."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(2, 2))
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), x.shape[0]))
        return x @ self.w


def data():
    X_tr, y_tr = make_moons(600, noise=0.15, seed=0)
    X_va, y_va = make_moons(200, noise=0.15, seed=1)
    return X_tr, y_tr, X_va, y_va


def test_mlp_shapes_dropout_and_modes():
    torch.manual_seed(0)
    model = MLP(d_in=2, hidden=16, d_out=3, dropout=0.5)
    x = torch.randn(7, 2)
    assert model(x).shape == (7, 3)
    assert any(isinstance(m, nn.Dropout) for m in model.modules()), "use nn.Dropout"
    assert sum(p.numel() for p in model.parameters()) == 2 * 16 + 16 + 16 * 16 + 16 + 16 * 3 + 3
    model.train()
    assert not torch.allclose(model(x), model(x)), "dropout should make training-mode outputs stochastic"
    model.eval()
    assert torch.allclose(model(x), model(x)), "eval-mode outputs must be deterministic"


def test_train_epoch_visits_every_sample_once_in_training_mode():
    torch.manual_seed(0)
    probe = Probe()
    X, y = make_moons(100, seed=2)
    frozen = torch.optim.SGD(probe.parameters(), lr=0.0)
    loss = train_epoch(probe, X, y, frozen, nn.CrossEntropyLoss(), batch_size=32)
    assert isinstance(loss, float)
    assert math.isclose(loss, math.log(2.0), rel_tol=1e-5), "with zero weights every batch costs exactly log 2"
    sizes = [c[2] for c in probe.calls]
    assert sum(sizes) == 100, f"batches covered {sum(sizes)} samples, expected 100"
    assert max(sizes) <= 32 and len(sizes) == 4
    assert all(c[0] for c in probe.calls), "train_epoch must run the model in training mode"
    assert all(c[1] for c in probe.calls), "gradients must be enabled during training"
    opt = torch.optim.SGD(probe.parameters(), lr=0.1)
    loss2 = train_epoch(probe, X, y, opt, nn.CrossEntropyLoss(), batch_size=32)
    assert loss2 < math.log(2.0), "later batches should already benefit from the earlier steps"
    assert not torch.allclose(probe.w, torch.zeros(2, 2)), "parameters should change after an epoch"


def test_evaluate_uses_eval_mode_and_no_grad():
    probe = Probe()
    X, y = make_moons(50, seed=3)
    loss, acc = evaluate(probe, X, y, nn.CrossEntropyLoss())
    assert isinstance(loss, float) and isinstance(acc, float)
    assert math.isclose(loss, math.log(2.0), rel_tol=1e-5)
    assert math.isclose(acc, (y == 0).float().mean().item()), "zero logits predict class 0 everywhere"
    assert len(probe.calls) >= 1
    assert all(not c[0] for c in probe.calls), "evaluate must call model.eval()"
    assert all(not c[1] for c in probe.calls), "evaluate must run under torch.no_grad()"


def test_reaches_95_percent_validation_accuracy():
    torch.manual_seed(0)
    X_tr, y_tr, X_va, y_va = data()
    model = MLP()
    h = fit(model, X_tr, y_tr, X_va, y_va, epochs=60, lr=1e-2, patience=10)
    assert max(h["val_acc"]) >= 0.95, f"best validation accuracy {max(h['val_acc']):.3f}"
    _, acc = evaluate(model, X_va, y_va, nn.CrossEntropyLoss())
    assert acc >= 0.95, f"restored model has accuracy {acc:.3f}"
    assert h["train_loss"][-1] < h["train_loss"][0]


def test_early_stopping_triggers_on_a_flat_run():
    torch.manual_seed(1)
    X_tr, y_tr, X_va, y_va = data()
    h = fit(MLP(), X_tr, y_tr, X_va, y_va, epochs=50, lr=0.0, patience=4)
    assert len(h["val_loss"]) == 5, f"lr = 0 cannot improve, so it should stop after patience + 1 = 5 epochs, ran {len(h['val_loss'])}"
    assert h["best_epoch"] == 0 and h["stopped_epoch"] == 4
    assert len(h["train_loss"]) == len(h["val_acc"]) == 5


def test_best_weights_are_restored_after_early_stopping():
    torch.manual_seed(2)
    X_tr, y_tr, X_va, y_va = data()
    model = MLP()
    h = fit(model, X_tr, y_tr, X_va, y_va, epochs=300, lr=1e-2, patience=3)
    n = len(h["val_loss"])
    assert n < 300, "the run should stop early once the validation loss plateaus"
    assert h["stopped_epoch"] == n - 1
    assert h["best_epoch"] == min(range(n), key=lambda i: h["val_loss"][i])
    assert h["best_epoch"] < h["stopped_epoch"], "by construction the last epochs were not improvements"
    loss, _ = evaluate(model, X_va, y_va, nn.CrossEntropyLoss())
    assert math.isclose(loss, h["val_loss"][h["best_epoch"]], rel_tol=1e-5), \
        f"restored loss {loss:.5f} != best recorded {h['val_loss'][h['best_epoch']]:.5f}"
