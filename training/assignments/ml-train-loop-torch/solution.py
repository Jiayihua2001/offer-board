import math

import torch
import torch.nn as nn


def make_moons(n: int, noise: float = 0.15, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * math.pi
    half = torch.arange(n) % 2
    x = torch.where(half == 0, torch.cos(t), 1.0 - torch.cos(t))
    y = torch.where(half == 0, torch.sin(t), 0.5 - torch.sin(t))
    X = torch.stack([x, y], dim=1) + noise * torch.randn(n, 2, generator=g)
    return X.float(), half.long()


class MLP(nn.Module):
    def __init__(self, d_in: int = 2, hidden: int = 32, d_out: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer,
                loss_fn, batch_size: int = 32) -> float:
    model.train()
    perm = torch.randperm(X.shape[0])
    total = 0.0
    for start in range(0, X.shape[0], batch_size):
        idx = perm[start:start + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)              # loss is a batch mean: weight by batch size
    return total / X.shape[0]


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = loss_fn(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def fit(model: nn.Module, X_train, y_train, X_val, y_val, epochs: int = 100, lr: float = 1e-2,
        patience: int = 5, batch_size: int = 32) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "best_epoch": 0, "stopped_epoch": 0}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, X_train, y_train, optimizer, loss_fn, batch_size)
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["stopped_epoch"] = epoch
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            history["best_epoch"] = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return history
