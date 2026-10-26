"""A PyTorch training loop with validation, early stopping and best-weight restore. CPU only."""
import math

import torch
import torch.nn as nn


def make_moons(n: int, noise: float = 0.15, seed: int = 0):
    """Two interleaving half-circles. Provided. Returns X (n, 2) float32 and y (n,) int64."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * math.pi
    half = torch.arange(n) % 2
    x = torch.where(half == 0, torch.cos(t), 1.0 - torch.cos(t))
    y = torch.where(half == 0, torch.sin(t), 0.5 - torch.sin(t))
    X = torch.stack([x, y], dim=1) + noise * torch.randn(n, 2, generator=g)
    return X.float(), half.long()


class MLP(nn.Module):
    """d_in -> hidden -> hidden -> d_out with ReLU and dropout after each hidden layer."""

    def __init__(self, d_in: int = 2, hidden: int = 32, d_out: int = 2, dropout: float = 0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def train_epoch(model: nn.Module, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer,
                loss_fn, batch_size: int = 32) -> float:
    """One pass over (X, y) in shuffled mini-batches, training mode. Returns the mean loss per sample."""
    raise NotImplementedError


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn):
    """(loss, accuracy) over the whole set, in eval mode with gradients disabled. Both Python floats."""
    raise NotImplementedError


def fit(model: nn.Module, X_train, y_train, X_val, y_val, epochs: int = 100, lr: float = 1e-2,
        patience: int = 5, batch_size: int = 32) -> dict:
    """Adam + cross-entropy with early stopping on validation loss; restores the best weights.

    Returns {"train_loss": [...], "val_loss": [...], "val_acc": [...], "best_epoch": int, "stopped_epoch": int}.
    """
    raise NotImplementedError
