"""LoRA wrapper around a frozen nn.Linear."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = base(x) + (x @ A.T @ B.T) * (alpha / r), with base frozen, A (r, in) random, B (out, r) zero."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def merge(self) -> None:
        """Fold scaling * B @ A into base.weight so forward uses the base layer alone. Idempotent."""
        raise NotImplementedError

    def unmerge(self) -> None:
        """Undo merge() exactly. Idempotent."""
        raise NotImplementedError
