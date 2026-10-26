"""A tiny GPT in PyTorch on a synthetic sorting task. The data is provided; write TinyGPT and train()."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIGITS = 10          # symbols 0..9
SEQ = 6                # digits per problem
SEP = N_DIGITS         # separator token between the problem and the answer
VOCAB = N_DIGITS + 1
MAX_LEN = 2 * SEQ + 1  # digits, SEP, sorted digits


def make_batch(batch_size: int, gen: torch.Generator):
    """Sorting problems. x: (B, 2*SEQ) = digits, SEP, sorted[:-1]; y: (B, 2*SEQ) = next tokens, -100 where ignored."""
    digits = torch.randint(0, N_DIGITS, (batch_size, SEQ), generator=gen)
    seq = torch.cat([digits, torch.full((batch_size, 1), SEP), digits.sort(dim=1).values], dim=1)
    x, y = seq[:, :-1].clone(), seq[:, 1:].clone()
    y[:, :SEQ] = -100
    return x, y


class TinyGPT(nn.Module):
    """Token + positional embeddings, n_layers pre-norm causal blocks, final LayerNorm, tied output head."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) long, T <= max_len. Returns logits (B, T, vocab_size)."""
        raise NotImplementedError


def train(model: nn.Module, data_fn, steps: int, lr: float) -> list[float]:
    """`steps` optimizer steps on batches (x, y) = data_fn(); cross-entropy with ignore_index=-100. Returns the losses."""
    raise NotImplementedError
