import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_DIGITS = 10
SEQ = 6
SEP = N_DIGITS
VOCAB = N_DIGITS + 1
MAX_LEN = 2 * SEQ + 1


def make_batch(batch_size: int, gen: torch.Generator):
    digits = torch.randint(0, N_DIGITS, (batch_size, SEQ), generator=gen)
    seq = torch.cat([digits, torch.full((batch_size, 1), SEP), digits.sort(dim=1).values], dim=1)
    x, y = seq[:, :-1].clone(), seq[:, 1:].clone()
    y[:, :SEQ] = -100
    return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.n_heads)
        scores = scores.masked_fill(~self.tril[:T, :T], float("-inf"))
        out = scores.softmax(dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, max_len) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def train(model: nn.Module, data_fn, steps: int, lr: float) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for _ in range(steps):
        x, y = data_fn()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses
