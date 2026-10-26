# Train a tiny GPT in PyTorch

The NumPy forward pass becomes a real model: an `nn.Module` with embeddings, causal self-attention blocks, a tied head, and a training loop that fits it. The task is synthetic so that the whole thing trains on a laptop CPU in seconds: given six digits and a separator, emit the digits in sorted order. It is the same "sort" demo minGPT ships with, and a model that cannot learn it has a bug you should be able to find.

The starter provides the task: `N_DIGITS = 10`, `SEQ = 6`, `SEP = 10`, `VOCAB = 11`, `MAX_LEN = 13`, and `make_batch(batch_size, gen)` which returns `x` of shape `(B, 12)` (the digits, the separator, the first five sorted digits) and `y` of shape `(B, 12)` (the next token at every position, with `-100` at the first six positions where there is nothing to predict). Write:

- `class TinyGPT(nn.Module)` with `__init__(self, vocab_size, d_model, n_heads, n_layers, max_len)` and `forward(self, tokens) -> logits` of shape `(B, T, vocab_size)`. Token plus learned positional embeddings, `n_layers` pre-norm blocks with causal multi-head self-attention (hand-written or `nn.MultiheadAttention`) and a GELU MLP of width `4 * d_model`, a final LayerNorm, and an output projection tied to the token embedding. No dropout.
- `train(model, data_fn, steps, lr) -> list[float]`: `steps` optimizer steps on batches from `data_fn()`, cross-entropy with `ignore_index=-100`, returning the loss at every step.

## Example

```
model = TinyGPT(VOCAB, d_model=64, n_heads=4, n_layers=2, max_len=MAX_LEN)   # about 100k parameters
gen = torch.Generator().manual_seed(1)
losses = train(model, lambda: make_batch(64, gen), steps=600, lr=1e-3)         # a few seconds on CPU
losses[0] -> about 2.4 (log 11);  losses[-1] -> below 0.05
digits [3, 1, 4, 1, 5, 9] + SEP, decoded greedily one token at a time -> [1, 1, 3, 4, 5, 9]
```

## Constraints

- The tests train with `steps=600, lr=1e-3` and a 64-batch and require at least 90% exact-match accuracy on 300 held-out problems decoded autoregressively (the reference reaches about 99%). That has to finish in well under a minute on CPU.
- The model must be causal (`model.eval()` logits at position `t` must not depend on later tokens) and must accept any `T <= max_len`.
- Between 50k and 200k parameters for the configuration above.

## What the interviewer listens for

- The data pipeline is the model: the loss is masked to the positions where the answer is being produced, and the evaluation is autoregressive, not teacher-forced, so one wrong digit early poisons the rest.
- The cost: for this model the attention matrices are tiny, and the time goes to Python overhead and the optimizer, which is why a GPU is pointless at this size and why `torch.compile` and fused optimizers matter at the real size.
- The failure mode: forgetting the causal mask trains to a perfect loss and a useless model (it copies from the future), and a learning rate that is too high with no clipping diverges silently in the first hundred steps.
- Follow-up: what changes to scale this (warm-up and cosine decay, weight decay only on matrices, mixed precision, gradient accumulation), and how you would decide the model size for a token budget.
