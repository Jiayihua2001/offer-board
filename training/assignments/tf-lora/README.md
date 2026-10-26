# LoRA on a linear layer

Fine-tuning a 7B model by updating every weight needs optimizer state for 7B parameters. LoRA (low-rank adaptation) freezes the pretrained weight `W` and learns a rank-`r` update `W + (alpha / r) B A` with `A` of shape `(r, in)` and `B` of shape `(out, r)`, so the trainable count drops by orders of magnitude, a checkpoint is a few megabytes, and at inference the update can be folded into `W` so there is no extra latency. Implement it as a wrapper around `nn.Linear`.

```
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0)
    def forward(self, x) -> base(x) + (x @ A.T @ B.T) * (alpha / r)      when not merged
    def merge()      # fold the update into base.weight; forward then uses base only
    def unmerge()    # undo merge exactly
```

- `self.A` is an `nn.Parameter` of shape `(r, in_features)` with a small random init; `self.B` is an `nn.Parameter` of shape `(out_features, r)` initialized to zero, so at creation the module computes exactly what `base` computes.
- `base` is kept as `self.base` and frozen: its weight and bias do not require grad. `A` and `B` are the only trainable parameters.
- `self.scaling = alpha / r`. `merge()` and `unmerge()` must be idempotent (calling `merge()` twice adds the update once).

## Example

```
base = nn.Linear(32, 16);  lora = LoRALinear(base, r=4, alpha=8)
lora(x) == base(x)                                        # B = 0 at init
[n for n, p in lora.named_parameters() if p.requires_grad] -> ["A", "B"]     (4*32 + 16*4 = 192 trainable of 720)
lora.merge();  lora.base.weight == W0 + 2.0 * B @ A;  lora(x) unchanged
lora.unmerge(); lora.base.weight == W0
```

## Constraints

- `A` and `B` must be found by `named_parameters()` under those names, and nothing else may require grad.
- Training must stay in the factored form: no `(out, in)` matrix is built in `forward`.
- The tests fit a rank-2 target update with Adam in a few hundred steps; it should take a second or two on CPU.

## What the interviewer listens for

- Why `B = 0` and not `A = 0`: either gives a zero update at init, but with `B = 0` the gradient to `A` is zero on the first step while `B` gets a signal, so training starts cleanly; with both zero nothing ever moves.
- The cost: `r (in + out)` parameters and optimizer states instead of `in * out`; the forward adds two thin matmuls (`x A^T` then `. B^T`) that cost `O(r (in + out))` per token, and merging removes even that.
- The failure mode: `alpha` and `r` are coupled, so changing `r` without changing `alpha` changes the effective learning rate of the update (rank-stabilized LoRA uses `alpha / sqrt(r)`); and merging into a quantized base weight is lossy.
- Follow-up: which matrices to adapt (query and value were the original choice; all linear layers is now common), and how QLoRA keeps the base in 4-bit while the adapters train in bf16.
