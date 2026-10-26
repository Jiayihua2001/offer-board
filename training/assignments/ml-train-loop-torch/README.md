# A complete PyTorch training loop with early stopping

Applied rounds often skip the math and ask you to write the loop you have typed a hundred times, then probe the details that bite in practice: `zero_grad` placement, `model.eval()` plus `torch.no_grad()` in evaluation, loss accounting over uneven batches, and early stopping that actually restores the best weights. CPU only, a tiny model, seconds of compute.

The starter provides `make_moons(n, noise, seed)`, which returns a two-class dataset of interleaving half-circles (`X` of shape `(n, 2)`, `y` of shape `(n,)` with labels 0/1). Implement:

- `class MLP(nn.Module)`, constructed as `MLP(d_in=2, hidden=32, d_out=2, dropout=0.1)`: two hidden layers with ReLU and `nn.Dropout(dropout)` after each, then a linear output. `forward(x)` returns logits of shape `(N, d_out)`.
- `train_epoch(model, X, y, optimizer, loss_fn, batch_size=32) -> float`: one pass over the data in shuffled mini-batches, in training mode. Returns the mean training loss per sample (weight each batch by its size: the last batch is smaller).
- `evaluate(model, X, y, loss_fn) -> (loss, accuracy)`: full-batch, in eval mode, with gradients disabled. Both values are Python floats.
- `fit(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-2, patience=5, batch_size=32) -> history`: Adam and `nn.CrossEntropyLoss`. After every epoch evaluate on the validation set; stop when the validation loss has not improved (strictly decreased) for `patience` consecutive epochs. Restore the best weights into `model` before returning. `history` is a dict with lists `"train_loss"`, `"val_loss"`, `"val_acc"` (one entry per epoch run) and ints `"best_epoch"` and `"stopped_epoch"` (0-based indices into those lists).

## Example

```
X_tr, y_tr = make_moons(600, noise=0.15, seed=0)
X_va, y_va = make_moons(200, noise=0.15, seed=1)
torch.manual_seed(0)
model = MLP()
h = fit(model, X_tr, y_tr, X_va, y_va, epochs=60, lr=1e-2, patience=10)
max(h["val_acc"])                                  -> about 0.99
evaluate(model, X_va, y_va, nn.CrossEntropyLoss()) -> (0.03, 0.99)   (the restored best epoch)

h = fit(MLP(), X_tr, y_tr, X_va, y_va, epochs=50, lr=0.0, patience=4)
len(h["val_loss"])   -> 5     (no improvement is possible, so it stops after patience + 1 epochs)
```

## Constraints

- Mini-batches must be reshuffled every epoch and cover every sample exactly once; a probe module in the tests counts what it sees.
- `evaluate` must run with `model.training == False` and `torch.is_grad_enabled() == False` inside the forward pass.
- Keep it fast: the whole test file should run in a few seconds on CPU.

## What the interviewer listens for

- Why `model.eval()` and `torch.no_grad()` are two different things: one changes the semantics of dropout and batch norm, the other skips building the graph (memory and speed). Forgetting the first silently hurts validation metrics; forgetting the second blows up memory.
- Why early stopping on validation loss with a copy of the best state, rather than stopping on accuracy (coarse, noisy) or returning the last weights (by construction the last `patience` epochs were worse).
- Loss accounting: `loss.item()` is a per-sample mean over the batch, so epoch loss must be weighted by batch size; and `.item()` forces a sync, which is why you do it once per step, not per element.
- Follow-up: what changes with a `DataLoader` and a GPU (pinned memory, `non_blocking`, moving the model once and the batches per step), gradient clipping, and a learning-rate scheduler stepped once per epoch versus once per batch.
