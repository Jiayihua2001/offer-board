# SGD with momentum, Adam, and a warmup-cosine schedule

"Implement Adam" is a favorite because it is short, everybody has used it, and almost nobody remembers the bias correction or why the first step is a sign step. Implement two optimizers with state and one schedule, in NumPy, following the PyTorch conventions so that you can defend every constant.

Each optimizer works on a single parameter array and returns the updated array. It keeps its state (velocity, moments, step count) inside the object and reads `self.lr` on every call, so a schedule can set `opt.lr` between steps.

- `SGD(lr=0.01, momentum=0.0)` with `step(p, g) -> p_new`. PyTorch form: `v = momentum * v + g`, `p_new = p - lr * v`. With `momentum = 0` this is plain gradient descent.
- `Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)` with `step(p, g) -> p_new`. Per-coordinate moments `m` and `v`, bias-corrected by `1 - beta1^t` and `1 - beta2^t` with `t` counting from 1, update `p - lr * m_hat / (sqrt(v_hat) + eps)`.
- `warmup_cosine(step, total_steps, warmup_steps, lr_max, lr_min=0.0) -> float`: linear warmup from 0 at `step = 0` to `lr_max` at `step = warmup_steps`, then cosine decay to `lr_min` at `step = total_steps`, staying at `lr_min` afterwards. The function is continuous in `step`.

## Example

```
opt = SGD(lr=0.1, momentum=0.9)
opt.step(np.array([1.0]), np.array([2.0]))   -> [0.8]    (v = 2)
opt.step(np.array([0.8]), np.array([2.0]))   -> [0.42]   (v = 0.9 * 2 + 2 = 3.8)

opt = Adam(lr=0.1)
opt.step(np.array([1.0, 1.0]), np.array([3.0, -0.001]))  -> [0.9, 1.1]  (first step is lr * sign(g), up to eps)

warmup_cosine(0, 100, 10, 1.0)    -> 0.0
warmup_cosine(10, 100, 10, 1.0)   -> 1.0
warmup_cosine(55, 100, 10, 1.0)   -> 0.5
warmup_cosine(100, 100, 10, 1.0)  -> 0.0
```

## Constraints

- Elementwise and allocation-free apart from the state: `step` must work for any parameter shape.
- `step` must not modify the arrays passed in (return a new array).
- A test runs both optimizers on an ill-conditioned quadratic and expects convergence; another changes `opt.lr` every step from the schedule.

## What the interviewer listens for

- Why Adam's bias correction exists: `m` starts at 0, so after one step it is `(1 - beta1) g`, a 10x underestimate; dividing by `1 - beta1^t` fixes the early steps and does nothing later. Without it the first step is about `3 x lr`, not `lr`.
- Why Adam is invariant to rescaling the gradient (both `m` and `sqrt(v)` scale the same way) and what that buys you: one learning rate works across layers with very different gradient magnitudes. `eps` is what breaks the invariance for tiny gradients.
- The momentum convention trap: `v = mu v + g; p -= lr v` versus `v = mu v + lr g; p -= v` are the same at constant `lr` and different under a schedule.
- Follow-up: where weight decay goes in AdamW versus L2 in the gradient, and why warmup matters for Adam (the second-moment estimate is noisy early, so the effective step is huge).
