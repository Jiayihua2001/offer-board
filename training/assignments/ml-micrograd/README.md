# A tiny scalar autograd engine

Writing a scalar reverse-mode autograd engine in 50 minutes is a real interview task at research labs, and it is the best single exercise for making backprop concrete: every node stores a local derivative, and the chain rule is just "multiply by the upstream gradient and accumulate". The design follows micrograd. Pure Python; the only numeric library allowed is `math`.

Implement a class `Value` with:

- `Value(data)`: attributes `data` (float) and `grad` (float, initially `0.0`). Keeps references to the inputs that produced it.
- Operators: addition, subtraction, multiplication, division, unary minus, and power with a Python number as the exponent (`x ** 2`). Each must also work with a plain `int` or `float` on either side (`2 * x`, `x + 1`, `1 / x`, `3 - x`).
- Methods `exp()`, `log()`, `tanh()`, `relu()`, each returning a new `Value`.
- `backward()`: sets `self.grad = 1.0` and fills in `grad` for every `Value` that contributed to `self`, visiting nodes in reverse topological order so each node's gradient is complete before it is propagated further.

Gradients accumulate: calling `backward()` twice without resetting doubles them, which is also what PyTorch does.

## Example

```
a = Value(2.0); b = Value(-3.0)
c = a * b + a          # c = -4
c.backward()
a.grad  -> -2.0        # dc/da = b + 1
b.grad  ->  2.0        # dc/db = a

x = Value(0.5)
y = (2 * x).tanh() ** 2
y.backward()
x.grad  -> 2 * tanh(1) * 2 * (1 - tanh(1)^2) = 1.280
```

## Constraints

- Every operation is O(1); `backward()` is O(nodes + edges) and visits each node once.
- `_backward` closures must add to the input grads (`+=`), never overwrite, so reused variables and diamond-shaped graphs are correct.
- `relu` has gradient 0 at and below 0; `log` and `**` may assume their inputs are in the valid domain.
- Tests compare against finite differences on composed expressions and against known derivatives; a last test trains a 1-D linear model with your engine.

## What the interviewer listens for

- Why reverse mode: one backward pass gives the gradient with respect to every input, so it costs about one forward pass regardless of the parameter count, while forward mode would need one pass per parameter.
- Why the topological order matters: a node's `_backward` multiplies by its own `grad`, which is only final once every consumer has pushed into it. A plain DFS that calls `_backward` on the way down gets diamond graphs wrong.
- The memory trade-off: the graph keeps every intermediate `data` alive until backward; that is the activation memory that gradient checkpointing trades for recomputation.
- Follow-up: how this generalizes to tensors (local derivatives become Jacobian-vector products, and broadcasting needs a sum over the broadcast axes in the backward).
