import math

from solution import Value


def numeric_grad(f, args, i, eps=1e-6):
    """Central difference of f (a function of floats) with respect to argument i."""
    hi = list(args)
    lo = list(args)
    hi[i] += eps
    lo[i] -= eps
    return (f(*hi) - f(*lo)) / (2 * eps)


def test_add_mul_basic_gradients():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a
    assert math.isclose(c.data, -4.0)
    c.backward()
    assert math.isclose(c.grad, 1.0), "the root's grad is 1 after backward"
    assert math.isclose(a.grad, -2.0), f"dc/da should be b + 1 = -2, got {a.grad}"
    assert math.isclose(b.grad, 2.0), f"dc/db should be a = 2, got {b.grad}"


def test_python_numbers_on_either_side():
    x = Value(3.0)
    y = 2 * x + 1 - x / 4 + (1 - x) + 8 / x + (-x) + x ** 2
    expected = 2 * 3 + 1 - 3 / 4 + (1 - 3) + 8 / 3 + (-3) + 9
    assert math.isclose(y.data, expected)
    y.backward()
    # dy/dx = 2 - 1/4 - 1 - 8/x^2 - 1 + 2x
    assert math.isclose(x.grad, 2 - 0.25 - 1 - 8 / 9 - 1 + 6, rel_tol=1e-9)


def test_gradient_accumulates_when_a_variable_is_reused():
    a = Value(3.0)
    y = a * a + a             # dy/da = 2a + 1 = 7
    y.backward()
    assert math.isclose(a.grad, 7.0), f"expected 7 (each use contributes), got {a.grad}"


def test_known_derivatives_of_unary_functions():
    x = Value(0.7)
    x.tanh().backward()
    assert math.isclose(x.grad, 1 - math.tanh(0.7) ** 2, rel_tol=1e-9)
    x = Value(0.7)
    x.exp().backward()
    assert math.isclose(x.grad, math.exp(0.7), rel_tol=1e-9)
    x = Value(0.7)
    x.log().backward()
    assert math.isclose(x.grad, 1 / 0.7, rel_tol=1e-9)
    x = Value(0.7)
    (x ** 3).backward()
    assert math.isclose(x.grad, 3 * 0.7 ** 2, rel_tol=1e-9)
    x = Value(0.7)
    (1 / x).backward()
    assert math.isclose(x.grad, -1 / 0.7 ** 2, rel_tol=1e-9)


def test_relu_gradient_is_a_mask():
    x = Value(-1.5)
    r = x.relu()
    assert r.data == 0.0
    r.backward()
    assert x.grad == 0.0
    x = Value(2.5)
    (3 * x.relu()).backward()
    assert math.isclose(x.grad, 3.0)


def test_diamond_graph_needs_topological_order():
    x = Value(2.0)
    a = x * x
    b = x + a
    c = a * b                 # c = x^2 (x + x^2) = x^3 + x^4, dc/dx = 3x^2 + 4x^3 = 44
    c.backward()
    assert math.isclose(c.data, 24.0)
    assert math.isclose(x.grad, 44.0), f"expected 44, got {x.grad} (is every consumer done before a node propagates?)"


def test_composed_expression_matches_finite_differences():
    def f_float(x, y):
        return math.tanh(x * y + x ** 2) * math.exp(-y) / (1 + max(x, 0.0)) + math.log(1 + x * x) - 3 * y

    def f_value(x, y):
        return (x * y + x ** 2).tanh() * (-y).exp() / (1 + x.relu()) + (1 + x * x).log() - 3 * y

    for xv, yv in [(0.3, -0.8), (1.2, 0.5), (-0.9, 1.7)]:
        x, y = Value(xv), Value(yv)
        out = f_value(x, y)
        assert math.isclose(out.data, f_float(xv, yv), rel_tol=1e-12)
        out.backward()
        gx = numeric_grad(f_float, (xv, yv), 0)
        gy = numeric_grad(f_float, (xv, yv), 1)
        assert math.isclose(x.grad, gx, rel_tol=1e-5, abs_tol=1e-7), f"d/dx at {(xv, yv)}: {x.grad} vs {gx}"
        assert math.isclose(y.grad, gy, rel_tol=1e-5, abs_tol=1e-7), f"d/dy at {(xv, yv)}: {y.grad} vs {gy}"


def test_trains_a_linear_model_with_the_engine():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]                     # y = 1 + 2 x
    w, b = Value(0.0), Value(0.0)
    loss_val = None
    for _ in range(200):
        loss = sum(((w * x + b - y) ** 2 for x, y in zip(xs, ys)), Value(0.0)) / len(xs)
        w.grad = b.grad = 0.0
        loss.backward()
        w.data -= 0.1 * w.grad
        b.data -= 0.1 * b.grad
        loss_val = loss.data
    assert loss_val < 1e-6, f"final loss {loss_val}"
    assert math.isclose(w.data, 2.0, abs_tol=1e-3) and math.isclose(b.data, 1.0, abs_tol=1e-3)
