"""A scalar autograd engine in the style of micrograd. Pure Python (math only)."""
import math


class Value:
    """A scalar with a gradient and a record of how it was computed."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None       # closure that pushes self.grad onto the inputs
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __pow__(self, k):
        """self ** k for a Python number k."""
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    # -- elementwise functions ---------------------------------------------
    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def relu(self):
        raise NotImplementedError

    # -- reverse-mode differentiation ---------------------------------------
    def backward(self):
        """Set self.grad = 1 and propagate gradients to every ancestor in reverse topological order."""
        raise NotImplementedError
