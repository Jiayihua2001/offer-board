class MinStack:
    """Each entry stores (value, minimum of the stack including this value)."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, int]] = []

    def push(self, x: int) -> None:
        current_min = min(x, self._stack[-1][1]) if self._stack else x
        self._stack.append((x, current_min))

    def pop(self) -> None:
        self._stack.pop()

    def top(self) -> int:
        return self._stack[-1][0]

    def get_min(self) -> int:
        return self._stack[-1][1]
