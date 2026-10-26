# Min Stack

Design a stack that supports retrieving its minimum element in constant time. Implement a class `MinStack` with these methods, each **O(1)**:

- `push(x)`: push the integer `x`.
- `pop()`: remove the top element (its return value is ignored).
- `top()`: return the top element without removing it.
- `get_min()`: return the smallest element currently in the stack.

`pop`, `top` and `get_min` are only ever called on a non-empty stack.

## Examples

```
s = MinStack()
s.push(-2); s.push(0); s.push(-3)
s.get_min()  -> -3
s.pop()
s.top()      -> 0
s.get_min()  -> -2
```

```
s = MinStack()
s.push(2); s.push(2); s.push(5)
s.pop(); s.get_min()  -> 2
s.pop(); s.get_min()  -> 2     (the duplicate minimum is still there)
```

## Constraints

- Up to `2 * 10^5` operations; values fit in a 64-bit integer.
- Target: O(1) time per operation, O(n) space. Computing `min()` over the stack on each call will time out.

## What the interviewer listens for

- You name the trade: O(n) extra memory buys O(1) `get_min`, and you can say when that is the wrong trade (huge values, rare `get_min` calls).
- The invariant in one sentence: the auxiliary data at position `i` is the minimum of everything at or below `i`, so popping cannot invalidate anything.
- The duplicate-minimum bug in the two-stack variant (`<=` versus `<`) and how you test for it.
- Follow-up readiness: a "max stack" is the same idea; a stack with O(1) `get_min` and O(1) memory overhead is possible for integers with the encoding trick, but you should explain why it is fragile.

## Follow-up to think about

How would you build a queue with O(1) amortised `get_min`? (Two min-stacks, like the classic two-stack queue.)
