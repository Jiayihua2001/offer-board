# Two Sum

Given a list of integers `nums` and an integer `target`, return the indices of the two numbers that add up to `target`, as a list `[i, j]` with `i < j`. Exactly one answer exists, and you may not use the same element twice.

## Examples

```
two_sum([2, 7, 11, 15], 9)  -> [0, 1]
two_sum([3, 2, 4], 6)       -> [1, 2]
two_sum([3, 3], 6)          -> [0, 1]
```

## Constraints

- `2 <= len(nums) <= 10^5`
- Values and the target fit in a 64-bit integer.
- Aim for O(n) time.

## What the interviewer listens for

- You state the brute-force O(n^2) idea in one sentence and then improve it, rather than jumping straight to the trick.
- You handle duplicates correctly (`[3, 3]`), which is why the lookup happens before the insert.
- You can say what the hash map costs in memory and when sorting plus two pointers would be the better trade (when you need the values, not the indices, or memory is tight).

## Follow-up to think about

How would you return all pairs if several answers could exist, without reporting the same pair twice?
