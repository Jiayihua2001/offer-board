# Trapping Rain Water

Given a list of non-negative integers `height`, where `height[i]` is the height of a wall of width 1 standing at position `i`, return how much water is trapped between the walls after it rains. Water cannot escape sideways past a wall, but it runs off the two ends of the array.

## Examples

```
trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])  -> 6
trap([4, 2, 0, 3, 2, 5])                    -> 9
trap([5, 0, 5])                             -> 5
trap([3, 2, 1])                             -> 0
```

## Constraints

- `0 <= len(height) <= 3 * 10^5`
- `0 <= height[i] <= 10^5`
- Target: O(n) time. Two passes with O(n) extra space is good; the two-pointer version with O(1) extra space is what a strong answer ends with.

## What the interviewer listens for

- A per-column formula before any algorithm: `water[i] = min(max(height[:i+1]), max(height[i:])) - height[i]`, and that the total is just the sum of that over `i`.
- The O(n^2) version, then the prefix-max and suffix-max arrays that make it O(n), with the memory cost named.
- For the two-pointer version, the invariant stated precisely: when `height[lo] <= height[hi]`, the water at `lo` depends only on `left_max`, because some wall on the right (at least `height[hi]`) is at least as high as `left_max`. Interviewers ask you to defend this; have the argument ready.
- A quick sanity check on an example where the two-pointer order is not obvious (for instance `[4, 2, 0, 3, 2, 5]`).

## Follow-up to think about

The stack-based solution processes the water in horizontal layers instead of columns. When is that view the right one (think of the 2D version, "Trapping Rain Water II")?
