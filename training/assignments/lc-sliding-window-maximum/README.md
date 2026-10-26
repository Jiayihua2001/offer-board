# Sliding Window Maximum

Given a list of integers `nums` and a window size `k` with `1 <= k <= len(nums)`, slide a window of `k` consecutive elements from the left end to the right end one step at a time, and return a list with the maximum of each window. There are `len(nums) - k + 1` windows.

## Examples

```
max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3)  -> [3, 3, 5, 5, 6, 7]
max_sliding_window([1], 1)                        -> [1]
max_sliding_window([5, 1, 1, 1, 1], 2)            -> [5, 1, 1, 1]
max_sliding_window([4, 3, 2, 1], 4)               -> [4]
```

## Constraints

- `1 <= k <= len(nums) <= 3 * 10^5`
- `-10^9 <= nums[i] <= 10^9`
- Target: O(n) time, O(k) extra space. O(nk) will time out on the large test (n = 300000, k = 50000); O(n log n) with a heap passes.

## What the interviewer listens for

- You cost the brute force as O(nk), not O(n), and say what the heap variant buys and costs (O(n log n), lazy removal of stale indices rather than true deletion).
- The dominance argument that drives the deque: an element that is both older and smaller than another can never be a window maximum, so it is safe to discard immediately.
- The invariant stated in full: the deque holds indices from the current window only, in increasing index order and strictly decreasing value order, so the front is always the answer.
- Amortised O(n): each index is pushed once and popped at most once, from either end. And the `<=` on the pop (equal values: keep the newer one) explained, not guessed.

## Follow-up to think about

Sliding window minimum is symmetric; what about a sliding window median, or the maximum over windows of varying sizes given as queries?
