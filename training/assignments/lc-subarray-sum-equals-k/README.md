# Subarray Sum Equals K

Given a list of integers `nums` and an integer `k`, return the number of contiguous, non-empty subarrays whose elements sum to exactly `k`. Subarrays at different positions count separately even if they hold the same values.

Values can be negative or zero, so a sliding window does not work.

## Examples

```
subarray_sum([1, 1, 1], 2)        -> 2      ([1,1] at 0..1 and at 1..2)
subarray_sum([1, 2, 3], 3)        -> 2      ([1,2] and [3])
subarray_sum([1, -1, 0], 0)       -> 3      ([1,-1], [-1,0], [0])
subarray_sum([0, 0, 0, 0], 0)     -> 10
```

## Constraints

- `0 <= len(nums) <= 2 * 10^5`
- `-1000 <= nums[i] <= 1000`, `-10^7 <= k <= 10^7`
- Target: O(n) time and O(n) extra space. O(n^2) over all subarrays will time out on the large test.

## What the interviewer listens for

- The O(n^2) prefix-sum version first (fix the start, extend the end), and the one-sentence reason a sliding window is wrong here: with negatives the window sum is not monotone in its length.
- The key insight, stated as an equation: `sum(nums[i:j]) == k` iff `P[j] - P[i] == k`, so each end index `j` only needs to know how many earlier prefix sums equal `P[j] - k`.
- Why `count[0] = 1` is there (the empty prefix) and the order of operations: look up before you insert, or you count the empty subarray when `k == 0`.
- The all-zeros case as a sanity check that counts are needed, not just a set of seen prefixes.

## Follow-up to think about

How would you return the number of subarrays whose sum is divisible by `k`, or the longest subarray summing to `k`, with the same idea?
