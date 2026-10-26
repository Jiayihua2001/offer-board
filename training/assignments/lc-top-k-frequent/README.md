# Top K Frequent Elements

Given a list of integers `nums` and an integer `k`, return the `k` distinct values that occur most often in `nums`. Return them in any order. `k` is always between 1 and the number of distinct values.

When several values tie at the boundary, any choice among the tied values is accepted: for `[1, 1, 2, 2, 3]` with `k = 1`, both `[1]` and `[2]` are correct. What is never correct is returning a value whose frequency is lower than that of a value you left out.

## Examples

```
top_k_frequent([1, 1, 1, 2, 2, 3], 2)  -> [1, 2]      (any order)
top_k_frequent([1], 1)                 -> [1]
top_k_frequent([4, 4, -1, -1, 0], 2)   -> [4, -1]     (any order)
```

## Constraints

- `1 <= len(nums) <= 3 * 10^5`
- `-10^9 <= nums[i] <= 10^9`
- `1 <= k <= number of distinct values`
- Target: O(n log k) with a heap, or O(n) with bucket sort. O(k * d) over `d` distinct values (repeatedly taking the max) will time out on the large test.

## What the interviewer listens for

- Counting is the obvious first step; the interesting part is what you do with the `d` distinct counts, and you should say the sort-based O(d log d) answer out loud before improving it.
- Why a min-heap of size k (not a max-heap of size d) is the right tool when `k` is much smaller than `d`, and what it costs in the worst case `k = d`.
- The bucket-sort idea: frequencies live in a bounded integer range `[1, n]`, so you can avoid comparison sorting altogether. This is the same observation that makes counting sort linear.
- You handle ties deliberately: either side of the boundary is acceptable, but your code must not drop a value that strictly beats one you kept.

## Follow-up to think about

Top-k by frequency over a stream, with the counts changing as data arrives: what structure keeps the answer current in O(log k) per update, and when is an approximate answer (count-min sketch) good enough?
