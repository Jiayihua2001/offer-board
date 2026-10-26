# Merge Intervals

Given a list of closed intervals `intervals`, where each interval is `[start, end]` with `start <= end`, merge every pair of overlapping intervals and return the result as a list of `[start, end]` pairs sorted by `start`. Intervals that merely touch (`[1, 4]` and `[4, 5]`) overlap at the point `4` and must be merged into `[1, 5]`. The input is in no particular order and may contain duplicates and zero-length intervals such as `[7, 7]`.

## Examples

```
merge([[1, 3], [2, 6], [8, 10], [15, 18]])  -> [[1, 6], [8, 10], [15, 18]]
merge([[1, 4], [4, 5]])                     -> [[1, 5]]
merge([[5, 9], [1, 2], [2, 3]])             -> [[1, 3], [5, 9]]
merge([])                                   -> []
```

## Constraints

- `0 <= len(intervals) <= 2 * 10^5`
- `0 <= start <= end <= 10^9`
- Target: O(n log n) time for the sort, O(n) space for the output. Pairwise merging until a fixed point is O(n^2) and will time out.

## What the interviewer listens for

- You spot that sorting by start is the whole trick, and you can say why: after sorting, an interval cannot reach back past the current merged interval to touch an earlier one.
- The overlap test written correctly for closed intervals (`start <= last_end`, not `<`), with the touching case called out.
- That the end must be extended with `max`, because a later interval can be entirely inside the current one (`[1, 10]` then `[2, 3]`).
- Complexity: O(n log n) dominated by the sort, a single O(n) sweep after it, and that the sweep is O(n) only because the input is sorted.

## Follow-up to think about

Insert one new interval into an already merged, sorted list in O(n) without re-sorting ("Insert Interval"); and how would you support many insertions efficiently?
