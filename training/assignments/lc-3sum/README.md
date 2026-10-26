# 3Sum

Given a list of integers `nums`, return all triplets `[a, b, c]` of values taken from three different positions such that `a + b + c == 0`. The result must not contain duplicate triplets: two triplets are the same if they hold the same three values, regardless of order.

Return the triplets in any order, and the values inside each triplet in any order. Return an empty list when there is no such triplet.

## Examples

```
three_sum([-1, 0, 1, 2, -1, -4])  -> [[-1, -1, 2], [-1, 0, 1]]   (any order)
three_sum([0, 1, 1])              -> []
three_sum([0, 0, 0, 0])           -> [[0, 0, 0]]
```

## Constraints

- `0 <= len(nums) <= 3000`
- `-10^5 <= nums[i] <= 10^5`
- Aim for O(n^2) time with O(1) extra space beyond the output (sorting in place is fine). O(n^3) will time out on the large test.

## What the interviewer listens for

- You say the O(n^3) brute force in one sentence, then reduce the problem to Two Sum with the first element fixed, and only then pick between a hash set and two pointers.
- Why sorting unlocks two pointers: with `nums` sorted, a sum that is too small can only be fixed by moving `lo` right, so each step discards candidates safely.
- Deduplication done by skipping equal neighbours rather than by stuffing tuples into a set, and the argument for why that does not miss any triplet.
- Complexity: O(n log n) for the sort plus O(n^2) for the scan, and that the output itself can be Theta(n^2) in size.

## Follow-up to think about

How do the two-pointer scan and the deduplication generalise to 4Sum, and what is the complexity then?
