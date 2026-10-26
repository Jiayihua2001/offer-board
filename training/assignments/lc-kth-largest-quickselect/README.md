# Kth Largest Element in an Array

Given a list of integers `nums` and an integer `k` with `1 <= k <= len(nums)`, return the `k`-th largest element in sorted order, counting duplicates. The 1st largest is the maximum, the `len(nums)`-th largest is the minimum, and in `[5, 5, 4]` the 2nd largest is `5`.

Sorting is the baseline. The interview target is average O(n) with quickselect; the tests also accept a heap, and they include an input full of repeated values, so make sure your partition does not fall into the quadratic case on duplicates.

## Examples

```
find_kth_largest([3, 2, 1, 5, 6, 4], 2)            -> 5
find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4)   -> 4
find_kth_largest([7], 1)                           -> 7
```

## Constraints

- `1 <= k <= len(nums) <= 3 * 10^5`
- `-10^9 <= nums[i] <= 10^9`
- Target: average O(n) time, O(1) extra space beyond the array (you may modify `nums`). O(n log k) with a heap is acceptable; O(n k) is not.

## What the interviewer listens for

- The ladder of answers with their costs: sort (O(n log n)), heap of size k (O(n log k), useful when k is small or the data streams), quickselect (average O(n), worst O(n^2)).
- Why quickselect is linear on average: the recurrence `T(n) = T(n/2) + O(n)` sums to O(n), unlike quicksort which recurses into both sides.
- What a random pivot buys you (no adversarial worst case) and what a three-way partition buys you (no quadratic blow-up on equal keys). Mentioning median-of-medians for a guaranteed O(n) is a bonus; implementing it is not expected.
- Clean index arithmetic: you want the element at sorted position `len(nums) - k`, and you can state which partition the answer falls into after each round.

## Follow-up to think about

If `nums` arrived as a stream too large for memory, which of the three approaches survives, and what would you keep in memory?
