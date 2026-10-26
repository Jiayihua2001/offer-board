# Partition Equal Subset Sum

Given a non-empty list of positive integers `nums`, return `True` if it can be split into two subsets with equal sums (every element goes to exactly one of them), and `False` otherwise.

## Examples

```
can_partition([1, 5, 11, 5])  -> True     ([1, 5, 5] and [11])
can_partition([1, 2, 3, 5])   -> False
can_partition([2, 2])         -> True
can_partition([7])            -> False
```

## Constraints

- `1 <= len(nums) <= 200`, `1 <= nums[i] <= 100`, so the target is at most 10,000.
- Target: O(n * target) time, O(target) space; the bitset trick runs around O(n * target / 64) in practice.

## What the interviewer listens for

- You reduce the problem to subset-sum with target `total / 2` and dispose of the odd total first.
- Brute force is 2^n subsets; the pseudo-polynomial DP exists because the sums are small integers, and you can say what "pseudo" means (the input size is measured in bits).
- The 1-D array and why the inner loop runs downward: an upward loop lets one number be used twice, which is the unbounded knapsack instead.
- Early exit once `reachable[target]` turns `True`, and the bitset version as the fast practical variant.

## Follow-up

Split into `k` equal-sum subsets instead of two. Why does this DP stop working, and what replaces it?
