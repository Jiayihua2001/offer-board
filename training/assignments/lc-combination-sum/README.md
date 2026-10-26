# Combination Sum

Given a list of distinct positive integers `candidates` and a positive integer `target`, return every unique combination of candidates whose numbers sum to `target`. The same candidate may be used any number of times. Two combinations are the same if they contain the same numbers with the same multiplicities, regardless of order; return each such combination exactly once, in any order and with any internal order.

## Examples

```
combination_sum([2, 3, 6, 7], 7)  -> [[2, 2, 3], [7]]
combination_sum([2, 3, 5], 8)     -> [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
combination_sum([2], 1)           -> []
```

## Constraints

- `1 <= len(candidates) <= 30`, `2 <= candidates[i] <= 40`, all distinct; `1 <= target <= 40`.
- Every test has fewer than 200 valid combinations, but the number of **ordered** sequences that sum to the target runs into the millions, so generating sequences and de-duplicating does not finish.
- Target: output-sensitive backtracking, roughly O(k * nodes of the pruned search tree) where `k` is the longest combination.

## What the interviewer listens for

- The start-index idea, explained as "combinations are non-decreasing sequences", so each multiset is produced exactly once with no de-duplication step.
- Reuse versus no reuse is a one-character difference in the recursive call (`i` versus `i + 1`); you can say which variant is which problem.
- Pruning by sorting and breaking early, with a feel for how much it saves.
- Copying the path on record, and that the running time is bounded by the output size times depth rather than by a neat closed form.

## Follow-up

Candidates may now contain duplicates and each may be used at most once (Combination Sum II). What changes in the loop?
