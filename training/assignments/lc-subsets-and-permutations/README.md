# Subsets and Permutations

Implement two functions over a list `nums` of distinct integers.

- `subsets(nums)` returns every subset (the power set), each as a list; the empty subset is included. There are `2^n` of them.
- `permutations(nums)` returns every ordering of all the elements. There are `n!` of them.

Any order of results is accepted, and any order inside each subset, but no result may appear twice.

## Examples

```
subsets([1, 2, 3])       -> [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
subsets([])              -> [[]]
permutations([1, 2, 3])  -> [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
permutations([])         -> [[]]
```

## Constraints

- `0 <= len(nums) <= 16` for subsets and `0 <= len(nums) <= 8` for permutations; elements are distinct.
- The output size is the cost, so the targets are O(n * 2^n) and O(n * n!). No `itertools` in the interview; the point is to write the recursion yourself.

## What the interviewer listens for

- You can state the recursion tree for each: "for each element, with it or without it" versus "for each unused element, place it next".
- You know why the output must be copied when it is recorded, and why the undo step (`pop`) restores the invariant for the next sibling.
- You can give the iterative alternatives (bitmask enumeration for subsets, next-permutation or Heap's algorithm for orderings) and state the complexity in terms of output size.
- What changes when `nums` has duplicates (sort, then skip equal siblings at the same depth), and why de-duplicating with a set afterwards is the slow way.

## Follow-up

Generate the subsets of exactly size `k` (combinations), and all permutations of a list with duplicates without producing repeats.
