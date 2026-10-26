# Median of Two Sorted Arrays

Given two sorted lists of integers `a` and `b` with `m + n >= 1` elements in total (either one may be empty), return the median of all `m + n` values as a `float`. For an odd total that is the middle value; for an even total it is the mean of the two middle values.

The O(m + n) merge is the acceptable first answer in an interview and the tests will not reject it; the target that interviewers wait for is **O(log(min(m, n)))** by binary searching the partition point.

## Examples

```
find_median_sorted_arrays([1, 3], [2])      -> 2.0
find_median_sorted_arrays([1, 2], [3, 4])   -> 2.5
find_median_sorted_arrays([], [5])          -> 5.0
find_median_sorted_arrays([1, 1], [1, 1])   -> 1.0
```

## Constraints

- `0 <= m, n <= 10^5` and `m + n >= 1`
- `-10^6 <= values <= 10^6`, each list sorted in non-decreasing order, duplicates allowed.
- Target: O(log(min(m, n))) time, O(1) extra space. Returning the answer to within `1e-9` is enough.

## What the interviewer listens for

- The O(m + n) merge (or the "walk two pointers until the middle" refinement with O(1) space) stated clearly, then a reason it is not optimal: you only need the boundary of the left half, not the order of everything before it.
- The reformulation: pick `i` elements from `a` and `j` from `b` to form the left half, with `i + j` fixed; the split is valid exactly when every left element is at most every right element, which reduces to two comparisons across the boundary.
- Why you binary search on the shorter array (so that `j` always stays in range) and how the sentinels `-inf` / `+inf` remove every edge case at the ends.
- The odd and even cases handled without duplicating the search, and a dry run on a tiny example like `[1, 3]` and `[2]` before you trust the code.

## Follow-up to think about

Generalise to the k-th smallest element of two sorted arrays, and then of `p` sorted arrays. Which parts of the argument survive?
