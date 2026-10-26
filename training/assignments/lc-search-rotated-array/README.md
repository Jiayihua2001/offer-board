# Search in Rotated Sorted Array

`nums` was sorted in ascending order with **distinct** values, then rotated at some unknown pivot: `[0, 1, 2, 4, 5, 6, 7]` may have become `[4, 5, 6, 7, 0, 1, 2]`. The rotation may also be by zero positions, so a plain sorted array is valid input.

Given `nums` and a `target`, return the index of `target` in `nums`, or `-1` if it is not present.

## Examples

```
search([4, 5, 6, 7, 0, 1, 2], 0)  -> 4
search([4, 5, 6, 7, 0, 1, 2], 3)  -> -1
search([1], 0)                    -> -1
search([3, 1], 1)                 -> 1
```

## Constraints

- `1 <= len(nums) <= 2 * 10^5`, all values distinct.
- `-10^9 <= nums[i], target <= 10^9`
- Target: O(log n) time per query, O(1) extra space. The large test asks thousands of queries on one array, so O(n) per query will time out.

## What the interviewer listens for

- You say the O(n) scan and then immediately ask what binary search needs: a way to decide which side to discard from `nums[lo]`, `nums[mid]` and `nums[hi]`.
- The key observation, said clearly: one of the two halves around `mid` is always fully sorted, and you can tell which with one comparison.
- The branch logic written without off-by-one errors: a range check `nums[lo] <= target < nums[mid]` against the sorted half, and the else branch handles everything else.
- You can explain what breaks with duplicates (`[1, 1, 1, 0, 1]`) and why the worst case degrades to O(n).

## Follow-up to think about

Alternative plan: find the rotation pivot with one binary search, then do an ordinary binary search on the right segment. Same complexity; which one would you rather code under time pressure, and why?
