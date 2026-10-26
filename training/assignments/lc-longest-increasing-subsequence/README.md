# Longest Increasing Subsequence

Given a list of integers `nums`, return the length of its longest strictly increasing subsequence. A subsequence keeps the original order but may skip elements. Return `0` for an empty list.

## Examples

```
length_of_lis([10, 9, 2, 5, 3, 7, 101, 18])  -> 4     (2, 3, 7, 101  or  2, 5, 7, 18)
length_of_lis([0, 1, 0, 3, 2, 3])            -> 4
length_of_lis([7, 7, 7, 7])                  -> 1     (strictly increasing)
```

## Constraints

- `0 <= len(nums) <= 10^5`, values in `[-10^9, 10^9]`.
- Target: O(n log n) time, O(n) space. The O(n^2) DP is a fine first answer but will not finish the 10^5 test.

## What the interviewer listens for

- You give the O(n^2) DP cleanly (state, recurrence, base case) and only then go after O(n log n). Jumping to the trick without being able to explain the DP is a red flag.
- The `tails` invariant, why it stays sorted, and why overwriting an entry never shortens the answer even though `tails` is not itself a valid subsequence.
- `bisect_left` versus `bisect_right` is exactly the difference between strictly increasing and non-decreasing.
- How to recover one actual subsequence (store predecessor indices), which the tails array alone cannot do.

## Follow-up

Count the number of longest increasing subsequences, or solve the version where "increasing" is a partial order on pairs (Russian doll envelopes).
