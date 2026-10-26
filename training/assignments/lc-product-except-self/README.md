# Product of Array Except Self

Given a list of integers `nums`, return a list `out` of the same length where `out[i]` is the product of every element of `nums` except `nums[i]`.

Do it **without using division**, in O(n) time, and with O(1) extra space beyond the output list. Zeros are allowed and must be handled by the algorithm itself, not special-cased.

## Examples

```
product_except_self([1, 2, 3, 4])        -> [24, 12, 8, 6]
product_except_self([-1, 1, 0, -3, 3])   -> [0, 0, 9, 0, 0]
product_except_self([2, 0, 0, 5])        -> [0, 0, 0, 0]
```

## Constraints

- `2 <= len(nums) <= 2 * 10^5`
- `-30 <= nums[i] <= 30`; Python integers do not overflow, but say what you would do in a fixed-width language.
- Target: O(n) time, O(1) extra space (the output list does not count).

## What the interviewer listens for

- The O(n^2) brute force, then the division shortcut and exactly why it is rejected: a single zero makes `total / nums[i]` undefined, and two zeros make the "count the zeros" patch ugly.
- The key insight: `out[i] = prefix[i - 1] * suffix[i + 1]`, which is two linear passes.
- How you reach O(1) extra space: the output array doubles as the prefix array and the suffix is a running scalar.
- Overflow awareness: in C++ or Java the products would need 64-bit or modular arithmetic; in Python the cost shows up as big-int multiplication time instead.

## Follow-up to think about

How would you answer many queries of the form "product of `nums[l:r]` except `nums[i]`" quickly, and does division become acceptable under a prime modulus?
