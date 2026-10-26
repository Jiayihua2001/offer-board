# Daily Temperatures

Given a list of daily temperatures `temps`, return a list `answer` of the same length where `answer[i]` is the number of days you have to wait after day `i` for a **strictly** warmer temperature. If no later day is warmer, `answer[i] = 0`.

## Examples

```
daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])  -> [1, 1, 4, 2, 1, 1, 0, 0]
daily_temperatures([30, 40, 50, 60])                  -> [1, 1, 1, 0]
daily_temperatures([50, 50, 50, 60])                  -> [3, 2, 1, 0]
daily_temperatures([])                                -> []
```

## Constraints

- `0 <= len(temps) <= 3 * 10^5`
- `30 <= temps[i] <= 100`
- Target: O(n) time, O(n) extra space for the stack. O(n^2) will time out on the large (decreasing) test.

## What the interviewer listens for

- The brute force and its worst case named precisely: a strictly decreasing sequence makes every scan run to the end.
- The reason a stack is the right structure: the unresolved days, read from the most recent backwards, have increasing temperatures, and a new day resolves them in exactly that order. That is what "monotonic stack" means; say it in your own words.
- The amortised argument: each index is pushed once and popped at most once, so the nested-looking loop is O(n) in total.
- The strictness of the comparison (`<` vs `<=`) decided consciously and tested on a plateau such as `[50, 50, 60]`.

## Follow-up to think about

The same stack answers "next greater element", "largest rectangle in a histogram" and "stock span". What changes between those problems, and what stays identical?
