# Coin Change

Given a list of coin denominations `coins` and an integer `amount`, return the fewest coins needed to make up exactly `amount`. You have an unlimited supply of every coin. If the amount cannot be made, return `-1`. An amount of `0` needs `0` coins.

## Examples

```
coin_change([1, 2, 5], 11)  -> 3     (5 + 5 + 1)
coin_change([2], 3)         -> -1
coin_change([1, 3, 4], 6)   -> 2     (3 + 3; greedy would answer 4 + 1 + 1)
coin_change([7], 0)         -> 0
```

## Constraints

- `1 <= len(coins) <= 12`, `1 <= coins[i] <= 2^31 - 1`, `0 <= amount <= 10^5`. Coins may repeat.
- Target: O(amount * len(coins)) time, O(amount) space.

## What the interviewer listens for

- You show that greedy is wrong with a two-line counterexample, then name the problem (unbounded knapsack, minimisation) instead of hand-waving "DP".
- You write the recurrence with its base case and the sentinel for unreachable amounts, and say why bottom-up over amounts is enough (each amount depends only on smaller ones).
- Complexity, and why memoised top-down recursion has the same cost but a stack depth of `amount / min(coins)` in Python.
- The BFS view (every coin is an edge of weight 1 between amounts) as an alternative that can stop early at the first solution.

## Follow-up

Count the number of distinct combinations that make the amount instead (Coin Change II). Why does the loop order over coins and amounts suddenly matter?
