# Best Time to Buy and Sell Stock with Cooldown

`prices[i]` is the price of a stock on day `i`. You may complete as many transactions as you like (buy one share, later sell it), with two rules: you cannot hold more than one share at a time, and after you sell you must wait one day before buying again (cooldown). Return the maximum profit; return `0` for an empty list or when no profitable trade exists.

## Examples

```
max_profit([1, 2, 3, 0, 2])  -> 3     (buy 1, sell 3, cooldown, buy 0, sell 2)
max_profit([1])              -> 0
max_profit([5, 4, 3, 2, 1])  -> 0
max_profit([2, 1, 4])        -> 3     (buy 1, sell 4)
```

## Constraints

- `0 <= len(prices) <= 10^5`, `0 <= prices[i] <= 1000`.
- Target: O(n) time, O(1) extra space.

## What the interviewer listens for

- You draw the state machine (hold / sold / rest) and its transitions before writing anything, and can say what each state's value means ("best profit so far when ending the day in this state").
- Initial values: why `sold` starts at minus infinity (or 0, and why that also works here), `hold` at `-prices[0]`, and why the answer never comes from `hold`.
- The brute force (recursively try every buy/sell day pair, exponential), then a DP over days with O(n^2) transitions, then the O(n) state machine.
- How the same template handles the other variants: transaction fee, at most `k` transactions, no cooldown.

## Follow-up

Add a transaction fee and remove the cooldown. Which transition changes, and by how much?
