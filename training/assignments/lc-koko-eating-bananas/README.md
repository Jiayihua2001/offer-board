# Koko Eating Bananas

There are `len(piles)` piles of bananas; pile `i` holds `piles[i]` bananas. Koko picks an eating speed `k` (bananas per hour). Each hour she chooses one pile and eats `k` bananas from it; if the pile has fewer than `k` left she finishes it and does not start another pile that hour.

Given `piles` and the number of hours `h` she has (always `h >= len(piles)`), return the **minimum** integer speed `k` such that she can eat every banana within `h` hours.

## Examples

```
min_eating_speed([3, 6, 7, 11], 8)        -> 4
min_eating_speed([30, 11, 23, 4, 20], 5)  -> 30
min_eating_speed([30, 11, 23, 4, 20], 6)  -> 23
min_eating_speed([1], 1)                  -> 1
```

## Constraints

- `1 <= len(piles) <= 10^4`
- `1 <= piles[i] <= 10^9`
- `len(piles) <= h <= 10^9`
- Target: O(n log(max(piles))) time, O(1) extra space. Scanning speeds one by one will time out when piles hold up to `10^9` bananas.

## What the interviewer listens for

- You name the search space (integer speeds from 1 to `max(piles)`) and explain why `max(piles)` is a valid upper bound before writing any code.
- The monotonicity argument, stated as the reason binary search applies: feasibility at `k` implies feasibility at `k + 1`.
- The feasibility check written with integer ceiling division, with the cost of each check named, and the overall complexity as `O(n log M)`.
- The binary search converges on the first feasible speed without an off-by-one: say which half you keep when `mid` is feasible and why `lo` ends at the answer.

## Follow-up to think about

This "binary search on the answer" pattern shows up constantly (split array largest sum, minimum days to make bouquets, capacity to ship packages). What two properties must a problem have for it to apply?
