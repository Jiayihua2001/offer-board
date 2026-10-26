# Edit Distance

Given two strings `word1` and `word2`, return the minimum number of operations needed to turn `word1` into `word2`. One operation inserts a character, deletes a character, or replaces a character with another.

## Examples

```
min_distance("horse", "ros")             -> 3     (horse -> rorse -> rose -> ros)
min_distance("intention", "execution")   -> 5
min_distance("", "abc")                  -> 3
min_distance("same", "same")             -> 0
```

## Constraints

- `0 <= len(word1), len(word2) <= 1000`; lowercase letters (any characters should work).
- Target: O(m * n) time; O(min(m, n)) space is expected once the full table works.

## What the interviewer listens for

- You define the subproblem in words ("prefixes of length i and j") before writing the recurrence, and you derive the three options from the three operations together with the base cases.
- You can argue why looking only at the last characters is enough: an optimal alignment ends in one of four ways (match, replace, insert, delete).
- Rolling rows: the order of updates, and which entry has to be saved before it is overwritten.
- Complexity, and that no algorithm is known to beat O(m * n) by more than polylog factors, which is why constant factors and space are what remain to optimise.

## Follow-up

Return one sequence of operations, not just the count. What do you need to store, and does the space optimisation survive?
