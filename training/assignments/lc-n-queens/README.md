# N-Queens

Place `n` queens on an `n x n` chessboard so that no two attack each other: no two share a row, a column or a diagonal.

- `count_n_queens(n)` returns the number of distinct solutions.
- `solve_n_queens(n)` returns one solution as a list of `n` strings of length `n`, with `"Q"` for a queen and `"."` for an empty square; return `[]` when there is no solution (`n = 2` and `n = 3`).

Any valid board is accepted.

## Examples

```
count_n_queens(4)  -> 2
solve_n_queens(4)  -> [".Q..", "...Q", "Q...", "..Q."]     (or its mirror image)
count_n_queens(1)  -> 1
solve_n_queens(1)  -> ["Q"]
solve_n_queens(3)  -> []
```

Counts for `n = 1..9`: `1, 0, 0, 2, 10, 4, 40, 92, 352`.

## Constraints

- `1 <= n <= 10`.
- Target: backtracking row by row with O(1) attack checks, so O(n) work per node of the search tree. Pruning keeps the tree far smaller than `n^n`; the unpruned `n^n` enumeration does not finish for `n = 10`.

## What the interviewer listens for

- Reducing the search from "any n squares" to one column choice per row, and then from `n!` column permutations to the pruned tree.
- The diagonal keys `r - c` and `r + c`, and why `r - c` needs an offset or a set rather than a plain list indexed by a possibly negative number.
- The difference between enumerating all solutions and stopping at the first one, and how to structure the recursion so one piece of code does both.
- Bitmask representation of the three constraint sets and what it buys (n = 12 to 14 in seconds in Python).

## Follow-up

Count solutions up to symmetry (rotations and reflections). How would you avoid generating mirror images in the first place?
