# Number of Islands

Given an `m x n` grid of strings where `"1"` is land and `"0"` is water, return the number of islands. An island is a maximal group of land cells connected horizontally or vertically (not diagonally). Everything outside the grid is water.

You may modify the grid in place; the tests pass you a copy.

## Examples

```
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"],
]
num_islands(grid) -> 3

num_islands([["1","0"],
             ["0","1"]])  -> 2       (diagonal cells are not connected)

num_islands([]) -> 0
```

## Constraints

- `0 <= m, n <= 300`, so up to 90,000 cells, and the grid can be entirely land.
- Target: O(m * n) time; extra space O(min(m, n)) for a BFS queue, or O(m * n) for a visited set or a DFS stack.

## What the interviewer listens for

- You recognise "count connected components" and argue that one flood fill per unvisited land cell visits each cell a constant number of times, so the whole thing is O(m * n) despite the nested loops.
- You choose iterative BFS or DFS and can say why: a recursive DFS on a 300 x 300 all-land grid is 90,000 frames deep, which overflows Python's default stack.
- You raise the trade-off of mutating the input to mark visited cells versus keeping a separate visited set, and ask whether mutation is acceptable.
- You know union-find as an alternative and when it wins (streaming edits: "number of islands after each new land cell").

## Follow-up

The grid is too large for memory and arrives one row at a time. How would you count islands with only two rows in memory?
