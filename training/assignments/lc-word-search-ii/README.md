# Word Search II

Given an `m x n` board of lowercase letters and a list of `words`, return every word that can be spelled by walking on the board: start at any cell, move to a horizontally or vertically adjacent cell at each step, and never use a cell twice within one word. Return each found word once, in any order.

You may mark cells in place while searching as long as you restore them; the tests pass you a copy.

## Examples

```
board = [
  ["o","a","a","n"],
  ["e","t","a","e"],
  ["i","h","k","r"],
  ["i","f","l","v"],
]
find_words(board, ["oath", "pea", "eat", "rain"])  -> ["eat", "oath"]     (any order)

find_words([["a","b"], ["c","d"]], ["abcd", "abdc", "ab"])  -> ["ab", "abdc"]   ("abcd" is not a path: b and c are not adjacent)
find_words([["a","a"]], ["aaa"])                            -> []             (cells cannot be reused)
```

## Constraints

- `1 <= m, n <= 15`, `1 <= len(words) <= 10^4`, words of length 1 to 12, lowercase; `words` may contain duplicates.
- Target: one trie-guided DFS over the board, O(m * n * 4 * 3^(L-1)) in the worst case but far less with pruning, plus O(total characters) to build the trie.
- The tests include a board that is entirely `a` with the words `a`, `aa`, ..., `aaaaaaaaaaaa`. A search that keeps exploring after a word has been found does not finish on it.

## What the interviewer listens for

- You start from Word Search I per word, give its cost O(W * m * n * 3^L), and explain that a trie shares the prefix work across words so the board is walked once.
- Visited marking by overwriting the cell with a sentinel (`"#"`) and restoring it on the way back, and why that is safe.
- Reporting each word once (clear the end-of-word marker) and pruning dead trie branches so that finished words stop guiding the search.
- The alternative of checking every prefix against a set of prefixes, and why the trie usually has the better constant factor.

## Follow-up

The word list is fixed but boards keep arriving (thousands of puzzles). What would you precompute, and does anything change if words may also be read diagonally?
