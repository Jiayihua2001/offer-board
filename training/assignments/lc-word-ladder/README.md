# Word Ladder

A transformation sequence from `begin_word` to `end_word` using a dictionary `word_list` is a sequence `begin_word -> s_1 -> s_2 -> ... -> s_k` in which every adjacent pair of words differs in exactly one letter, every `s_i` is in `word_list`, and `s_k == end_word`. `begin_word` itself does not need to be in `word_list`.

Return the number of words in the shortest such sequence, counting both ends, or `0` if there is none.

## Examples

```
ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])  -> 5
    hit -> hot -> dot -> dog -> cog

ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"])         -> 0
    "cog" is not in the list

ladder_length("hit", "hot", ["hot"])                                      -> 2
```

## Constraints

- All words have the same length `L`, `1 <= L <= 10`, lowercase letters only; `begin_word != end_word`.
- `1 <= len(word_list) <= 10^4`; it may contain duplicates.
- Target: O(N * L^2) time and O(N * L) space with wildcard buckets (or O(26 * N * L^2) with letter substitution). The O(N^2 * L) all-pairs adjacency build does not finish on the largest test.

## What the interviewer listens for

- You state the graph model and that a shortest chain means BFS, and that the answer counts nodes on the path rather than edges (the usual off-by-one).
- You compare the two ways of finding neighbours, 26 * L substitutions checked against a set versus wildcard buckets, and give the cost of each.
- Visited handling: mark a word when it is enqueued, not when it is dequeued, or the queue explodes on dense dictionaries.
- You can sketch bidirectional BFS and explain why it helps (each frontier grows like b^(d/2) instead of b^d).

## Follow-up

Return all shortest sequences (Word Ladder II). What extra bookkeeping does the BFS need so that reconstruction does not repeat work?
