# Word Break

Given a string `s` and a list of dictionary words `word_dict`, return `True` if `s` can be written as a sequence of dictionary words concatenated together, and `False` otherwise. A word may be reused any number of times. The empty string counts as breakable.

## Examples

```
word_break("leetcode", ["leet", "code"])                        -> True
word_break("applepenapple", ["apple", "pen"])                   -> True
word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])  -> False
word_break("aaaaab", ["a", "aa", "aaa"])                        -> False
```

## Constraints

- `0 <= len(s) <= 300`, `1 <= len(word_dict) <= 1000`, words of length 1 to 20, lowercase; the dictionary may contain duplicates.
- Target: O(n * W) substring checks where `W` is the longest word length, O(n) space. The tests include the exponential worst case for memo-less recursion.

## What the interviewer listens for

- You start from the recursive formulation and name the overlapping subproblems (prefixes or suffixes), then memoise or go bottom-up, rather than saying "DP" as a magic word.
- The dictionary goes in a set, and you can state the cost of a lookup and of the slice `s[j:i]`.
- The `max_word_len` bound on the inner loop, and what the complexity is with and without it.
- Why greedy fails: with `["aa", "a", "ab"]` and `s = "aab"`, taking the longest match first (`"aa"`) leaves `"b"` and misses `"a" + "ab"`.

## Follow-up

Return every possible segmentation (Word Break II). Why can the number of answers blow up, and how does memoisation on suffixes still help?
