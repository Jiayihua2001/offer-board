# Longest Substring Without Repeating Characters

Given a string `s`, return the length of the longest contiguous substring in which no character appears twice. Only the length is needed, not the substring itself. The empty string has answer `0`.

Characters are compared exactly: `"a"` and `"A"` are different, and a space is a character like any other.

## Examples

```
length_of_longest_substring("abcabcbb")  -> 3    ("abc")
length_of_longest_substring("bbbbb")     -> 1    ("b")
length_of_longest_substring("pwwkew")    -> 3    ("wke"; "pwke" is a subsequence, not a substring)
length_of_longest_substring("")          -> 0
```

## Constraints

- `0 <= len(s) <= 3 * 10^5`
- `s` may contain any Unicode characters, so do not assume a small alphabet: one test has 150,000 distinct characters in a row.
- Aim for O(n) time and O(min(n, alphabet)) extra space. Anything that restarts from every index will time out on that test.

## What the interviewer listens for

- You start from the O(n^2) "extend from every start" idea and point out that a substring with a repeat can never be fixed by extending it further right, which is what justifies the window.
- The invariant you maintain: the window `s[left:right+1]` never contains a duplicate, and `left` only ever moves right.
- The classic bug: moving `left` to `last_seen[c] + 1` when that index is already behind `left` (`"abba"`), and why `max(left, ...)` fixes it.
- Complexity: each index enters and leaves the window at most once, so the two-pointer loop is O(n) even though it looks nested.

## Follow-up to think about

How does the solution change if you must return the substring itself, and if at most `k` repeats of each character are allowed?
