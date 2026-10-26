# Implement Trie

Implement a prefix tree with three operations:

- `insert(word)` stores `word`.
- `search(word)` returns `True` if exactly `word` was inserted before, else `False`.
- `starts_with(prefix)` returns `True` if some inserted word starts with `prefix`, else `False`.

Words and prefixes are non-empty strings of lowercase letters. Inserting the same word twice is allowed and changes nothing.

## Examples

```
t = Trie()
t.insert("apple")
t.search("apple")       -> True
t.search("app")         -> False     ("app" is a prefix, not a stored word)
t.starts_with("app")    -> True
t.insert("app")
t.search("app")         -> True
```

## Constraints

- Up to `3 * 10^4` calls in total, strings of length 1 to 2000.
- Target: O(L) per operation for a string of length `L`, independent of the number of stored words; O(total characters) space.

## What the interviewer listens for

- Why a plain set of words answers `search` but not `starts_with` efficiently, and why a set of all prefixes costs O(L^2) characters per word.
- The node representation: a dict of children versus a fixed array of 26 slots, and the memory versus speed trade-off.
- The end-of-word flag is what separates `search` from `starts_with`; forgetting it is the usual bug.
- Each `Trie()` instance must own its data: mutable class-level defaults are a classic Python mistake, and the tests create several tries.

## Follow-up

Add `delete(word)`, which removes a word and frees nodes no longer needed, and `count_prefix(prefix)` in O(L) (keep a counter in each node).
