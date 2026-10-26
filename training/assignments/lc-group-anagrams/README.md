# Group Anagrams

Given a list of strings `strs`, group the anagrams together and return the groups as a list of lists. Two strings are anagrams when one can be rearranged into the other, so every group holds strings with exactly the same letters with the same multiplicities.

Return the groups in any order, and the strings inside a group in any order. Every input string must appear in exactly one group, with duplicates kept (two copies of `"ab"` go in the same group as two entries). A string with no anagram partner is a group of one.

## Examples

```
group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
  -> [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]   (any order)

group_anagrams([""])   -> [[""]]
group_anagrams(["a"])  -> [["a"]]
```

## Constraints

- `0 <= len(strs) <= 10^5`
- `0 <= len(strs[i]) <= 100`, lowercase English letters only.
- Aim for O(n k log k) time with `k` the maximum string length, or O(n k) with a counting key. O(n) extra space for the buckets.

## What the interviewer listens for

- You name the brute force (compare every pair with a letter count) and its O(n^2 k) cost before improving it.
- The key insight, stated plainly: map every string to a canonical form, so grouping becomes a hash-map lookup.
- The trade-off between the two keys: sorting costs O(k log k) per string and is simple; a 26-letter count tuple is O(k) but needs a hashable (immutable) key and assumes a fixed alphabet.
- You can say why a `list` cannot be a dict key and what you would do for Unicode input.

## Follow-up to think about

How would you stream this over a file far larger than memory, if you only needed the size of the largest group?
