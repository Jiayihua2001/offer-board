# Merge k Sorted Lists

You are given a list `lists` of `k` singly linked lists, each sorted in non-decreasing order. Some entries may be `None` (an empty list), and `lists` itself may be empty. Merge them all into one sorted linked list and return its head, or `None` if there are no nodes at all.

Use the `ListNode` class defined in the starter (`val` and `next` attributes). The tests build the inputs from Python lists and read your result back by walking `next` pointers; they only check the sequence of values, so you may reuse the input nodes or build new ones.

## Examples

```
lists = [[1, 4, 5], [1, 3, 4], [2, 6]]   (as linked lists)
merge_k_lists(lists)  -> 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6

merge_k_lists([])            -> None
merge_k_lists([None, None])  -> None
merge_k_lists([None, [7]])   -> 7
```

## Constraints

- `0 <= k <= 2 * 10^4`, `N` = total number of nodes `<= 2 * 10^5`
- `-10^9 <= val <= 10^9`
- Target: O(N log k) time, O(k) extra space (plus the output). Merging the lists one by one into a growing result is O(N k) and will time out on the large test.

## What the interviewer listens for

- The three candidate approaches with their costs: sort everything (O(N log N)), sequential merge (O(N k)), heap or divide and conquer (O(N log k)); and why `log k` beats `log N` when k is small.
- How you make heap entries comparable when values tie: a tuple with a tiebreaker (the list index or a counter) so Python never tries to compare two `ListNode` objects.
- Pointer hygiene: a dummy head node so the empty case and the first append need no special code, and the tail pointer always advanced after linking.
- Edge cases enumerated before coding: no lists, all lists empty, a single list, lists of very different lengths.

## Follow-up to think about

If the k lists were sitting on k different machines, which approach maps to a streaming merge with bounded memory, and what is the per-element network cost?
