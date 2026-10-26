"""Merge k Sorted Lists. Keep the class and function names; the tests import both."""


class ListNode:
    def __init__(self, val: int = 0, next: "ListNode | None" = None) -> None:
        self.val = val
        self.next = next


def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
    """Merge k sorted linked lists into one sorted list and return its head (None if empty)."""
    raise NotImplementedError
