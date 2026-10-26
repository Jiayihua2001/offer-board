import heapq


class ListNode:
    def __init__(self, val: int = 0, next: "ListNode | None" = None) -> None:
        self.val = val
        self.next = next


def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None:
    """Min-heap over the current heads: O(N log k) time, O(k) extra space."""
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node is not None]
    heapq.heapify(heap)                     # the index i breaks ties, so nodes are never compared
    dummy = tail = ListNode()
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next is not None:
            heapq.heappush(heap, (node.next.val, i, node.next))
    tail.next = None
    return dummy.next
