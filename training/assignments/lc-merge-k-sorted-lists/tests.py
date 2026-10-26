import random

from solution import ListNode, merge_k_lists


def build(values):
    head = None
    for v in reversed(values):
        head = ListNode(v, head)
    return head


def to_list(node, limit):
    out = []
    while node is not None:
        out.append(node.val)
        assert len(out) <= limit, "the result has more nodes than the inputs combined (a cycle?)"
        node = node.next
    return out


def check(value_lists):
    heads = [build(v) for v in value_lists]
    expected = sorted(v for vs in value_lists for v in vs)
    got = to_list(merge_k_lists(heads), len(expected))
    assert got == expected, f"expected {expected[:12]}..., got {got[:12]}..."


def test_example():
    heads = [build([1, 4, 5]), build([1, 3, 4]), build([2, 6])]
    assert to_list(merge_k_lists(heads), 8) == [1, 1, 2, 3, 4, 4, 5, 6]


def test_no_lists_at_all():
    assert merge_k_lists([]) is None


def test_only_empty_lists():
    assert merge_k_lists([None]) is None
    assert merge_k_lists([None, None, None]) is None


def test_single_list_and_mixed_empty():
    assert to_list(merge_k_lists([build([1, 2, 3])]), 3) == [1, 2, 3]
    assert to_list(merge_k_lists([None, build([7])]), 1) == [7]
    assert to_list(merge_k_lists([None, build([1]), None, build([0, 2])]), 3) == [0, 1, 2]


def test_duplicates_and_negatives():
    check([[-3, -3, 0], [-3, 5], [0]])
    check([[2, 2, 2], [2, 2], [2]])


def test_very_different_lengths():
    check([list(range(0, 1000, 2)), [1], [], list(range(1, 1000, 2))])
    check([[100], list(range(50))])


def test_random_against_sorting():
    rng = random.Random(127)
    for _ in range(300):
        k = rng.randrange(0, 7)
        value_lists = [sorted(rng.randrange(-5, 6) for _ in range(rng.randrange(0, 7))) for _ in range(k)]
        check(value_lists)


def test_many_lists_are_fast():
    rng = random.Random(131)
    k, per_list = 20_000, 10
    value_lists = [sorted(rng.randrange(-10**9, 10**9) for _ in range(per_list)) for _ in range(k)]
    check(value_lists)
