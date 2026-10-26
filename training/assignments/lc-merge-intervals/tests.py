import random

from solution import merge


def as_lists(result):
    return [list(x) for x in result]


def brute_force(intervals):
    """Merge any overlapping pair until nothing changes. O(n^3), for small inputs."""
    items = [list(x) for x in intervals]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a[0] <= b[1] and b[0] <= a[1]:
                    items[i] = [min(a[0], b[0]), max(a[1], b[1])]
                    del items[j]
                    changed = True
                    break
            if changed:
                break
    return sorted(items)


def event_sweep(intervals):
    """Independent O(n log n) reference: +1 at each start, -1 at each end, starts first on ties."""
    events = []
    for s, e in intervals:
        events.append((s, 0))
        events.append((e, 1))
    events.sort()
    out, depth, start = [], 0, None
    for x, kind in events:
        if kind == 0:
            if depth == 0:
                start = x
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                out.append([start, x])
    return out


def test_example():
    assert as_lists(merge([[1, 3], [2, 6], [8, 10], [15, 18]])) == [[1, 6], [8, 10], [15, 18]]


def test_touching_intervals_merge():
    assert as_lists(merge([[1, 4], [4, 5]])) == [[1, 5]]
    assert as_lists(merge([[1, 4], [5, 6]])) == [[1, 4], [5, 6]], "a gap of one keeps them apart"


def test_empty_and_single():
    assert as_lists(merge([])) == []
    assert as_lists(merge([[2, 2]])) == [[2, 2]]
    assert as_lists(merge([[3, 9]])) == [[3, 9]]


def test_unsorted_input_and_nesting():
    assert as_lists(merge([[5, 9], [1, 2], [2, 3]])) == [[1, 3], [5, 9]]
    assert as_lists(merge([[1, 10], [2, 3], [4, 5], [11, 12]])) == [[1, 10], [11, 12]]
    assert as_lists(merge([[2, 3], [1, 10]])) == [[1, 10]], "the end must be extended with max, not replaced"


def test_duplicates_and_zero_length():
    assert as_lists(merge([[1, 3], [1, 3], [1, 3]])) == [[1, 3]]
    assert as_lists(merge([[7, 7], [7, 7]])) == [[7, 7]]
    assert as_lists(merge([[3, 5], [5, 5], [6, 6]])) == [[3, 5], [6, 6]]


def test_random_against_brute_force():
    rng = random.Random(71)
    for _ in range(300):
        n = rng.randrange(0, 10)
        intervals = []
        for _ in range(n):
            s = rng.randrange(0, 20)
            intervals.append([s, s + rng.randrange(0, 6)])
        got = as_lists(merge([list(x) for x in intervals]))
        assert got == brute_force(intervals), f"wrong for {intervals}: got {got}"
        assert got == event_sweep(intervals)


def test_large_input_is_fast():
    rng = random.Random(73)
    intervals = []
    for _ in range(200_000):
        s = rng.randrange(0, 10**7)
        intervals.append([s, s + rng.randrange(0, 100)])
    got = as_lists(merge(intervals))
    assert got == event_sweep(intervals)
