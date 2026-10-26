import random
from collections import Counter

from solution import three_sum


def reference_triplets(nums):
    """O(n^2): sort, and for each pair look up the third value at a later index."""
    a = sorted(nums)
    last = {v: i for i, v in enumerate(a)}
    out = set()
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            k = last.get(-(a[i] + a[j]))
            if k is not None and k > j:
                out.add((a[i], a[j], a[k]))
    return out


def check(nums):
    got = three_sum(nums)
    counts = Counter(nums)
    seen = set()
    for t in got:
        assert len(t) == 3, f"not a triplet: {t}"
        assert sum(t) == 0, f"{t} does not sum to 0"
        for v, c in Counter(t).items():
            assert counts[v] >= c, f"{t} uses {v} more times than nums contains it"
        key = tuple(sorted(t))
        assert key not in seen, f"duplicate triplet {t}"
        seen.add(key)
    expected = reference_triplets(nums)
    assert seen == expected, f"missing {sorted(expected - seen)[:5]}, extra {sorted(seen - expected)[:5]}"


def test_example():
    got = sorted(tuple(sorted(t)) for t in three_sum([-1, 0, 1, 2, -1, -4]))
    assert got == [(-1, -1, 2), (-1, 0, 1)], got


def test_too_few_elements():
    assert three_sum([]) == []
    assert three_sum([0]) == []
    assert three_sum([1, -1]) == []


def test_no_solution():
    assert three_sum([0, 1, 1]) == []
    assert three_sum([1, 2, 3, 4]) == []


def test_all_zeros_gives_one_triplet():
    assert [sorted(t) for t in three_sum([0, 0, 0, 0, 0])] == [[0, 0, 0]]


def test_duplicates_reported_once():
    check([-2, 0, 1, 1, 2])
    check([-1, -1, -1, 2, 2, 2, 0, 0])
    check([3, -3, 0, 0, 3, -3, 0])


def test_random_against_reference():
    rng = random.Random(42)
    for _ in range(300):
        n = rng.randrange(0, 16)
        check([rng.randrange(-6, 7) for _ in range(n)])


def test_large_input_is_fast():
    rng = random.Random(8)
    nums = [rng.randrange(-20_000, 20_001) for _ in range(3000)]
    check(nums)
