import random
from collections import Counter

from solution import top_k_frequent


def check(nums, k):
    got = list(top_k_frequent(nums, k))
    counts = Counter(nums)
    assert len(got) == k, f"expected {k} values, got {len(got)}"
    assert len(set(got)) == k, f"values must be distinct: {got}"
    for v in got:
        assert v in counts, f"{v} does not occur in nums"
    top = sorted(counts.values(), reverse=True)[:k]
    mine = sorted((counts[v] for v in got), reverse=True)
    assert mine == top, f"returned frequencies {mine} but the top {k} frequencies are {top}"
    return got


def test_example_has_a_unique_answer():
    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]


def test_single_value():
    assert list(top_k_frequent([1], 1)) == [1]
    assert list(top_k_frequent([9, 9, 9, 9], 1)) == [9]


def test_k_equals_number_of_distinct_values():
    assert sorted(top_k_frequent([3, 1, 2, 1, 3, 3], 3)) == [1, 2, 3]


def test_ties_accept_any_boundary_choice():
    got = check([1, 2, 3, 1, 2, 3], 2)
    assert set(got) <= {1, 2, 3}
    got = check([5, 5, 6, 6, 7], 1)
    assert got[0] in (5, 6), f"7 occurs less often than 5 and 6, got {got}"
    check([5, 5, 6, 6, 7], 2)


def test_negative_and_zero_values():
    assert sorted(top_k_frequent([4, 4, -1, -1, 0], 2)) == [-1, 4]
    assert list(top_k_frequent([0, 0, 0, -3, -3, 8], 1)) == [0]


def test_random_against_frequency_check():
    rng = random.Random(61)
    for _ in range(300):
        n = rng.randrange(1, 40)
        nums = [rng.randrange(-5, 6) for _ in range(n)]
        d = len(set(nums))
        check(nums, rng.randrange(1, d + 1))


def test_large_input_is_fast():
    rng = random.Random(67)
    # Skewed: a few very common values, a long tail of rare ones, about 100k distinct values.
    nums = [rng.randrange(0, 200) for _ in range(100_000)]
    nums += [rng.randrange(0, 150_000) for _ in range(200_000)]
    rng.shuffle(nums)
    for k in (1, 100, 50_000):
        check(nums, k)
