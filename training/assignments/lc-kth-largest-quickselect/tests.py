import random

from solution import find_kth_largest


def test_examples():
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4


def test_single_element_and_extremes():
    assert find_kth_largest([7], 1) == 7
    nums = [9, -2, 4, 4, 0, 11]
    assert find_kth_largest(list(nums), 1) == 11, "k = 1 is the maximum"
    assert find_kth_largest(list(nums), 6) == -2, "k = n is the minimum"


def test_duplicates_count_separately():
    assert find_kth_largest([5, 5, 4], 2) == 5
    assert find_kth_largest([5, 5, 4], 3) == 4
    assert find_kth_largest([2, 2, 2, 2], 3) == 2


def test_sorted_and_reverse_sorted_inputs():
    asc = list(range(1, 2001))
    assert find_kth_largest(list(asc), 1) == 2000
    assert find_kth_largest(list(asc), 2000) == 1
    assert find_kth_largest(list(reversed(asc)), 500) == 1501


def test_random_against_sorting():
    rng = random.Random(47)
    for _ in range(300):
        n = rng.randrange(1, 30)
        nums = [rng.randrange(-10, 11) for _ in range(n)]
        k = rng.randrange(1, n + 1)
        assert find_kth_largest(list(nums), k) == sorted(nums)[-k], f"wrong for {nums}, k={k}"


def test_large_input_with_k_in_the_middle_is_fast():
    rng = random.Random(53)
    nums = [rng.randrange(-10**9, 10**9) for _ in range(300_000)]
    k = 150_000
    expected = sorted(nums)[-k]
    assert find_kth_largest(list(nums), k) == expected


def test_many_duplicates_do_not_go_quadratic():
    rng = random.Random(59)
    nums = [rng.choice([1, 2, 3]) for _ in range(60_000)]
    expected = sorted(nums)[-30_000]
    assert find_kth_largest(list(nums), 30_000) == expected
    assert find_kth_largest([4] * 100_000, 50_000) == 4
