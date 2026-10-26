import random

from solution import can_partition


def brute_force(nums):
    """Independent answer: enumerate every subset with a bitmask."""
    total = sum(nums)
    if total % 2:
        return False
    for mask in range(1 << len(nums)):
        if sum(x for i, x in enumerate(nums) if mask >> i & 1) * 2 == total:
            return True
    return False


def test_examples():
    assert can_partition([1, 5, 11, 5]) is True
    assert can_partition([1, 2, 3, 5]) is False


def test_tiny_lists():
    assert can_partition([7]) is False
    assert can_partition([2, 2]) is True
    assert can_partition([2, 3]) is False
    assert can_partition([1, 1, 1, 1]) is True


def test_odd_total_is_impossible():
    assert can_partition([1, 2, 3, 4, 5]) is False
    assert can_partition([100, 99]) is False


def test_even_total_is_not_enough():
    assert can_partition([2, 2, 3, 5]) is False        # total 12, but no subset sums to 6
    assert can_partition([1, 3, 4, 8]) is True         # 1 + 3 + 4 = 8
    assert can_partition([3, 3, 3, 4, 5]) is True      # 3 + 3 + 3 = 4 + 5


def test_each_number_used_at_most_once():
    assert can_partition([1, 2, 5]) is False           # an upward loop would find 1 + 1 + 2 = 4 twice
    assert can_partition([3, 5]) is False
    assert can_partition([4, 4, 4, 12]) is True


def test_random_lists_match_brute_force():
    rng = random.Random(37)
    for _ in range(300):
        n = rng.randrange(1, 13)
        nums = [rng.randrange(1, 12) for _ in range(n)]
        want = brute_force(nums)
        assert can_partition(nums) is want, f"expected {want} for {nums}"


def test_large_lists_are_fast():
    rng = random.Random(6)
    half = [rng.randrange(1, 101) for _ in range(100)]
    nums = half + half[:]
    rng.shuffle(nums)
    assert can_partition(nums) is True                 # each half is one subset
    nums = [100] * 198 + [1, 3]                        # total 19804, target 9902: 2 mod 100 is unreachable
    assert can_partition(nums) is False
