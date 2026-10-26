import random

from solution import subarray_sum


def brute_force(nums, k):
    count = 0
    for i in range(len(nums)):
        s = 0
        for j in range(i, len(nums)):
            s += nums[j]
            if s == k:
                count += 1
    return count


def test_examples():
    assert subarray_sum([1, 1, 1], 2) == 2
    assert subarray_sum([1, 2, 3], 3) == 2


def test_empty_and_single_element():
    assert subarray_sum([], 0) == 0, "no non-empty subarray exists"
    assert subarray_sum([5], 5) == 1
    assert subarray_sum([5], 4) == 0


def test_negatives_and_zero_target():
    assert subarray_sum([1, -1, 0], 0) == 3
    assert subarray_sum([-1, -1, 1], 0) == 1
    assert subarray_sum([3, -3, 3, -3], 0) == 4


def test_all_zeros_counts_every_subarray():
    assert subarray_sum([0, 0, 0, 0], 0) == 10


def test_no_match():
    assert subarray_sum([1, 2, 3], 7) == 0
    assert subarray_sum([2, 4, 6], 1) == 0


def test_random_against_brute_force():
    rng = random.Random(13)
    for _ in range(400):
        n = rng.randrange(0, 15)
        nums = [rng.randrange(-3, 4) for _ in range(n)]
        k = rng.randrange(-4, 5)
        assert subarray_sum(nums, k) == brute_force(nums, k), f"wrong for {nums}, k={k}"


def test_large_input_is_fast():
    rng = random.Random(2)
    nums = [rng.randrange(-5, 6) for _ in range(200_000)]
    got = subarray_sum(nums, 0)
    assert got > 0
    # Independent count from prefix sums, computed differently (sorted runs of equal prefix sums).
    prefix, s = [0], 0
    for x in nums:
        s += x
        prefix.append(s)
    prefix.sort()
    expected, i = 0, 0
    while i < len(prefix):
        j = i
        while j < len(prefix) and prefix[j] == prefix[i]:
            j += 1
        run = j - i
        expected += run * (run - 1) // 2
        i = j
    assert got == expected, f"expected {expected}, got {got}"
