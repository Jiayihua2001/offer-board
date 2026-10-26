import random

from solution import length_of_lis


def quadratic_dp(nums):
    """Independent answer: the O(n^2) DP."""
    best = []
    for i, x in enumerate(nums):
        best.append(1 + max((best[j] for j in range(i) if nums[j] < x), default=0))
    return max(best, default=0)


def test_examples():
    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
    assert length_of_lis([7, 7, 7, 7]) == 1


def test_empty_and_single():
    assert length_of_lis([]) == 0
    assert length_of_lis([42]) == 1


def test_monotone_inputs():
    assert length_of_lis(list(range(50))) == 50
    assert length_of_lis(list(range(50, 0, -1))) == 1
    assert length_of_lis([-3, -2, -2, -1, 0, 0, 1]) == 5


def test_strictness_and_negatives():
    assert length_of_lis([1, 3, 6, 7, 9, 4, 10, 5, 6]) == 6
    assert length_of_lis([-10**9, 10**9, -10**9, 10**9]) == 2
    assert length_of_lis([4, 10, 4, 3, 8, 9]) == 3


def test_random_lists_match_quadratic_dp():
    rng = random.Random(13)
    for _ in range(300):
        n = rng.randrange(0, 60)
        nums = [rng.randrange(-10, 11) for _ in range(n)]
        want = quadratic_dp(nums)
        assert length_of_lis(nums) == want, f"expected {want} for {nums}"


def test_large_input_is_fast():
    rng = random.Random(1)
    n_blocks, block = 1000, 100
    nums = []                                   # 1000 decreasing runs of 100: the LIS takes one per run
    for b in range(n_blocks):
        run = sorted((rng.randrange(block * 10) + b * block * 10 for _ in range(block)), reverse=True)
        nums.extend(run)
    assert len(nums) == 100_000
    assert length_of_lis(nums) == n_blocks
    assert length_of_lis(list(range(100_000))) == 100_000
    assert length_of_lis(list(range(100_000, 0, -1))) == 1
