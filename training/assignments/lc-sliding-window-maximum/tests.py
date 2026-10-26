import random

from solution import max_sliding_window


def brute_force(nums, k):
    return [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]


def block_reference(nums, k):
    """Independent O(n) method: prefix max inside blocks of size k and suffix max inside blocks,
    so max(nums[i:i+k]) == max(suffix[i], prefix[i+k-1])."""
    n = len(nums)
    prefix = [0] * n
    suffix = [0] * n
    for i in range(n):
        prefix[i] = nums[i] if i % k == 0 else max(prefix[i - 1], nums[i])
    for i in range(n - 1, -1, -1):
        suffix[i] = nums[i] if (i == n - 1 or (i + 1) % k == 0) else max(suffix[i + 1], nums[i])
    return [max(suffix[i], prefix[i + k - 1]) for i in range(n - k + 1)]


def test_example():
    assert list(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3)) == [3, 3, 5, 5, 6, 7]


def test_single_element_and_window_equal_to_length():
    assert list(max_sliding_window([1], 1)) == [1]
    assert list(max_sliding_window([4, 3, 2, 1], 4)) == [4]
    assert list(max_sliding_window([-2, -9, -4], 3)) == [-2]


def test_window_of_one_is_the_array():
    nums = [3, -1, 4, 1, -5, 9]
    assert list(max_sliding_window(nums, 1)) == nums


def test_monotonic_inputs():
    assert list(max_sliding_window([1, 2, 3, 4, 5], 2)) == [2, 3, 4, 5]
    assert list(max_sliding_window([5, 4, 3, 2, 1], 2)) == [5, 4, 3, 2]


def test_old_maximum_must_leave_the_window():
    assert list(max_sliding_window([5, 1, 1, 1, 1], 2)) == [5, 1, 1, 1]
    assert list(max_sliding_window([9, 8, 7, 1, 2, 3], 3)) == [9, 8, 7, 3]


def test_duplicates():
    assert list(max_sliding_window([2, 2, 2, 1, 1, 3], 2)) == [2, 2, 2, 1, 3]
    assert list(max_sliding_window([7, 7, 7, 7], 3)) == [7, 7]


def test_random_against_brute_force():
    rng = random.Random(109)
    for _ in range(400):
        n = rng.randrange(1, 20)
        k = rng.randrange(1, n + 1)
        nums = [rng.randrange(-6, 7) for _ in range(n)]
        assert list(max_sliding_window(nums, k)) == brute_force(nums, k), f"wrong for {nums}, k={k}"


def test_large_input_is_fast():
    rng = random.Random(113)
    nums = [rng.randrange(-10**9, 10**9) for _ in range(300_000)]
    k = 50_000
    assert list(max_sliding_window(nums, k)) == block_reference(nums, k)
