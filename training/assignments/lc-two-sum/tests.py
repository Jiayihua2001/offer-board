import random

from solution import two_sum


def check(nums, target):
    i, j = two_sum(nums, target)
    assert 0 <= i < j < len(nums), f"bad indices {i}, {j}"
    assert nums[i] + nums[j] == target, f"nums[{i}] + nums[{j}] != {target}"


def test_examples():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]


def test_duplicates_are_two_different_indices():
    assert two_sum([3, 3], 6) == [0, 1]


def test_negative_numbers():
    check([-3, 4, 3, 90], 0)


def test_answer_may_be_at_the_end():
    check(list(range(1, 50)) + [1000, 2000], 3000)


def test_large_input_is_fast():
    rng = random.Random(7)
    nums = [rng.randrange(-10**9, 10**9) for _ in range(200_000)]
    nums[123_456], nums[199_999] = 5, -5
    check(nums, 0)
