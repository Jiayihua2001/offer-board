import random

from solution import search


def rotated(values, k):
    return values[k:] + values[:k]


def test_examples():
    assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search([4, 5, 6, 7, 0, 1, 2], 3) == -1


def test_single_element():
    assert search([1], 1) == 0
    assert search([1], 0) == -1


def test_two_elements_both_rotations():
    assert search([1, 3], 3) == 1
    assert search([3, 1], 1) == 1
    assert search([3, 1], 3) == 0
    assert search([3, 1], 2) == -1


def test_not_rotated_and_rotated_by_one():
    nums = list(range(10, 40, 3))
    for t in nums:
        assert search(nums, t) == nums.index(t)
    r = rotated(nums, 1)
    for t in r:
        assert search(r, t) == r.index(t)
    assert search(r, 11) == -1


def test_target_at_every_position_for_every_rotation():
    base = [-7, -3, 0, 2, 5, 9, 14, 21]
    for k in range(len(base)):
        nums = rotated(base, k)
        for i, v in enumerate(nums):
            assert search(nums, v) == i, f"{nums} target {v}"
        for miss in (-8, -1, 1, 22):
            assert search(nums, miss) == -1, f"{nums} target {miss} should be absent"


def test_random_against_linear_scan():
    rng = random.Random(29)
    for _ in range(500):
        n = rng.randrange(1, 15)
        base = sorted(rng.sample(range(-30, 30), n))
        nums = rotated(base, rng.randrange(n))
        t = rng.randrange(-32, 32)
        expected = nums.index(t) if t in nums else -1
        assert search(nums, t) == expected, f"{nums} target {t}"


def test_many_queries_on_large_array_are_fast():
    rng = random.Random(37)
    n = 200_000
    base = sorted(rng.sample(range(-5 * n, 5 * n), n))
    nums = rotated(base, 61_803)
    position = {v: i for i, v in enumerate(nums)}
    for _ in range(30_000):
        if rng.random() < 0.7:
            i = rng.randrange(n)
            assert search(nums, nums[i]) == i
        else:
            t = rng.randrange(-5 * n, 5 * n)
            assert search(nums, t) == position.get(t, -1)
