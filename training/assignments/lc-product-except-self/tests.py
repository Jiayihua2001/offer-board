import random

from solution import product_except_self


def brute_force(nums):
    out = []
    for i in range(len(nums)):
        p = 1
        for j, x in enumerate(nums):
            if j != i:
                p *= x
        out.append(p)
    return out


def test_examples():
    assert list(product_except_self([1, 2, 3, 4])) == [24, 12, 8, 6]
    assert list(product_except_self([-1, 1, 0, -3, 3])) == [0, 0, 9, 0, 0]


def test_two_elements():
    assert list(product_except_self([3, 5])) == [5, 3]
    assert list(product_except_self([0, 7])) == [7, 0]


def test_one_zero_zeroes_everything_but_its_own_slot():
    assert list(product_except_self([4, 0, 2])) == [0, 8, 0]


def test_two_zeros_zero_everything():
    assert list(product_except_self([2, 0, 0, 5])) == [0, 0, 0, 0]
    assert list(product_except_self([0, 0])) == [0, 0]


def test_negatives_and_ones():
    assert list(product_except_self([-2, -3, 1, -1])) == [3, 2, -6, 6]
    assert list(product_except_self([1, 1, 1, 1])) == [1, 1, 1, 1]


def test_random_against_brute_force():
    rng = random.Random(21)
    for _ in range(300):
        n = rng.randrange(2, 12)
        nums = [rng.randrange(-4, 5) for _ in range(n)]
        assert list(product_except_self(nums)) == brute_force(nums), f"wrong for {nums}"


def test_large_input_is_fast():
    rng = random.Random(9)
    nums = [rng.choice([-1, 1]) for _ in range(200_000)]
    nums[777] = 2
    nums[150_000] = 3
    out = list(product_except_self(nums))
    total = 1
    for x in nums:
        total *= x
    assert len(out) == len(nums)
    assert out[777] == total // 2, "out[i] must be the product of all other elements"
    assert out[150_000] == total // 3
    assert all(abs(v) == 6 for i, v in enumerate(out) if i not in (777, 150_000))
