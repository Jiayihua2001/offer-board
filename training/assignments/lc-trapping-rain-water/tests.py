import random

from solution import trap


def brute_force(height):
    total = 0
    for i in range(len(height)):
        left = max(height[: i + 1])
        right = max(height[i:])
        total += min(left, right) - height[i]
    return total


def test_examples():
    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    assert trap([4, 2, 0, 3, 2, 5]) == 9


def test_too_short_to_hold_water():
    assert trap([]) == 0
    assert trap([7]) == 0
    assert trap([2, 5]) == 0


def test_monotonic_and_flat_hold_nothing():
    assert trap([1, 2, 3, 4, 5]) == 0
    assert trap([5, 4, 3, 2, 1]) == 0
    assert trap([3, 3, 3, 3]) == 0


def test_single_valley_and_plateau_inside():
    assert trap([5, 0, 5]) == 5
    assert trap([5, 0, 0, 5]) == 10
    assert trap([5, 2, 2, 5]) == 6
    assert trap([2, 0, 2, 0, 2]) == 4


def test_unequal_walls_use_the_lower_one():
    assert trap([9, 0, 3]) == 3
    assert trap([3, 0, 9]) == 3
    assert trap([1, 0, 10, 0, 1]) == 2


def test_random_against_brute_force():
    rng = random.Random(23)
    for _ in range(400):
        n = rng.randrange(0, 20)
        h = [rng.randrange(0, 8) for _ in range(n)]
        assert trap(h) == brute_force(h), f"wrong for {h}"


def test_large_input_is_fast():
    rng = random.Random(31)
    h = [rng.randrange(0, 1000) for _ in range(300_000)]
    got = trap(h)
    # Independent O(n) check with explicit prefix/suffix max arrays.
    n = len(h)
    left = [0] * n
    right = [0] * n
    m = 0
    for i in range(n):
        m = max(m, h[i])
        left[i] = m
    m = 0
    for i in range(n - 1, -1, -1):
        m = max(m, h[i])
        right[i] = m
    expected = sum(min(left[i], right[i]) - h[i] for i in range(n))
    assert got == expected, f"expected {expected}, got {got}"
