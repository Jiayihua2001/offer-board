import random

from solution import min_eating_speed


def hours_needed(piles, k):
    return sum((p + k - 1) // k for p in piles)


def brute_force(piles, h):
    k = 1
    while hours_needed(piles, k) > h:
        k += 1
    return k


def check(piles, h):
    k = min_eating_speed(piles, h)
    assert k >= 1, f"speed must be positive, got {k}"
    assert hours_needed(piles, k) <= h, f"speed {k} needs {hours_needed(piles, k)} hours, more than {h}"
    assert k == 1 or hours_needed(piles, k - 1) > h, f"speed {k - 1} would also work, so {k} is not minimal"


def test_examples():
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23


def test_single_pile():
    assert min_eating_speed([1], 1) == 1
    assert min_eating_speed([10], 1) == 10
    assert min_eating_speed([10], 3) == 4
    assert min_eating_speed([10], 100) == 1


def test_hours_equal_to_piles_means_largest_pile():
    assert min_eating_speed([5, 9, 2, 7], 4) == 9


def test_plenty_of_time_means_speed_one():
    assert min_eating_speed([5, 9, 2, 7], 10**9) == 1


def test_large_piles_do_not_overflow_or_crawl():
    assert min_eating_speed([10**9], 2) == 500_000_000
    assert min_eating_speed([10**9, 10**9], 3) == 10**9
    assert min_eating_speed([10**9] * 4, 10**9) == 4


def test_random_against_brute_force():
    rng = random.Random(19)
    for _ in range(300):
        n = rng.randrange(1, 8)
        piles = [rng.randrange(1, 40) for _ in range(n)]
        h = rng.randrange(n, 3 * n + 20)
        assert min_eating_speed(piles, h) == brute_force(piles, h), f"wrong for {piles}, h={h}"


def test_large_input_is_fast():
    rng = random.Random(41)
    piles = [rng.randrange(1, 10**9) for _ in range(10_000)]
    for h in (10_000, 25_000, 10**6, 10**7):
        check(piles, h)
