import random

from solution import find_median_sorted_arrays


def brute_force(a, b):
    merged = sorted(a + b)
    n = len(merged)
    if n % 2:
        return float(merged[n // 2])
    return (merged[n // 2 - 1] + merged[n // 2]) / 2


def check(a, b):
    got = find_median_sorted_arrays(list(a), list(b))
    expected = brute_force(a, b)
    assert abs(got - expected) < 1e-9, f"a={a[:8]}..., b={b[:8]}...: expected {expected}, got {got}"


def test_examples():
    assert abs(find_median_sorted_arrays([1, 3], [2]) - 2.0) < 1e-9
    assert abs(find_median_sorted_arrays([1, 2], [3, 4]) - 2.5) < 1e-9


def test_one_list_empty():
    assert abs(find_median_sorted_arrays([], [5]) - 5.0) < 1e-9
    assert abs(find_median_sorted_arrays([2, 3], []) - 2.5) < 1e-9
    assert abs(find_median_sorted_arrays([], [1, 2, 3]) - 2.0) < 1e-9


def test_all_of_one_list_before_the_other():
    assert abs(find_median_sorted_arrays([1, 2, 3], [10, 20]) - 3.0) < 1e-9
    assert abs(find_median_sorted_arrays([10, 20], [1, 2, 3]) - 3.0) < 1e-9
    assert abs(find_median_sorted_arrays([1, 2], [3, 4, 5, 6]) - 3.5) < 1e-9


def test_duplicates_and_negatives():
    assert abs(find_median_sorted_arrays([1, 1], [1, 1]) - 1.0) < 1e-9
    assert abs(find_median_sorted_arrays([1, 2, 2], [2, 2, 3]) - 2.0) < 1e-9
    assert abs(find_median_sorted_arrays([-5, -3], [-4, -2, -1]) - (-3.0)) < 1e-9


def test_very_uneven_lengths():
    check([1, 2, 3, 4, 5, 6, 7, 8], [3])
    check([4], [1, 2, 3, 5, 6, 7, 8, 9])
    check([100], list(range(50)))
    check(list(range(50)), [-100])


def test_random_against_brute_force():
    rng = random.Random(79)
    for _ in range(500):
        m = rng.randrange(0, 9)
        n = rng.randrange(0 if m else 1, 9)
        a = sorted(rng.randrange(-10, 11) for _ in range(m))
        b = sorted(rng.randrange(-10, 11) for _ in range(n))
        check(a, b)


def test_large_input():
    rng = random.Random(83)
    total = 200_000
    for m in (1, 50_000, 100_000):
        a = sorted(rng.randrange(-10**6, 10**6) for _ in range(m))
        b = sorted(rng.randrange(-10**6, 10**6) for _ in range(total - m))
        check(a, b)
