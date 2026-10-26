import random

from solution import daily_temperatures


def brute_force(temps):
    out = [0] * len(temps)
    for i in range(len(temps)):
        for j in range(i + 1, len(temps)):
            if temps[j] > temps[i]:
                out[i] = j - i
                break
    return out


def jump_reference(temps):
    """Independent O(n) method: scan right to left, hopping along already computed answers."""
    n = len(temps)
    out = [0] * n
    for i in range(n - 2, -1, -1):
        j = i + 1
        while j < n and temps[j] <= temps[i]:
            if out[j] == 0:
                j = n
            else:
                j += out[j]
        if j < n:
            out[i] = j - i
    return out


def test_example():
    assert list(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])) == [1, 1, 4, 2, 1, 1, 0, 0]


def test_increasing_sequences():
    assert list(daily_temperatures([30, 40, 50, 60])) == [1, 1, 1, 0]
    assert list(daily_temperatures([30, 60, 90])) == [1, 1, 0]


def test_empty_and_single_day():
    assert list(daily_temperatures([])) == []
    assert list(daily_temperatures([75])) == [0]


def test_no_warmer_day_anywhere():
    assert list(daily_temperatures([90, 80, 70, 60])) == [0, 0, 0, 0]
    assert list(daily_temperatures([50, 50, 50])) == [0, 0, 0], "equal is not strictly warmer"


def test_plateau_then_rise():
    assert list(daily_temperatures([50, 50, 50, 60])) == [3, 2, 1, 0]
    assert list(daily_temperatures([60, 50, 50, 70, 50])) == [3, 2, 1, 0, 0]


def test_random_against_brute_force():
    rng = random.Random(89)
    for _ in range(300):
        n = rng.randrange(0, 20)
        temps = [rng.randrange(30, 40) for _ in range(n)]
        assert list(daily_temperatures(temps)) == brute_force(temps), f"wrong for {temps}"


def test_large_decreasing_input_is_fast():
    n = 300_000
    temps = list(range(n + 30, 30, -1)) + [n + 31]
    got = list(daily_temperatures(temps))
    assert got[-1] == 0
    assert all(got[i] == n - i for i in range(n)), "every day waits for the warm day at the very end"


def test_large_random_input_against_reference():
    rng = random.Random(97)
    temps = [rng.randrange(30, 101) for _ in range(300_000)]
    assert list(daily_temperatures(temps)) == jump_reference(temps)
