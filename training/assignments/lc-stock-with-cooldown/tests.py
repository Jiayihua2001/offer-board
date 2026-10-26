import random

from solution import max_profit


def brute_force(prices):
    """Independent answer: try every action on every day, no memo (only for short lists)."""
    def best(day, holding, cooldown):
        if day == len(prices):
            return 0
        if cooldown:
            return best(day + 1, False, False)
        skip = best(day + 1, holding, False)
        if holding:
            return max(skip, prices[day] + best(day + 1, False, True))
        return max(skip, -prices[day] + best(day + 1, True, False))
    return best(0, False, False)


def test_examples():
    assert max_profit([1, 2, 3, 0, 2]) == 3
    assert max_profit([1]) == 0
    assert max_profit([5, 4, 3, 2, 1]) == 0
    assert max_profit([2, 1, 4]) == 3


def test_empty_and_two_days():
    assert max_profit([]) == 0
    assert max_profit([3, 3]) == 0
    assert max_profit([3, 8]) == 5
    assert max_profit([8, 3]) == 0


def test_cooldown_costs_a_trade():
    assert max_profit([1, 2, 1, 2]) == 1            # without the cooldown: 2 (sell day 1, buy day 2 is banned)
    assert max_profit([1, 2, 3, 0, 2]) == 3         # without cooldown: 4
    assert max_profit([1, 4, 2]) == 3


def test_strictly_increasing_is_one_trade():
    assert max_profit(list(range(1, 51))) == 49
    assert max_profit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9


def test_random_prices_match_brute_force():
    rng = random.Random(41)
    for _ in range(300):
        n = rng.randrange(0, 13)
        prices = [rng.randrange(0, 10) for _ in range(n)]
        want = brute_force(prices)
        assert max_profit(prices) == want, f"expected {want} for {prices}"


def test_large_inputs_are_fast():
    n = 100_000
    assert max_profit(list(range(n))) == n - 1
    assert max_profit([1, 3] * (n // 2)) == 2 * ((n + 2) // 4)      # one 2-profit trade every four days
    assert max_profit([5] * n) == 0
