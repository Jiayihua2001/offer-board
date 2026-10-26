import random
from collections import deque

from solution import coin_change


def bfs_reference(coins, amount):
    """Independent answer: BFS over amounts, each coin an edge of weight 1."""
    coins = [c for c in set(coins) if c <= amount]
    dist = {0: 0}
    queue = deque([0])
    while queue:
        a = queue.popleft()
        if a == amount:
            return dist[a]
        for c in coins:
            b = a + c
            if b <= amount and b not in dist:
                dist[b] = dist[a] + 1
                queue.append(b)
    return -1


def test_examples():
    assert coin_change([1, 2, 5], 11) == 3
    assert coin_change([2], 3) == -1
    assert coin_change([1, 3, 4], 6) == 2


def test_amount_zero_and_single_coin():
    assert coin_change([7], 0) == 0
    assert coin_change([7], 7) == 1
    assert coin_change([7], 14) == 2
    assert coin_change([7], 15) == -1


def test_coins_larger_than_amount_and_duplicates():
    assert coin_change([5, 10, 20], 3) == -1
    assert coin_change([3, 3, 3, 5], 9) == 3
    assert coin_change([2, 5, 10, 1], 27) == 4


def test_greedy_counterexamples():
    assert coin_change([1, 5, 6, 9], 11) == 2          # 5 + 6, greedy gives 9 + 1 + 1
    assert coin_change([4, 7], 16) == 4                 # 4 * 4, not 7 + 7 + ...
    assert coin_change([186, 419, 83, 408], 6249) == 20


def test_random_cases_match_bfs():
    rng = random.Random(21)
    for _ in range(300):
        k = rng.randrange(1, 6)
        coins = [rng.randrange(1, 15) for _ in range(k)]
        amount = rng.randrange(0, 80)
        want = bfs_reference(coins, amount)
        assert coin_change(coins, amount) == want, f"expected {want} for coins={coins}, amount={amount}"


def test_large_amount_is_fast():
    euro = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]      # a canonical system: greedy is optimal
    amount = 99_999
    greedy, rest = 0, amount
    for c in reversed(euro):
        greedy += rest // c
        rest %= c
    assert coin_change(euro, amount) == greedy
    coins = [186, 419, 83, 408]
    assert coin_change(coins, 100_000) == bfs_reference(coins, 100_000)
