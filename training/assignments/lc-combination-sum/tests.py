import random
from itertools import product

from solution import combination_sum


def count_combinations(candidates, target):
    """Independent count of multisets summing to target (the coin-change-II DP)."""
    ways = [1] + [0] * target
    for c in candidates:
        for t in range(c, target + 1):
            ways[t] += ways[t - c]
    return ways[target]


def brute_force(candidates, target):
    """Independent enumeration: choose how many copies of each candidate to use."""
    out = set()
    for counts in product(*(range(target // c + 1) for c in candidates)):
        if sum(k * c for k, c in zip(counts, candidates)) == target:
            out.add(tuple(sorted(c for k, c in zip(counts, candidates) for _ in range(k))))
    return out


def check(candidates, target):
    got = combination_sum(list(candidates), target)
    assert isinstance(got, list), f"expected a list, got {type(got).__name__}"
    normal = []
    for combo in got:
        assert isinstance(combo, list) and combo, f"bad combination {combo!r}"
        assert sum(combo) == target, f"{combo} does not sum to {target}"
        assert set(combo) <= set(candidates), f"{combo} uses a number not in {candidates}"
        normal.append(tuple(sorted(combo)))
    assert len(set(normal)) == len(normal), "the same combination was returned more than once"
    assert len(normal) == count_combinations(candidates, target), \
        f"expected {count_combinations(candidates, target)} combinations, got {len(normal)}"
    return set(normal)


def test_examples():
    assert check([2, 3, 6, 7], 7) == {(2, 2, 3), (7,)}
    assert check([2, 3, 5], 8) == {(2, 2, 2, 2), (2, 3, 3), (3, 5)}


def test_no_solution():
    assert combination_sum([2], 1) == []
    assert combination_sum([3, 5], 4) == []
    assert combination_sum([4, 6], 9) == []


def test_single_candidate():
    assert combination_sum([7], 7) == [[7]]
    assert combination_sum([7], 21) == [[7, 7, 7]]
    assert combination_sum([7], 20) == []


def test_candidate_order_does_not_matter():
    assert check([7, 3, 2, 6], 7) == {(2, 2, 3), (7,)}
    assert check([5, 2, 3], 8) == {(2, 2, 2, 2), (2, 3, 3), (3, 5)}


def test_random_cases_match_brute_force():
    rng = random.Random(71)
    for _ in range(200):
        candidates = rng.sample(range(2, 10), rng.randrange(1, 5))
        target = rng.randrange(1, 21)
        want = brute_force(candidates, target)
        assert check(candidates, target) == want, f"wrong combinations for {candidates}, {target}"


def test_many_combinations_but_far_more_orderings():
    got = check([2, 3, 5, 7, 11, 13], 40)          # 100-ish multisets; millions of ordered sequences
    assert (2,) * 20 in got and (13, 13, 7, 7) not in got and (7, 7, 13, 13) in got
    got = check([2, 3, 5, 7], 30)
    assert (3, 3, 3, 7, 7, 7) in got
