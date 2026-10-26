import random
from itertools import combinations, permutations as iter_permutations
from math import factorial

from solution import permutations, subsets


def check_subsets(nums):
    got = subsets(list(nums))
    assert isinstance(got, list), f"expected a list, got {type(got).__name__}"
    assert len(got) == 2 ** len(nums), f"expected {2 ** len(nums)} subsets of {nums}, got {len(got)}"
    for s in got:
        assert isinstance(s, list), f"each subset must be a list, got {s!r}"
    want = {tuple(sorted(c)) for k in range(len(nums) + 1) for c in combinations(nums, k)}
    assert {tuple(sorted(s)) for s in got} == want, f"wrong subsets for {nums}"
    return got


def check_permutations(nums):
    got = permutations(list(nums))
    assert isinstance(got, list), f"expected a list, got {type(got).__name__}"
    assert len(got) == factorial(len(nums)), f"expected {factorial(len(nums))} permutations of {nums}, got {len(got)}"
    for p in got:
        assert isinstance(p, list), f"each permutation must be a list, got {p!r}"
    assert {tuple(p) for p in got} == set(iter_permutations(nums)), f"wrong permutations for {nums}"
    return got


def test_subsets_example():
    check_subsets([1, 2, 3])


def test_permutations_example():
    got = check_permutations([1, 2, 3])
    assert [3, 1, 2] in got


def test_empty_and_single_element():
    assert subsets([]) == [[]]
    assert permutations([]) == [[]]
    assert sorted(subsets([5])) == [[], [5]]
    assert permutations([5]) == [[5]]


def test_negative_and_zero_values():
    check_subsets([-1, 0, 7])
    check_permutations([-1, 0, 7])


def test_random_inputs_match_itertools():
    rng = random.Random(61)
    for _ in range(40):
        nums = rng.sample(range(-20, 21), rng.randrange(0, 7))
        check_subsets(nums)
        check_permutations(nums)


def test_large_outputs_are_fast():
    got = subsets(list(range(16)))
    assert len(got) == 65_536
    assert len({tuple(sorted(s)) for s in got}) == 65_536, "duplicate subsets"
    assert all(len(set(s)) == len(s) and set(s) <= set(range(16)) for s in got)
    got = permutations(list(range(8)))
    assert len(got) == 40_320
    assert len({tuple(p) for p in got}) == 40_320, "duplicate permutations"
    assert all(sorted(p) == list(range(8)) for p in got)
