import random
from functools import lru_cache

from solution import min_distance


def brute_force(a, b):
    """Independent answer: top-down recursion on suffixes."""
    @lru_cache(maxsize=None)
    def go(i, j):
        if i == len(a):
            return len(b) - j
        if j == len(b):
            return len(a) - i
        if a[i] == b[j]:
            return go(i + 1, j + 1)
        return 1 + min(go(i + 1, j), go(i, j + 1), go(i + 1, j + 1))
    return go(0, 0)


def test_examples():
    assert min_distance("horse", "ros") == 3
    assert min_distance("intention", "execution") == 5


def test_empty_strings():
    assert min_distance("", "") == 0
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "") == 3


def test_identical_and_single_characters():
    assert min_distance("same", "same") == 0
    assert min_distance("a", "b") == 1
    assert min_distance("a", "a") == 0
    assert min_distance("ab", "ba") == 2


def test_prefix_relations_only_need_inserts_or_deletes():
    assert min_distance("kitten", "kitt") == 2
    assert min_distance("kitt", "kitten") == 2
    assert min_distance("kitten", "sitting") == 3


def test_symmetric():
    pairs = [("sunday", "saturday"), ("abcdef", "azced"), ("zzzz", "zz"), ("flaw", "lawn")]
    for a, b in pairs:
        assert min_distance(a, b) == min_distance(b, a) == brute_force(a, b)


def test_random_strings_match_brute_force():
    rng = random.Random(19)
    for _ in range(300):
        a = "".join(rng.choice("abc") for _ in range(rng.randrange(0, 9)))
        b = "".join(rng.choice("abc") for _ in range(rng.randrange(0, 9)))
        want = brute_force(a, b)
        assert min_distance(a, b) == want, f"expected {want} for {a!r} -> {b!r}"


def test_long_strings_are_fast():
    assert min_distance("a" * 1000, "b" * 1000) == 1000
    assert min_distance("a" * 1000, "a" * 500 + "b" * 500) == 500
    assert min_distance("ab" * 500, "ba" * 500) == 2
    assert min_distance("abc" * 300, "bca" * 300 + "x") == 3
