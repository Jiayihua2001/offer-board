import random

from solution import group_anagrams


def normalize(groups):
    return sorted(sorted(list(g)) for g in groups)


def reference(strs):
    buckets = {}
    for s in strs:
        buckets.setdefault("".join(sorted(s)), []).append(s)
    return list(buckets.values())


def check(strs):
    got = group_anagrams(strs)
    flat = [s for g in got for s in g]
    assert sorted(flat) == sorted(strs), "every input string must appear exactly once across the groups"
    assert normalize(got) == normalize(reference(strs)), f"wrong grouping for {strs[:10]}..."


def test_example():
    got = normalize(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    assert got == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]], got


def test_empty_input_and_empty_string():
    assert group_anagrams([]) == [], "no strings means no groups"
    assert normalize(group_anagrams([""])) == [[""]]
    assert normalize(group_anagrams(["", ""])) == [["", ""]]


def test_single_string():
    assert normalize(group_anagrams(["a"])) == [["a"]]


def test_duplicates_stay_in_one_group():
    got = normalize(group_anagrams(["ab", "ba", "ab", "abc"]))
    assert got == [["ab", "ab", "ba"], ["abc"]], got


def test_same_letters_different_counts_are_different_groups():
    got = normalize(group_anagrams(["aab", "abb", "bba", "baa"]))
    assert got == [["aab", "baa"], ["abb", "bba"]], got


def test_random_against_reference():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randrange(0, 12)
        strs = ["".join(rng.choice("abc") for _ in range(rng.randrange(0, 4))) for _ in range(n)]
        check(strs)


def test_large_input_is_fast():
    rng = random.Random(11)
    strs = ["".join(rng.choice("abcdef") for _ in range(6)) for _ in range(100_000)]
    check(strs)
