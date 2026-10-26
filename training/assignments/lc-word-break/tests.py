import random

from solution import word_break


def brute_force(s, words):
    """Independent answer: plain recursion over suffixes, no memo (only for short s)."""
    if s == "":
        return True
    return any(s.startswith(w) and brute_force(s[len(w):], words) for w in set(words))


def test_examples():
    assert word_break("leetcode", ["leet", "code"]) is True
    assert word_break("applepenapple", ["apple", "pen"]) is True
    assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False


def test_empty_string_and_single_characters():
    assert word_break("", ["a"]) is True
    assert word_break("a", ["a"]) is True
    assert word_break("a", ["b"]) is False
    assert word_break("b", ["bb", "bbb"]) is False


def test_reuse_and_duplicates_in_dictionary():
    assert word_break("abab", ["ab"]) is True
    assert word_break("ababa", ["ab", "ab"]) is False
    assert word_break("aaaaaaa", ["aaa", "aaaa", "aaa"]) is True


def test_greedy_longest_match_fails():
    assert word_break("aab", ["aa", "a", "ab"]) is True
    assert word_break("cars", ["car", "ca", "rs"]) is True
    assert word_break("abcd", ["a", "abc", "b", "cd"]) is True


def test_random_cases_match_brute_force():
    rng = random.Random(29)
    for _ in range(300):
        n = rng.randrange(0, 15)
        s = "".join(rng.choice("ab") for _ in range(n))
        words = ["".join(rng.choice("ab") for _ in range(rng.randrange(1, 4))) for _ in range(rng.randrange(1, 6))]
        want = brute_force(s, words)
        assert word_break(s, words) is want, f"expected {want} for {s!r} with {words}"


def test_worst_case_for_exponential_recursion():
    words = ["a" * k for k in range(1, 11)]
    assert word_break("a" * 300 + "b", words) is False
    assert word_break("a" * 300, words) is True
    assert word_break("a" * 150 + "b" + "a" * 149, words + ["ba"]) is True


def test_long_random_sentence_is_fast():
    rng = random.Random(4)
    words = list({"".join(rng.choice("abcdefgh") for _ in range(rng.randrange(1, 8))) for _ in range(300)})
    s = ""
    while len(s) < 290:
        s += rng.choice(words)
    assert word_break(s, words) is True
    assert word_break(s + "z", words) is False
