import random

from solution import length_of_longest_substring


def brute_force(s):
    best = 0
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            best = max(best, j - i + 1)
    return best


def test_examples():
    assert length_of_longest_substring("abcabcbb") == 3
    assert length_of_longest_substring("bbbbb") == 1
    assert length_of_longest_substring("pwwkew") == 3


def test_empty_and_single_character():
    assert length_of_longest_substring("") == 0
    assert length_of_longest_substring("a") == 1
    assert length_of_longest_substring(" ") == 1, "a space is a character"


def test_left_pointer_never_moves_backwards():
    assert length_of_longest_substring("abba") == 2, "after the second b, the first a must not pull the window back"
    assert length_of_longest_substring("tmmzuxt") == 5
    assert length_of_longest_substring("dvdf") == 3


def test_all_distinct_is_whole_string():
    s = "abcdefghijklmnopqrstuvwxyz0123456789"
    assert length_of_longest_substring(s) == len(s)


def test_case_sensitive_and_symbols():
    assert length_of_longest_substring("aA") == 2
    assert length_of_longest_substring("a!b!c") == 3
    assert length_of_longest_substring("  ") == 1


def test_random_against_brute_force():
    rng = random.Random(5)
    for _ in range(300):
        n = rng.randrange(0, 25)
        s = "".join(rng.choice("abcd") for _ in range(n))
        assert length_of_longest_substring(s) == brute_force(s), f"wrong answer for {s!r}"


def test_large_ascii_input_is_fast():
    rng = random.Random(17)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    s = "".join(rng.choice(alphabet) for _ in range(300_000))
    s = s[:150_000] + alphabet + "0123456789" + s[150_000:]
    assert length_of_longest_substring(s) >= 36


def test_large_alphabet_makes_the_answer_long():
    # 150k distinct Unicode characters next to a random ASCII tail: the answer is at least 150_000,
    # so "extend from every start until a repeat" does quadratic work here.
    rng = random.Random(18)
    distinct = "".join(chr(0x10000 + i) for i in range(150_000))
    tail = "".join(rng.choice("abcdefghij") for _ in range(150_000))

    def distinct_run(chars):
        seen, n = set(), 0
        for c in chars:
            if c in seen:
                break
            seen.add(c)
            n += 1
        return n

    assert length_of_longest_substring(distinct + tail) == 150_000 + distinct_run(tail)
    assert length_of_longest_substring(tail + distinct) == 150_000 + distinct_run(reversed(tail))
