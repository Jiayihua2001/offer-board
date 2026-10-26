import random

from solution import Trie


def test_example():
    t = Trie()
    t.insert("apple")
    assert t.search("apple")
    assert not t.search("app")
    assert t.starts_with("app")
    t.insert("app")
    assert t.search("app")


def test_empty_trie():
    t = Trie()
    assert not t.search("a")
    assert not t.starts_with("a")


def test_prefixes_are_not_words_until_inserted():
    t = Trie()
    for w in ["car", "card", "care", "cart"]:
        t.insert(w)
    assert t.search("car") and t.search("card") and t.search("cart")
    assert not t.search("ca") and t.starts_with("ca")
    assert not t.search("cards") and not t.starts_with("cards")
    assert t.starts_with("care") and not t.starts_with("cat")


def test_repeated_insert_and_longer_queries():
    t = Trie()
    t.insert("hello")
    t.insert("hello")
    assert t.search("hello")
    assert not t.search("helloo") and not t.starts_with("helloo")
    assert t.starts_with("hello") and t.starts_with("h")


def test_instances_do_not_share_state():
    a, b = Trie(), Trie()
    a.insert("alpha")
    assert not b.search("alpha") and not b.starts_with("al")
    b.insert("beta")
    assert not a.search("beta") and b.search("beta")
    assert a.search("alpha")


def test_random_operations_match_a_set():
    rng = random.Random(53)
    for _ in range(60):
        t, words, prefixes = Trie(), set(), set()
        for _ in range(rng.randrange(1, 30)):
            w = "".join(rng.choice("ab") for _ in range(rng.randrange(1, 6)))
            t.insert(w)
            words.add(w)
            prefixes.update(w[:i] for i in range(1, len(w) + 1))
            q = "".join(rng.choice("ab") for _ in range(rng.randrange(1, 6)))
            assert bool(t.search(q)) is (q in words), f"search({q!r}) wrong after inserting {sorted(words)}"
            assert bool(t.starts_with(q)) is (q in prefixes), f"starts_with({q!r}) wrong after inserting {sorted(words)}"


def test_many_words_are_fast():
    rng = random.Random(54)
    letters = "abcdefghij"
    words = ["".join(rng.choice(letters) for _ in range(rng.randrange(3, 12))) for _ in range(20_000)]
    t, word_set, prefixes = Trie(), set(), set()
    for w in words:
        t.insert(w)
        word_set.add(w)
        prefixes.update(w[:i] for i in range(1, len(w) + 1))
    for w in words[:10_000]:
        assert t.search(w) and t.starts_with(w[:-1])
    for _ in range(20_000):
        q = "".join(rng.choice(letters) for _ in range(rng.randrange(1, 12)))
        assert bool(t.search(q)) is (q in word_set)
        assert bool(t.starts_with(q)) is (q in prefixes)
