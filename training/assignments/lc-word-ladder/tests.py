import random
from collections import deque

from solution import ladder_length


def differ_by_one(a, b):
    return sum(x != y for x, y in zip(a, b)) == 1


def brute_force(begin, end, words):
    """Independent answer: build the adjacency explicitly (O(N^2 L)) and BFS."""
    nodes = list(dict.fromkeys([begin] + words))
    if end not in words:
        return 0
    adj = {w: [v for v in nodes if differ_by_one(w, v)] for w in nodes}
    dist = {begin: 1}
    queue = deque([begin])
    while queue:
        w = queue.popleft()
        if w == end:
            return dist[w]
        for v in adj[w]:
            if v not in dist:
                dist[v] = dist[w] + 1
                queue.append(v)
    return 0


def substitution_bfs(begin, end, words, alphabet):
    """Independent answer for large cases: generate neighbours by substituting letters."""
    words = set(words)
    if end not in words:
        return 0
    dist = {begin: 1}
    queue = deque([begin])
    while queue:
        w = queue.popleft()
        if w == end:
            return dist[w]
        for i in range(len(w)):
            for ch in alphabet:
                v = w[:i] + ch + w[i + 1:]
                if v in words and v not in dist:
                    dist[v] = dist[w] + 1
                    queue.append(v)
    return 0


def test_example():
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5


def test_end_word_missing_or_unreachable():
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0
    assert ladder_length("hit", "cog", ["hot", "cog"]) == 0


def test_one_step_and_short_words():
    assert ladder_length("hit", "hot", ["hot"]) == 2
    assert ladder_length("a", "c", ["a", "b", "c"]) == 2
    assert ladder_length("ab", "cd", ["ad", "cd"]) == 3


def test_duplicates_and_begin_word_in_list():
    assert ladder_length("hot", "dog", ["hot", "hot", "dot", "dot", "dog"]) == 3


def test_shortest_among_several_routes():
    words = ["hot", "dot", "dog", "cog", "hog", "lot", "log"]
    assert ladder_length("hit", "cog", words) == 4           # hit -> hot -> hog -> cog


def test_random_dictionaries_match_brute_force():
    rng = random.Random(31)
    for _ in range(200):
        L = rng.randrange(1, 5)
        alphabet = "abc"
        n = rng.randrange(1, 40)
        words = ["".join(rng.choice(alphabet) for _ in range(L)) for _ in range(n)]
        begin = "".join(rng.choice(alphabet) for _ in range(L))
        end = rng.choice(words)
        if begin == end:
            continue
        want = brute_force(begin, end, words)
        got = ladder_length(begin, end, list(words))
        assert got == want, f"expected {want}, got {got} for {begin}->{end} with {words}"


def test_large_dictionary_is_fast():
    rng = random.Random(77)
    alphabet, L = "abcd", 7
    words = list({"".join(rng.choice(alphabet) for _ in range(L)) for _ in range(8000)})
    begin, end = "aaaaaaa", "dddbbbb"
    if end not in words:
        words.append(end)
    words = [w for w in words if w != begin]
    rng.shuffle(words)
    want = substitution_bfs(begin, end, words, alphabet)
    assert want > 0
    assert ladder_length(begin, end, words) == want
