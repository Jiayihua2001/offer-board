import random

from solution import find_words


def exists(board, word):
    """Word Search I: DFS for a single word from every cell."""
    m, n = len(board), len(board[0])

    def dfs(r, c, k):
        if board[r][c] != word[k]:
            return False
        if k == len(word) - 1:
            return True
        board[r][c] = "#"
        found = any(0 <= nr < m and 0 <= nc < n and dfs(nr, nc, k + 1)
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))
        board[r][c] = word[k]
        return found

    return any(dfs(r, c, 0) for r in range(m) for c in range(n))


def brute_force(board, words):
    return {w for w in set(words) if exists([row[:] for row in board], w)}


def check(board, words):
    got = find_words([row[:] for row in board], list(words))
    assert isinstance(got, list), f"expected a list, got {type(got).__name__}"
    assert len(set(got)) == len(got), f"a word was returned more than once: {got}"
    want = brute_force(board, words)
    assert set(got) == want, f"expected {sorted(want)}, got {sorted(got)}"
    return set(got)


def random_walk_word(rng, board, length):
    m, n = len(board), len(board[0])
    r, c = rng.randrange(m), rng.randrange(n)
    used, word = {(r, c)}, board[r][c]
    while len(word) < length:
        options = [(nr, nc) for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                   if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in used]
        if not options:
            break
        r, c = rng.choice(options)
        used.add((r, c))
        word += board[r][c]
    return word


EXAMPLE = [["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]]


def test_example():
    assert check(EXAMPLE, ["oath", "pea", "eat", "rain"]) == {"eat", "oath"}


def test_cells_cannot_be_reused():
    assert check([["a", "a"]], ["aaa", "aa", "a"]) == {"aa", "a"}
    assert check([["a", "b"], ["c", "d"]], ["abcd", "abdc", "ab", "dcab"]) == {"abdc", "ab", "dcab"}


def test_duplicates_and_single_cell_board():
    assert check([["a"]], ["a", "a", "b", "ab"]) == {"a"}
    assert check(EXAMPLE, ["oath", "oath", "eat", "eat"]) == {"eat", "oath"}
    assert check([["x"]], ["xx"]) == set()


def test_word_present_in_many_places_is_reported_once():
    board = [["a"] * 3 for _ in range(3)]
    assert check(board, ["aa", "aaaa", "b"]) == {"aa", "aaaa"}


def test_words_share_prefixes_and_contain_each_other():
    words = ["oa", "oat", "oath", "oaths", "e", "ea", "eat", "eate", "eates"]
    assert check(EXAMPLE, words) == {"oa", "oat", "oath", "e", "ea", "eat", "eate"}


def test_random_boards_match_brute_force():
    rng = random.Random(83)
    for _ in range(60):
        m, n = rng.randrange(1, 6), rng.randrange(1, 6)
        letters = rng.choice(["ab", "abc"])
        board = [[rng.choice(letters) for _ in range(n)] for _ in range(m)]
        words = ["".join(rng.choice(letters) for _ in range(rng.randrange(1, 6))) for _ in range(30)]
        words += [random_walk_word(rng, board, rng.randrange(1, 7)) for _ in range(5)]
        check(board, words)


def test_all_a_board_needs_pruning():
    board = [["a"] * 15 for _ in range(15)]
    words = ["a" * k for k in range(1, 13)] + ["b", "ab", "ba", "aab"]
    got = find_words([row[:] for row in board], words)
    assert sorted(got) == sorted("a" * k for k in range(1, 13)), f"got {sorted(got)}"


def test_large_board_and_word_list_are_fast():
    rng = random.Random(84)
    letters = "abcde"
    board = [[rng.choice(letters) for _ in range(12)] for _ in range(12)]
    words = [random_walk_word(rng, board, rng.randrange(3, 8)) for _ in range(800)]
    words += ["".join(rng.choice(letters) for _ in range(rng.randrange(3, 8))) for _ in range(800)]
    got = check(board, words)
    assert len(got) >= 700
