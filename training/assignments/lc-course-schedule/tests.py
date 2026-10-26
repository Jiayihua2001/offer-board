import random

from solution import find_order


def is_valid_order(num_courses, prerequisites, order):
    if sorted(order) != list(range(num_courses)):
        return False
    pos = {course: i for i, course in enumerate(order)}
    return all(pos[b] < pos[a] for a, b in prerequisites)


def has_cycle(num_courses, prerequisites):
    """Independent check: three-colour DFS on an explicit stack."""
    adj = [[] for _ in range(num_courses)]
    for a, b in prerequisites:
        adj[b].append(a)
    colour = [0] * num_courses                 # 0 new, 1 on the current path, 2 finished
    for start in range(num_courses):
        if colour[start]:
            continue
        colour[start] = 1
        stack = [(start, iter(adj[start]))]
        while stack:
            node, it = stack[-1]
            for nxt in it:
                if colour[nxt] == 1:
                    return True
                if colour[nxt] == 0:
                    colour[nxt] = 1
                    stack.append((nxt, iter(adj[nxt])))
                    break
            else:
                colour[node] = 2
                stack.pop()
    return False


def check(num_courses, prerequisites):
    order = find_order(num_courses, [list(p) for p in prerequisites])
    if has_cycle(num_courses, prerequisites):
        assert order == [], f"the prerequisites contain a cycle, expected [] but got {order[:20]}"
    else:
        assert is_valid_order(num_courses, prerequisites, order), \
            f"{order[:20]}{'...' if len(order) > 20 else ''} is not a valid order for {prerequisites[:20]}"
    return order


def test_examples():
    assert find_order(2, [[1, 0]]) == [0, 1]
    check(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
    assert find_order(2, [[1, 0], [0, 1]]) == []
    assert find_order(1, []) == [0]


def test_no_prerequisites_is_any_permutation():
    order = check(5, [])
    assert sorted(order) == [0, 1, 2, 3, 4]


def test_self_loop_and_repeated_pairs():
    assert find_order(3, [[0, 0]]) == []
    assert find_order(3, [[1, 0], [1, 0], [2, 1], [2, 1]]) == [0, 1, 2]


def test_disconnected_components():
    check(6, [[1, 0], [3, 2], [5, 4]])
    assert find_order(6, [[1, 0], [3, 2], [4, 3], [2, 4]]) == []     # cycle 2 -> 4 -> 3 -> 2 far from 0


def test_random_graphs():
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randrange(1, 12)
        m = rng.randrange(0, 2 * n + 1)
        prereqs = [[rng.randrange(n), rng.randrange(n)] for _ in range(m)]
        check(n, prereqs)


def test_long_chain():
    n = 100_000
    order = find_order(n, [[i + 1, i] for i in range(n - 1)])
    assert order == list(range(n))


def test_large_random_dag_is_fast():
    rng = random.Random(9)
    n, m = 50_000, 150_000
    rank = list(range(n))
    rng.shuffle(rank)                     # rank[c] is the position of course c in a hidden valid order
    prereqs = []
    while len(prereqs) < m:
        a, b = rng.randrange(n), rng.randrange(n)
        if rank[b] < rank[a]:
            prereqs.append([a, b])
    order = find_order(n, prereqs)
    assert is_valid_order(n, prereqs, order), "not a valid order for the large DAG"
    a, b = prereqs[0]
    prereqs.append([b, a])                # the reverse of an existing pair closes a cycle
    assert find_order(n, prereqs) == []
