import random

from solution import network_delay_time


def bellman_ford(times, n, k):
    """Independent answer: relax every edge n - 1 times."""
    INF = float("inf")
    dist = [INF] * (n + 1)
    dist[k] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in times:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    worst = max(dist[1:])
    return -1 if worst == INF else worst


def test_examples():
    assert network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    assert network_delay_time([[1, 2, 1]], 2, 1) == 1
    assert network_delay_time([[1, 2, 1]], 2, 2) == -1
    assert network_delay_time([], 1, 1) == 0


def test_parallel_edges_take_the_cheapest():
    assert network_delay_time([[1, 2, 10], [1, 2, 3], [2, 3, 1]], 3, 1) == 4


def test_more_hops_can_be_faster():
    assert network_delay_time([[1, 2, 10], [1, 3, 1], [3, 4, 1], [4, 2, 1]], 4, 1) == 3


def test_zero_weights_and_cycles():
    assert network_delay_time([[1, 2, 0], [2, 3, 0], [3, 1, 5]], 3, 1) == 0
    assert network_delay_time([[1, 2, 2], [2, 1, 2], [2, 3, 3], [3, 2, 1]], 3, 3) == 3


def test_unreachable_node_far_from_source():
    times = [[1, 2, 1], [2, 3, 1], [3, 4, 1]]
    assert network_delay_time(times, 4, 1) == 3
    assert network_delay_time(times + [[5, 4, 1]], 5, 1) == -1      # 5 reaches 4, but nothing reaches 5


def test_random_graphs_match_bellman_ford():
    rng = random.Random(17)
    for _ in range(200):
        n = rng.randrange(1, 15)
        m = rng.randrange(0, 3 * n + 1)
        times = [[rng.randrange(1, n + 1), rng.randrange(1, n + 1), rng.randrange(0, 20)] for _ in range(m)]
        k = rng.randrange(1, n + 1)
        want = bellman_ford(times, n, k)
        assert network_delay_time(times, n, k) == want, f"expected {want} for n={n}, k={k}, times={times}"


def test_large_sparse_graph_is_fast():
    rng = random.Random(23)
    n, k = 20_000, 1
    dist = [0] * (n + 1)
    times = []
    for v in range(2, n + 1):                    # a random tree from k fixes the shortest distances
        u = rng.randrange(1, v)
        w = rng.randrange(1, 11)
        dist[v] = dist[u] + w
        times.append([u, v, w])
    while len(times) < 100_000:                   # extra edges that never shorten a path
        u, v = rng.randrange(1, n + 1), rng.randrange(1, n + 1)
        if u != v:
            times.append([u, v, max(0, dist[v] - dist[u]) + rng.randrange(0, 5)])
    rng.shuffle(times)
    assert network_delay_time(times, n, k) == max(dist)
    times.append([n + 1, 1, 1])                  # one more node that nothing reaches
    assert network_delay_time(times, n + 1, k) == -1
