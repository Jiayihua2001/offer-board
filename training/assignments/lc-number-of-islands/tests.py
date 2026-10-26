import random

from solution import num_islands


def union_find_count(grid):
    """Independent answer: union-find over the land cells."""
    m = len(grid)
    n = len(grid[0]) if m else 0
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    count = 0
    for r in range(m):
        for c in range(n):
            if grid[r][c] != "1":
                continue
            parent[(r, c)] = (r, c)
            count += 1
            for nb in ((r - 1, c), (r, c - 1)):
                if nb in parent:
                    a, b = find(nb), find((r, c))
                    if a != b:
                        parent[b] = a
                        count -= 1
    return count


def run(grid):
    return num_islands([row[:] for row in grid])


def test_example():
    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"],
    ]
    assert run(grid) == 3


def test_one_island_touching_everything():
    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
    ]
    assert run(grid) == 1


def test_diagonals_do_not_connect():
    assert run([["1", "0"], ["0", "1"]]) == 2
    assert run([["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]]) == 5


def test_empty_grids_and_single_cells():
    assert run([]) == 0
    assert run([[]]) == 0
    assert run([["0"]]) == 0
    assert run([["1"]]) == 1
    assert run([["0", "0"], ["0", "0"]]) == 0


def test_thin_grids():
    assert run([["1", "0", "1", "1", "0", "1"]]) == 3
    assert run([["1"], ["1"], ["0"], ["1"]]) == 2


def test_random_grids_match_union_find():
    rng = random.Random(42)
    for _ in range(150):
        m, n = rng.randrange(1, 13), rng.randrange(1, 13)
        p = rng.choice([0.2, 0.4, 0.5, 0.6, 0.8])
        grid = [["1" if rng.random() < p else "0" for _ in range(n)] for _ in range(m)]
        want = union_find_count(grid)
        assert run(grid) == want, f"expected {want} islands for {grid}"


def test_all_land_grid_is_one_island():
    grid = [["1"] * 300 for _ in range(300)]        # 90,000 cells: too deep for recursive DFS
    assert run(grid) == 1


def test_large_random_grid_is_fast():
    rng = random.Random(7)
    grid = [["1" if rng.random() < 0.5 else "0" for _ in range(400)] for _ in range(400)]
    assert run(grid) == union_find_count(grid)
