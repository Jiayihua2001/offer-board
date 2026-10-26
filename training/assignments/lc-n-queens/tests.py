from solution import count_n_queens, solve_n_queens

COUNTS = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724}


def assert_valid_board(board, n):
    assert isinstance(board, list) and len(board) == n, f"expected {n} rows, got {board!r}"
    cols = []
    for row in board:
        assert isinstance(row, str) and len(row) == n, f"bad row {row!r} for n={n}"
        assert row.count("Q") == 1 and set(row) <= {"Q", "."}, f"bad row {row!r}"
        cols.append(row.index("Q"))
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            c1, c2 = cols[r1], cols[r2]
            assert c1 != c2, f"queens in rows {r1} and {r2} share column {c1}"
            assert abs(c1 - c2) != r2 - r1, f"queens in rows {r1} and {r2} share a diagonal"


def test_counts_for_small_boards():
    for n in range(1, 9):
        assert count_n_queens(n) == COUNTS[n], f"wrong count for n={n}"


def test_one_queen():
    assert count_n_queens(1) == 1
    assert solve_n_queens(1) == ["Q"]


def test_no_solution_for_two_and_three():
    assert solve_n_queens(2) == []
    assert solve_n_queens(3) == []
    assert count_n_queens(2) == 0 and count_n_queens(3) == 0


def test_four_queens_board():
    board = solve_n_queens(4)
    assert_valid_board(board, 4)
    assert board in ([".Q..", "...Q", "Q...", "..Q."], ["..Q.", "Q...", "...Q", ".Q.."])


def test_boards_are_valid_for_every_n():
    for n in range(4, 11):
        assert_valid_board(solve_n_queens(n), n)


def test_calls_do_not_interfere():
    assert count_n_queens(6) == 4
    assert_valid_board(solve_n_queens(6), 6)
    assert count_n_queens(6) == 4
    assert count_n_queens(5) == 10


def test_larger_boards_are_fast():
    assert count_n_queens(9) == 352
    assert count_n_queens(10) == 724
    assert_valid_board(solve_n_queens(10), 10)
