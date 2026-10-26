def count_n_queens(n: int) -> int:
    """Bitmask backtracking: bit c of each mask means column c is attacked in the current row."""
    full = (1 << n) - 1

    def go(cols, diag, anti):
        if cols == full:
            return 1
        total = 0
        free = full & ~(cols | diag | anti)
        while free:
            bit = free & -free                   # lowest free column
            free ^= bit
            total += go(cols | bit, ((diag | bit) << 1) & full, (anti | bit) >> 1)
        return total

    return go(0, 0, 0)


def solve_n_queens(n: int) -> list[str]:
    """Set-based backtracking that stops at the first full placement."""
    cols, diag, anti = set(), set(), set()       # used columns, r - c diagonals, r + c anti-diagonals
    placement = []                               # placement[r] = column of the queen in row r

    def place(r):
        if r == n:
            return True
        for c in range(n):
            if c in cols or (r - c) in diag or (r + c) in anti:
                continue
            cols.add(c), diag.add(r - c), anti.add(r + c)
            placement.append(c)
            if place(r + 1):
                return True
            placement.pop()
            cols.remove(c), diag.remove(r - c), anti.remove(r + c)
        return False

    if not place(0):
        return []
    return ["." * c + "Q" + "." * (n - c - 1) for c in placement]
