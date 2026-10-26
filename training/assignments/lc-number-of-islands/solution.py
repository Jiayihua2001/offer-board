from collections import deque


def num_islands(grid: list[list[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    count = 0
    for r0 in range(m):
        for c0 in range(n):
            if grid[r0][c0] != "1":
                continue
            count += 1
            grid[r0][c0] = "0"                   # sink the island as we go: no visited set needed
            queue = deque([(r0, c0)])
            while queue:
                r, c = queue.popleft()
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == "1":
                        grid[nr][nc] = "0"
                        queue.append((nr, nc))
    return count
