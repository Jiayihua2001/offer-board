from collections import deque


def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    adj = [[] for _ in range(num_courses)]
    indeg = [0] * num_courses
    for a, b in prerequisites:
        adj[b].append(a)                      # b must come before a
        indeg[a] += 1
    queue = deque(c for c in range(num_courses) if indeg[c] == 0)
    order = []
    while queue:
        course = queue.popleft()
        order.append(course)
        for nxt in adj[course]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return order if len(order) == num_courses else []      # anything left over is on or behind a cycle
