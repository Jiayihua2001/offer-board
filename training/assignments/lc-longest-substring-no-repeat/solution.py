def length_of_longest_substring(s: str) -> int:
    last_seen: dict[str, int] = {}
    best = left = 0
    for right, c in enumerate(s):
        if last_seen.get(c, -1) >= left:
            left = last_seen[c] + 1
        last_seen[c] = right
        best = max(best, right - left + 1)
    return best
