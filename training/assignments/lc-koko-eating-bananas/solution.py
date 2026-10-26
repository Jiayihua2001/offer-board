def min_eating_speed(piles: list[int], h: int) -> int:
    def hours(k: int) -> int:
        return sum((p + k - 1) // k for p in piles)

    lo, hi = 1, max(piles)                    # hi is always feasible (every pile in one hour)
    while lo < hi:
        mid = (lo + hi) // 2
        if hours(mid) <= h:
            hi = mid                          # mid works; the answer is mid or smaller
        else:
            lo = mid + 1                      # mid is too slow
    return lo
