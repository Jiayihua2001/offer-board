def find_median_sorted_arrays(a: list[int], b: list[int]) -> float:
    """Binary search the cut in the shorter array: O(log(min(m, n)))."""
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    half = (m + n + 1) // 2                  # size of the left part (gets the extra element when odd)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2                   # a[:i] goes left
        j = half - i                         # b[:j] goes left
        a_left = a[i - 1] if i > 0 else float("-inf")
        a_right = a[i] if i < m else float("inf")
        b_left = b[j - 1] if j > 0 else float("-inf")
        b_right = b[j] if j < n else float("inf")
        if a_left <= b_right and b_left <= a_right:
            if (m + n) % 2:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        if a_left > b_right:
            hi = i - 1                       # too many from a on the left
        else:
            lo = i + 1                       # too few from a on the left
    raise ValueError("inputs must be sorted")
