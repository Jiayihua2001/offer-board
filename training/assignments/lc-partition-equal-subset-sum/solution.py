def can_partition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    reachable = [False] * (target + 1)           # reachable[t]: some subset of the numbers seen so far sums to t
    reachable[0] = True
    for x in nums:
        for t in range(target, x - 1, -1):       # downward, so each x is used at most once
            if reachable[t - x]:
                reachable[t] = True
        if reachable[target]:
            return True
    return False


def can_partition_bitset(nums: list[int]) -> bool:
    """Same DP with a Python int as the boolean array: bit t set means sum t is reachable."""
    total = sum(nums)
    if total % 2:
        return False
    bits = 1
    for x in nums:
        bits |= bits << x
    return bool(bits >> (total // 2) & 1)
