from bisect import bisect_left


def length_of_lis(nums: list[int]) -> int:
    tails = []                                  # tails[k] = smallest value ending an increasing run of length k + 1
    for x in nums:
        i = bisect_left(tails, x)               # first tail >= x; bisect_right would allow equal values
        if i == len(tails):
            tails.append(x)                     # x extends the longest run found so far
        else:
            tails[i] = x                        # a run of length i + 1 can now end lower
    return len(tails)
