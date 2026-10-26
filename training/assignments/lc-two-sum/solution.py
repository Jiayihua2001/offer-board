def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for j, x in enumerate(nums):
        i = seen.get(target - x)
        if i is not None:
            return [i, j]
        seen[x] = j
    raise ValueError("no answer")
