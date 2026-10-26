def subarray_sum(nums: list[int], k: int) -> int:
    count = {0: 1}          # the empty prefix
    total = answer = 0
    for x in nums:
        total += x
        answer += count.get(total - k, 0)
        count[total] = count.get(total, 0) + 1
    return answer
