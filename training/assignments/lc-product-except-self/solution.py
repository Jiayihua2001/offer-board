def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    out = [1] * n
    for i in range(1, n):                 # out[i] = product of nums[:i]
        out[i] = out[i - 1] * nums[i - 1]
    suffix = 1                             # product of nums[i + 1:]
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]
    return out
