from collections import deque


def max_sliding_window(nums: list[int], k: int) -> list[int]:
    out: list[int] = []
    window: deque[int] = deque()           # indices in the window, values strictly decreasing
    for i, x in enumerate(nums):
        while window and nums[window[-1]] <= x:
            window.pop()                   # older and not larger: never a maximum again
        window.append(i)
        if window[0] <= i - k:
            window.popleft()               # the front has slid out of the window
        if i >= k - 1:
            out.append(nums[window[0]])
    return out
