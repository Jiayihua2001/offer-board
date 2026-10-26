import random


def find_kth_largest(nums: list[int], k: int) -> int:
    """Quickselect with a random pivot and a three-way partition: average O(n)."""
    want = len(nums) - k                       # index in ascending sorted order
    lo, hi = 0, len(nums) - 1
    while True:
        pivot = nums[random.randint(lo, hi)]
        # Dutch-flag partition of nums[lo..hi]: [lo, lt) < pivot, [lt, gt] == pivot, (gt, hi] > pivot.
        lt, i, gt = lo, lo, hi
        while i <= gt:
            x = nums[i]
            if x < pivot:
                nums[lt], nums[i] = x, nums[lt]
                lt += 1
                i += 1
            elif x > pivot:
                nums[gt], nums[i] = x, nums[gt]
                gt -= 1
            else:
                i += 1
        if want < lt:
            hi = lt - 1
        elif want > gt:
            lo = gt + 1
        else:
            return pivot
