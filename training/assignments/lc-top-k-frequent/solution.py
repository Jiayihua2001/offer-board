from collections import Counter


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Bucket by frequency: O(n) time and space, no comparisons."""
    counts = Counter(nums)
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
    for value, freq in counts.items():
        buckets[freq].append(value)
    out: list[int] = []
    for freq in range(len(nums), 0, -1):
        for value in buckets[freq]:
            out.append(value)
            if len(out) == k:
                return out
    return out
