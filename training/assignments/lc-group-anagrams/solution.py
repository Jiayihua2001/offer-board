def group_anagrams(strs: list[str]) -> list[list[str]]:
    buckets: dict[str, list[str]] = {}
    for s in strs:
        buckets.setdefault("".join(sorted(s)), []).append(s)
    return list(buckets.values())
