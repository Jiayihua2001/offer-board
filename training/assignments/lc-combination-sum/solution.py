def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    candidates = sorted(candidates)
    out, path = [], []

    def go(start, remaining):
        if remaining == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            c = candidates[i]
            if c > remaining:
                break                        # sorted, so nothing after c fits either
            path.append(c)
            go(i, remaining - c)             # i (not i + 1): the same candidate may be reused
            path.pop()

    go(0, target)
    return out
