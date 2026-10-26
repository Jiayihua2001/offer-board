def subsets(nums: list[int]) -> list[list[int]]:
    out, path = [], []

    def go(start):
        out.append(path[:])                  # every node of the recursion tree is one subset
        for i in range(start, len(nums)):
            path.append(nums[i])
            go(i + 1)                        # only later elements, so each subset is built once
            path.pop()

    go(0)
    return out


def permutations(nums: list[int]) -> list[list[int]]:
    out, path, used = [], [], [False] * len(nums)

    def go():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if not used[i]:
                used[i] = True
                path.append(x)
                go()
                path.pop()                   # undo, so the next sibling starts from the same state
                used[i] = False

    go()
    return out
