def min_distance(word1: str, word2: str) -> int:
    if len(word1) < len(word2):
        word1, word2 = word2, word1               # roll over the shorter string: O(min(m, n)) space
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))                     # distance from "" to word2[:j] is j inserts
    for i in range(1, m + 1):
        cur = [i] + [0] * n                       # distance from word1[:i] to "" is i deletes
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j],         # delete word1[i-1]
                                 cur[j - 1],      # insert word2[j-1]
                                 prev[j - 1])     # replace
        prev = cur
    return prev[n]
