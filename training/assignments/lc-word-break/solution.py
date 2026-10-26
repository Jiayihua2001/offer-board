def word_break(s: str, word_dict: list[str]) -> bool:
    words = set(word_dict)
    max_len = max((len(w) for w in words), default=0)
    n = len(s)
    dp = [False] * (n + 1)                        # dp[i]: can s[:i] be segmented
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):   # the last word is s[j:i]; longer words cannot match
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]
