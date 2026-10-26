def coin_change(coins: list[int], amount: int) -> int:
    INF = amount + 1                              # more coins than any real answer can use
    dp = [0] + [INF] * amount                     # dp[a] = fewest coins making exactly a
    for c in set(coins):
        for a in range(c, amount + 1):
            if dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] <= amount else -1
