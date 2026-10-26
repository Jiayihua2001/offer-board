def max_profit(prices: list[int]) -> int:
    if not prices:
        return 0
    hold = -prices[0]                 # best profit so far while holding a share
    sold = float("-inf")              # best profit so far having sold today (cooldown tomorrow)
    rest = 0                          # best profit so far holding nothing and free to buy
    for p in prices[1:]:
        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)
    return max(sold, rest)            # ending while holding a share is never optimal
