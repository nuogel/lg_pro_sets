def coinChange(coins, amount: int) -> int:
    dp = [0]
    for amounti in range(1, amount + 1):
        mincoins = 10000
        for coini in coins:
            if amounti - coini >= 0 and dp[amounti - coini] != -1:
                mincoins = min(mincoins, dp[amounti - coini] + 1)
        if mincoins != 10000:
            dp.append(mincoins)
        else:
            dp.append(-1)

    return dp[-1]


coinChange([2, 5, 10, 1], 27)
