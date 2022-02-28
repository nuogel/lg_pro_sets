def coinChange(coins, amount: int) -> int:
    coins.insert(0,0)
    matrix =[[0 for i in range(amount+1)] for i in range(len(coins))]

    for i in range(len(coins)):
        for j in range(amount+1):
            if coins[j]>j:
                matrix[i][j]=matrix[i-1][j]
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j-coins[j]]+1)
    return matrix[-1][-1]

coinChange([1,2,5], 11)