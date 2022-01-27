class Solution:
    def maxValue(self, grid) -> int:
        # write code here
        m, n = len(grid), len(grid[0])
        maxgrid = [[0 for i in range(n + 1)] for j in range(m+ 1)]

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                maxgrid[i][j] = max(maxgrid[i + 1][j], maxgrid[i][j + 1]) + grid[i][j]

        return maxgrid[0][0]


if __name__ == '__main__':
    grid = [[9, 1, 4, 8]]
    s = Solution()
    s.maxValue(grid)
