def maxProduct(nums) -> int:
    maxv = [nums[0]]
    minv = [nums[0]]

    for ni in nums[1:]:
       maxvi = max(max(ni, maxv[-1]*ni), max(ni, minv[-1]*ni))
       minvi = min(min(ni, maxv[-1]*ni), min(ni, minv[-1]*ni))
       maxv.append(maxvi)
       minv.append(minvi)

    return max(maxv)

maxProduct([2,3,-2,4])

