def fourSum(nums, target: int):
    output = []
    lastpop = 10000000
    nums.sort()

    def sovle(ii, path, ids, lastpop):
        if len(path) == 4 and sum(path) == target:
            pathi = path[:]
            pathi.sort()
            if pathi not in output:
                output.append(pathi)
                return
        elif len(path) > 4:
            return

        for i in range(ii, len(nums)):
            if i in ids or nums[i] == lastpop: continue
            path.append(nums[i])
            ids.append(i)
            sovle(ii, path, ids, lastpop)
            lastpop = path.pop()
            ids.pop()

    sovle(0, [], [], lastpop)
    return output

fourSum([-1,0,1,2,-1,-4],-1)