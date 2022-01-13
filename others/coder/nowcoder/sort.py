import numpy as np
import math


def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(1, len(arr) - i):
            if arr[j - 1] > arr[j]:
                max = arr[j - 1]
                arr[j - 1] = arr[j]
                arr[j] = max
    return arr


# 归并排序
def merge_sort(alist):
    if len(alist) <= 1:
        return alist
    # 二分分解
    num = len(alist) // 2
    left = merge_sort(alist[:num])
    right = merge_sort(alist[num:])

    # 合并
    def merge(left, right):
        '''合并操作，将两个有序数组left[]和right[]合并成一个大的有序数组'''
        # left与right的下标指针
        l, r = 0, 0
        result = []
        while l < len(left) and r < len(right):
            if left[l] < right[r]:
                result.append(left[l])
                l += 1
            else:
                result.append(right[r])
                r += 1
        result += left[l:]
        result += right[r:]
        return result

    return merge(left, right)


# 希尔排序
def shellsort(arr):
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            tmpi = arr[i]
            while i >= gap and arr[i - gap] > tmpi:
                arr[i], arr[i - gap] = arr[i - gap], arr[i]
                i -= gap
        gap = gap // 2
    return arr


# 计数排序
def countingSort(arr):
    _max = max(arr)
    _min = min(arr)

    n = len(arr)
    c = [0] * (_max - _min + 1)
    output = [0] * n
    for arri in arr:
        c[arri - _min] += 1

    for i in range(1, (_max - _min + 1)):
        c[i] = c[i - 1] + c[i]

    for arri in arr:
        output[c[arri - _min] - 1] = arri
        c[arri - _min] -= 1

    return output


# 计数排序 好理解
def count_sort(arr):  # arr为待排序列表，max_count为最大元素
    _max = max(arr)
    _min = min(arr)
    count = [0 for _ in range(_max - _min + 1)]  # 列表推导式生成0到500的列表，用来记录元素出现多少次
    for val in arr:
        count[val - _min] += 1  # 如果元素出现则对应count列表索引处+1
    arr.clear()  # 直接清除原列表，不在生成新列表，节省内容空间
    for index, val in enumerate(count):  # 获取index下标，val对应的值
        # for循环里， index索引出现了val次
        for i in range(val):
            arr.append(index + _min)  # 把index添加到arr，次数为val
    return arr


if __name__ == '__main__':
    arr = list(np.random.randint(0, 20, 10))
    print(arr)
    y = bubble_sort(arr)
    print(y)
    y = merge_sort(arr)
    print(y)
    y = shellsort(arr)
    print(y)
    y = countingSort(arr)
    print(y)
    y = count_sort(arr)
    print(y)
