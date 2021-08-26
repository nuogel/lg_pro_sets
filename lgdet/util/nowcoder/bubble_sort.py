def bubble_sort(list_t):
    for i in range(len(list_t)):
        for j in range(1, len(list_t) - i):
            if list_t[j - 1] > list_t[j]:
                max = list_t[j - 1]
                list_t[j - 1] = list_t[j]
                list_t[j] = max

    print(list_t)


list_t = [2, 3, 4, 6, 3, 6, 3, 9, 12, 5, 32, 1]
bubble_sort(list_t)


def b_search(list_t, left, right, target):
    if left <= right:
        mid = int(left + (right - left) / 2)
        if list_t[mid] == target:
            return mid, target
        elif list_t[mid] < target:
            return b_search(list_t, mid + 1, right, target)
        else:
            return b_search(list_t, left, mid - 1, target)
    else:
        return None


print(b_search(list_t, 0, len(list_t) - 1, 9))
