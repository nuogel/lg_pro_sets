def power(x,n):
    def power(x, n):
        if n == 0:
            return 1
        if n % 2 == 1:
            ad = x
        else:
            ad = 1
        powr = power(x, n // 2)
        out = powr * powr * ad
        return out

    if n < 0:
        return 1 / power(x, -n)
    else:
        return power(x, n)

a =power(2, -2)
a=0