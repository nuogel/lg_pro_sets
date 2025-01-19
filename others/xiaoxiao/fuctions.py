import sympy as sy

def solveFun1():
    a,b = sy.symbols("a b")
    out = sy.solve(a-5-b-5, a-b)
    print(out)




if __name__ == "__main__":
    solveFun1()