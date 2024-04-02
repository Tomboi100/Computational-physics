import numpy as np
from iterSimEqn_resources import fillVectorOfKnowns, getValueForA, viewSolution

def richardsonMethod(A, b, x0, iterations):
    n = len(b)
    x = x0.copy()

    for _ in range(iterations):
        x = (np.identity(n) - A) @ x + b

    return x

def fillMatrixA(numberOfNodesPerEdge):
    n = numberOfNodesPerEdge ** 2
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = getValueForA(numberOfNodesPerEdge, i, j, richardson=True)

    return A

def printMatrixAAndVectorB(numberOfNodesPerEdge):
    b = fillVectorOfKnowns(numberOfNodesPerEdge, richardson=True)
    A = fillMatrixA(numberOfNodesPerEdge)

    print("Matrix A:")
    print(A)

    print("\nVector b:")
    print(b)

    x = richardsonMethod(A, b, x0, m)
    viewSolution(x, numberOfNodesPerEdge)

numberOfNodesPerEdge = 5
m = 2000
x0 = np.zeros(numberOfNodesPerEdge**2)

printMatrixAAndVectorB(numberOfNodesPerEdge)