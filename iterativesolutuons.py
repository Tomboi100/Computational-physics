from iterSimEqn_resources import fillVectorOfKnowns, getValueForA

def fillMatrixA(numberOfNodesPerEdge):
    n = numberOfNodesPerEdge ** 2
    A = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            A[i][j] = getValueForA(numberOfNodesPerEdge, i, j)

    return A

def printMatrixAAndVectorB(numberOfNodesPerEdge):
    b = fillVectorOfKnowns(numberOfNodesPerEdge)
    A = fillMatrixA(numberOfNodesPerEdge)

    print("Matrix A:")
    for row in A:
        print(row)

    print("\nVector b:")
    print(b)

printMatrixAAndVectorB(5)  # Example with numberOfNodesPerEdge = 5
