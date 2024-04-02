import numpy as np
import jupyter as jp
from iterSimEqn_resources import fillVectorOfKnowns, getValueForA, viewSolution
import matplotlib.pyplot as plt

def l1_norm(vector):
    return np.sum(np.abs(vector))

def l2_norm(vector):
    return np.sqrt(np.sum(vector**2))

def linf_norm(vector):
    return np.max(np.abs(vector))

def richardsonMethod(A, b, x0, iterations):
    n = len(b)
    x = x0.copy()

    for _ in range(iterations):
        x = (np.identity(n) - A) @ x + b

    norms_l1 = []
    norms_l2 = []
    norms_linf = []

    for k in range(iterations):
        r = b - np.dot(A, x)
        x = x + omega * r
        residuals = np.dot(A, x) - b
        norm_l1 = l1_norm(residuals)
        norm_l2 = l2_norm(residuals)
        norm_linf = linf_norm(residuals)
        print(f"Iteration {k + 1}: L1 Norm: {norm_l1}, L2 Norm: {norm_l2}, L∞ Norm: {norm_linf}")
        norms_l1.append(norm_l1)
        norms_l2.append(norm_l2)
        norms_linf.append(norm_linf)
    return x, norms_l1, norms_l2, norms_linf

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

    x, norms_l1, norms_l2, norms_linf = richardsonMethod(A, b, x0, m)
    viewSolution(x, numberOfNodesPerEdge)
    plotNorms(norms_l1)
    plotNorms(norms_l2)
    plotNorms(norms_linf)

def plotNorms(norms):
    iterations = range(1, len(norms)+1)
    plt.plot(iterations, norms)
    plt.xlabel('Iteration')
    plt.ylabel('Norm Value')
    plt.title('Norm Values vs. Number of Iterations')
    plt.show()

# Parameters
numberOfNodesPerEdge = 5
m = 2000

plotNorms(norms_l1)
plotNorms(norms_l2)
plotNorms(norms_linf)