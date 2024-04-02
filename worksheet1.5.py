import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1

        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

def forward_substitution(L, b):
    n = b.size
    x = np.empty(n)
    for i in range(n):
        sm = 0
        for j in range(i):
            sm += L[i, j] * x[j]
        x[i] = (b[i] - sm) / L[i, i]
    return x

def backward_substitution(U, b):
    n = b.size
    x = np.empty(n)
    for i in reversed(range(n)):
        sm = 0
        for j in range(i + 1, n):
            sm += U[i, j] * x[j]
        x[i] = (b[i] - sm) / U[i, i]
    return x

if __name__ == '__main__':
    A = np.array([[2, 4, -1, 3],
                  [4, -3, 0, -5],
                  [3, -2, 5,-4],
                  [3, -2, 5,-4]], dtype=float)

    b = np.array([6, -3, 2, 5], dtype=float)

    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)

    print(x)

    t = np.matmul(A, x)
    if (t==b).all():
        print("yay")

