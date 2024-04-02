import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    # Perform LU decomposition with partial pivoting
    P = np.eye(n)
    for k in range(n):
        pivot = np.argmax(abs(A[k:, k])) + k
        if pivot != k:
            A[[pivot, k], :] = A[[k, pivot], :]
            P[[pivot, k], :] = P[[k, pivot], :]
        L[k+1:, k] = A[k+1:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        U[k+1:, k+1:] = A[k+1:, k+1:] - L[k+1:, k][:, np.newaxis] * U[k, k+1:]

    return L, U, P

def solve_lu_decomposition(A, b):
    L, U, P = lu_decomposition(A)
    Pb = P @ b

    # Solve Lz = Pb using forward substitution
    n = A.shape[0]
    z = np.zeros(n)
    for i in range(n):
        z[i] = Pb[i] - np.dot(L[i, :i], z[:i])

    # Solve Ux = z using backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (z[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

# Example usage
A = np.array([[2, 4, -1, 3],
              [4, -3, 0, -5],
              [3, -2, 5, -4],
              [8, 1, -2, 6]])
b = np.array([6, -3, 2, 5])

x = solve_lu_decomposition(A, b)
print("Solution:")
print(f"x1 = {x[0]:.3f}")
print(f"x2 = {x[1]:.3f}")
print(f"x3 = {x[2]:.3f}")
print(f"x4 = {x[3]:.3f}")
