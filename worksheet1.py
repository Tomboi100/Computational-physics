import numpy as np

def diag(A, b):
    x = np.empty(b.size)

    for i in range(b.size):
        x[i] = b[i] / A[i, i]

    return x

def forwardSubstitution(A, b):
    n = b.size
    x = np.empty(n)
    for i in range(n):
        sm = 0
        for j in range(i):
            sm += A[i, j]*x[j]
        x[i] = (1 / A[i, i]) * ((b[i])-sm)
        # [i] = (b[i]-sm) / A[i, i] both lines are the same as above
    return x

def backSubtitution(A,b):
    n = b.size
    x = np.empty(n)
    for i in reversed(range(n)):
        sm = 0
        for j in range(i+1, n):
            sm += A[i, j] * x[j]
        x[i] = (1 / A[i, i]) * ((b[i]) - sm)
        #[i] = (b[i]-sm) / A[i, i] both lines are the same as above
    return x

def forwardSubstitutionForPermuatation(A, b, p):
    n = b.size
    x = np.empty(n)
    for i in range(n):
        sm = 0
        for j in range(i):
            sm += A[p[i], j]*x[j]
        x[i] = (1 / A[p[i], i]) * ((b[p[i]])-sm)
        # [i] = (b[i]-sm) / A[i, i] both lines are the same as above
    return x

def swap(a, b):
    temp = a
    a = b
    b = temp
    return a,b

def LUdecompWithPivoting(A,b,p):
    n = b.size
    for i in range(n):
        p[i] = i
        s = abs(A).max(axis=1)
    for j in range(n-1):
        for i in range(j+1, n+1):
            if abs((A[p[i],j])/(s[p[i]])) > abs((A[p[j],j])/(s[p[j]]))

                p[i], p[j] = swap(p[i],p[j])
        for i in range(j + 1, n + 1):
            (A[p[j], j]) = (A[p[j],j])/(A[p[j],j])
            for k in range(j + 1, n + 1):
                A[p[i], k] = A[p[i], k] - A[p[i], k]*A[p[j], k]
    return A,p


if __name__ == '__main__':

    # A = np.array([[4, 0, 0, 0],
    #               [0,-4, 0, 0],
    #               [0, 0, 3, 0],
    #               [0, 0, 0, 6]], dtype=float)

    # A = np.array([[2, 1, 3, 8],
    #               [0, -3, -6, 4],
    #               [0, 0, 5, -2],
    #               [0, 0, 0, 1]], dtype=float)

    # A = np.array([[2, 0, 0, 0],
    #               [1, -3, 0, 0],
    #               [3, -6, 5, 0],
    #               [8, 4, -2, 1]], dtype=float)

    # A = np.array([[1, -3, 0, 0],
    #               [8, -4, -2, 1],
    #               [2, -0, 0, 0],
    #               [3, -6, 5, 0]], dtype=float)

    A = np.array([[2, -4, -1, 3],
                  [4, -3, -0, -5],
                  [3, -2, 5, -4],
                  [8, -1, -2, 6]], dtype=float)

    #b = np.array([2,4,-3,1], dtype=float)
    b = np.array([4, 1, 2, -3], dtype=float)

    #p = [2, 0, 3, 1]
    p = [4, 1, 3, 2]

    #x = diag(A,b)

    #x = forwardSubstitution(A,b)

    #x = backSubtitution(A,b)

    #x = forwardSubstitutionForPermuatation(A,b,p)

    x = LUdecompWithPivoting(A,b,p)

    print(x)

    t = np.matmul(A, x)
    if (t==b).all():
        print("yay")