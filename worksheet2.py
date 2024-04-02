import math

from iterSimEqn_resources import *
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

numberofNodesPerEdge = 5
m = 99
n = numberofNodesPerEdge**2

b = fillVectorOfKnowns(numberofNodesPerEdge, richardson=True)

x = np.zeros(b.size)

A = np.zeros((n,n))

def firstMethod(A,b):
    for i in range(4):
        for j in range(4):
            A[i, j] = getValueForA(4, i, j, True)
    return(A)

#print(firstMethod(A,b))

def richardsonMethod(A,b,x,m,n):
    r = np.zeros(n)
    for k in range(1,m):
        for i in range(1,n):
            sm = 0
            for j in range(n):
                sm += A[i,j]*x[j]
            r[i] = b[i] - sm
        for i in range(1,n):
            x[i] = x[i] + r[i]
    return(x)

#print(richardsonMethod(A,b,x,m,n))

def norm1(z):
    return np.abs(z).sum()

def norm2(z):
    return np.sqrt((z**2).sum())

def norm3(z):
    return np.abs(z).max()

def richardsonNorm(A,b,x,m,n):
    r = np.zeros(n)
    #xTrue = npr.random(n)
    l1s, l2s, Linfs = [norm1(x - xTrue)], [norm2(x - xTrue)], [norm3(x - xTrue)]
    #l1s, l2s, Linfs = [],[],[]
    for k in range(m):
        for i in range(n):
            sm = 0
            for j in range(n):
                sm += A[i,j]*x[j]
            r[i] = b[i] - sm
        for i in range(n):
            x[i] = x[i] + r[i]
        #l1s, l2s, Linfs = [norm1(x - xTrue)], [norm2(x - xTrue)], [norm3(x - xTrue)]
        l1s.append(norm1(x-xTrue))
        l2s.append(norm2(x-xTrue))
        Linfs.append(norm3(x-xTrue))
    return(l1s, l2s, Linfs, x)

xTrue = getSolution(numberofNodesPerEdge)
l1s,l2s,Linfs, x = richardsonNorm(A,b,x,m,n)

# plt.plot(xTrue, l1s, color='r')
# plt.plot(xTrue, l2s, color='g')
# plt.plot(xTrue, Linfs, color='r')
#plt.show()
# xTrue = npr.random(n)
# l1s,l2s,Linfs = [norm1(x-xTrue)],[norm2(x-xTrue)],[norm3(x-xTrue)]

# Ap = np.array([[45, 19, 82],
#                   [10, 14, 77],
#                   [32, 40],
#                   [22, 12],
#                   [18, 73],
#                   [61, 53],
#                   [38, 27, 76, 25],
#                   [13, 49],
#                   [16, 91, 29]], dtype=float)
#
# Jp = np.array([[0, 1, 6],
#                [1,]],dtype=float)

def firstMethodLISTofLIST(n):
    #n = n^2, n squared
    #n2 # just n. squareroot n^2
    n2 = n**2
    Ap = []
    Jp = []
    for i in range(n2):
        atemp = []
        jtemp = []
        for j in range(n2):
            ValueA = getValueForA(n, i, j, richardson=True)
            if ValueA != 0:
                atemp.append(ValueA)
                jtemp.append(j)
        Ap.append(atemp)
        Jp.append(jtemp)
    return(Ap, Jp)

Ap, Jp = firstMethodLISTofLIST(numberofNodesPerEdge)
print(Ap)
print(Jp)

def practicalIterativeMeth(m,n,b,Ap,Jp,x):
    r = np.zeros(n)
    for k in range(m):
        for i in range(n):
            sm = 0
            atemp = Ap[i]
            jtemp = Jp[i]
            for j in range(len(jtemp)):
                sm += atemp[j]*x[j]
            r[i] = b[i] - sm
        for i in range(n):
            x[i] += r[i]
    return(x)

print(practicalIterativeMeth(m,n,b,Ap,Jp,x))

# def Jacobi(Ap, Jp ,b,x=None,MaxIterations=None, tolerance=1.e-3):
#     check = lambda k :True if MaxIterations is None else k<MaxIterations
#     #j = q
#     if x is None: x=b.copy()
#     for i in range(b.size):
#         q = Jp[i].index(i)
#         gamma = 1./Ap[i][q]
#         b[i] *= gamma
#         A[i] = [gamma*a for p, a in enumerate(Ap[i]) if p!=q]
#         Jp.remove(i)
#     u = x.copy()
#     k = 0
#     while check(k):
#         k += 1
#         for i in range(b.size):
#             sm=0.
#             for aij, q in zip(Ap[i], Jp[i]): sm += aij*x[q]
#             u[i] = b[i] = sm
#         done = l1(x-u)<tolerance
#         x = u.copy()
#         if done: break
#         return x, k

def jacobi(A, b, x0, max_iter=1000, tol=1e-6):
    n = len(b)
    x = np.zeros(n)
    for iteration in range(max_iter):
        for i in range(n):
            s = sum([A[i][j] * x0[j] for j in range(n) if j != i])
            x[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x - x0) < tol:
            break
        x0 = x.copy()
    return x

def gauss_seidel(A, b, x0, max_iter=1000, tol=1e-6):
    n = len(b)
    x = x0.copy()
    for iteration in range(max_iter):
        for i in range(n):
            s1 = sum([A[i][j] * x[j] for j in range(i)])
            s2 = sum([A[i][j] * x0[j] for j in range(i+1, n)])
            x[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x - x0) < tol:
            break
        x0 = x.copy()
    return x
