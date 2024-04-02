import numpy as np
import matplotlib.pyplot as plt

def exptaylor(x, x0, nmax):
    #x: argument
    #x0: arugment at which the derivatives will be caluclated
    #nmax: n at which the series will terminate
    t = 0
    for n in range(nmax+1):
        t = t + np.exp(x0) * (x-x0)**n / np.math.factorial(n)
    return t

# print(exptaylor(1, 0, 10))
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim([-5,100])
#
# x_list = np.linspace(-5,5,101)
# plt.scatter(x_list, np.exp(x_list))
#
# nmax = 10
# plt.plot(x_list, exptaylor(x_list, 0, nmax), 'red')
# plt.show()

def sinTaylor(x, nmax):
    #x: argument
    #nmax: n at which the series will terminate
    t = 0
    for n in range(nmax+1):
        t = t + (-1)**n + x**(2*n+1) / np.math.factorial(2*n+1)
    return t

# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim([-2,2])
#
# x_list = np.linspace(-10,10,101)
# plt.scatter(x_list, np.sin(x_list))
#
# plt.plot(x_list, sinTaylor(x_list, 3), 'blue')
# plt.plot(x_list, sinTaylor(x_list, 6), 'red')
# plt.plot(x_list, sinTaylor(x_list, 9), 'red')
# plt.show()
#
# print(sinTaylor(10.5, 0))

def derivative(f, x, h):
    # f: function
    # x: argument of f
    # h: stepsize
    return(f(x+h) - f(x)) / h

def func(x):
    return 2*np.sin(x)**2 + x

def nDerivative(f, x, h, n):
    # f: function
    # x: argument of f
    # h: stepsize
    # n: nth derivative
    t = 0
    for k in range(n+1):
        t = t + (-1)**(k+n) / (np.math.factorial(k) * np.math.factorial(n-k)) * f(x + k*h)
    return t / h**n

x0 = 10.5
h = 0.1
print(func(x0))
print(derivative(func, x0, h))
print(nDerivative(func, x0, h, 0))
print(nDerivative(func, x0, h, 1))
print(nDerivative(func, x0, h, 2))

def taylor(f, x, x0, nmax, h):
    #x: argument
    #x0: arugment at which the derivatives will be caluclated
    #nmax: n at which the series will terminate
    #h: stepsize
    t = 0
    for n in range(nmax+1):
        t = t + nDerivative(f, x0, h, n) * (x-x0)**n / np.math.factorial(n)
    return t

# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim([-5,8])
#
# x_list = np.linspace(-5,5,101)
# plt.scatter(x_list, func(x_list))
#
# nmax = 25
# h = 0.05
#
# plt.plot(x_list, taylor(func, x_list, 0, nmax, h), 'blue')
# plt.plot(x_list, taylor(func, x_list, 2, nmax, h), 'red')
# plt.plot(x_list, taylor(func, x_list, -3, nmax, h), 'green')
# plt.show()

def correctfunction(x):
    return 15 + 2.4*x - 0.5*x**2 - 0.35*x**3

# npoints = 21
# x_list = np.linspace(-5,5,npoints)
# data0 = np.array([x_list, correctfunction(x_list)])
#
# plt.xlabel('x')
# plt.xlabel('y')
# plt.scatter(data0[0], data0[1])
# plt.show()
#
# 0.1 * (2*np.random.rand(npoints)-1)
# data = np.array([data0[0] + 0.25 * (2*np.random.rand(npoints)-1), data0[1] + 5.0 * (2*np.random.rand(npoints)-1)])
#
# plt.xlabel('x')
# plt.xlabel('y')
# plt.plot(data0[0], data0[1], 'black')
# plt.scatter(data[0], data[1])
# plt.show()

