import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def explicit(mu, v):
    for n in range(v.shape[1]-1): # 0,1,2,3,...,Nt-2
        for i in range(1, v.shape[0]-1): # 1,2,3,...,Nx-2
            v[i, n+1] = mu*(v[i-1, n]+v[i+1, n])+(1-2*mu)*v[i, n]
    """
    for n in range(v.shape[1]-1): # 0,1,2,3,...,Nt-2
        v[1:-1, n+1] = mu(v[:-2, n]+v[2:, n])+(1-2*mu)*v[1:-1, n]
    """
    return v

def thomasAlgorithm(a, b, c, d):
    """
    Tri-diagonal matrix solver using Thomas Algorithm.
    a, c: arrays for the lower and upper diagonals
    b: array for the main diagonal
    d: right hand side
    Returns the solution x
    """
    n = len(d)
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)

    # Forward sweep
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n-1):
        temp = b[i] - a[i-1]*c_[i-1]
        c_[i] = c[i] / temp
        d_[i] = (d[i] - a[i-1]*d_[i-1]) / temp
    d_[n-1] = (d[n-1] - a[n-2]*d_[n-2]) / (b[n-1] - a[n-2]*c_[n-2])

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]

    return x


def implicit(mu, v):
    Nx = v.shape[0]
    Nt = v.shape[1]

    a = np.full(Nx - 2, -mu)
    b = np.full(Nx - 1, 1 + 2 * mu)
    c = np.full(Nx - 2, -mu)

    a[0] = 0
    c[-1] = 0

    for n in range(0, Nt - 1):
        d = v[1:-1, n].copy()
        d[0] += mu * v[0, n + 1]
        d[-1] += mu * v[-1, n + 1]
        v[1:-1, n + 1] = thomasAlgorithm(a, b, c, d)

    return v

#nx number of spatial steps, number of size of timestep
def solveDiffusion1D(Nx=11, Nt=100, alpha=0.5,
                     dt=0.01, xmin=0.0, xmax=1.0,
                     initialConditions=lambda x : 0.,
                     lowerBoundary    =lambda t : 0.,
                     upperBoundary    =lambda t : 0.,
                     method=explicit ):
    x = np.linspace(xmin, xmax, Nx)
    dx = x[1]-x[0]
    t = np.arange(0, Nt)*dt
    v = np.empty([Nx, Nt])
    v[ :, 0] = initialConditions(x)
    v[ 0, :] = lowerBoundary(t)
    v[-1, :] = upperBoundary(t)
    mu = alpha*dt/(dx**2)
    v = method(mu, v)
    return x, t, v

def f(x):
    return np.sin(np.pi*x)

def g(t):
    return f(0.)

def h(t):
    return f(1.)

def solution(x, t, alpha=0.5):
    return np.exp(-alpha * (np.pi ** 2.) * t)*f(x)


x, t, v = solveDiffusion1D(initialConditions=f,
                           lowerBoundary=g,
                           upperBoundary=h)



T, X = np.meshgrid(t, x)
u = solution(X, T)
kwargs = {'projection':'3d'}
#fig, ax = plt.subplots(1, 1, subplot_kw=kwargs)
fig, axs = plt.subplots(1, 3, subplot_kw=kwargs, sharex=True, sharey=True)
axs = list(axs)
ax = axs[0]
ax.plot_wireframe(X, T, v, color='red')
ax.plot_wireframe(X, T, u, color='blue')
ax.set_zlabel(r'$v_i^{<n>}$')
ax = axs[1]
ax.plot_wireframe(X[1:-1, :], T[1:-1, :], np.abs(v-u)[1:-1, :], color='green')
ax.set_zlabel(r'$v_i^{<n>}-u(x_i,t^{<n>})|$')
ax.set_xlim(*axs[0].get_xlim())
ax.set_ylim(*axs[0].get_xlim())
ax = axs[2]
ax.plot_wireframe(X[1:-1, :], T[1:-1, :], np.abs(v-u)[1:-1, :]/u[1:-1, :], color='purple')
#ax.set_zlabel(r'$\left|\dfrac{v_i^{<n>}-u(x_i,t^{<n>})|$')
ax.set_xlim(*axs[0].get_xlim())
ax.set_ylim(*axs[0].get_xlim())
for ax in axs:
    ax.set_xlabel(r'$x_i$')
    ax.set_ylabel(r'$t^{<n>}$')
    ax.set_zlabel(r'$v_i^{<n>}$')
plt.show()