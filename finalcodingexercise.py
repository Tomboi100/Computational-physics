import numpy as np
import matplotlib.pyplot as plt

def thomasAlgorithm(a, b, c, d):
    y = np.empty(a.size)
    x = y.copy()

    gamma = 1./b[0]
    y[0] = c[0]*gamma
    x[0] = d[0]*gamma

    for j in range(1, a.size):
        gamma = 1./( b[j] - a[j]*y[j-1])
        y[j] = c[j]*gamma
        x[j] = (d[j]+a[j]*x[j-1]) * gamma

    for j in reversed(range(a.size-1)):
        x[j] += y[j]*x[j+1]
    return x

def hybrid_Neumann(mu, v, theta=0.5):
    N = v.shape[0]

    a = theta * mu * np.ones(N)
    b = (1 + theta * mu) * np.ones(N)
    c = a.copy()
    d = np.empty(N)

    a[0] = 0
    c[-1] = 0.
    b[1:-1] += theta * mu

    for n in range(v.shape[1]-1):
        d[0] = (1.-(1.-theta) * mu)*v[0, n] + (1.-theta) * mu * v[1, n]
        d[1:-1] = (1.-2. * (1.-theta) * mu)*v[1:-1, n]+(1.-theta) * mu * (v[:-2, n]+v[2:, n])
        d[-1] = (1.-theta) * mu * v[-2, n]+(1.-(1.-theta) * mu)*v[-1, n]
        v[:, n + 1] = thomasAlgorithm(a, b, c, d)
    return v

def solveDiffusion1D_withCellCenteredGrid(Nx=20, Nt=50, alpha=0.5, dt=0.01, xmin=0.0, xmax=1.0,
                                          initialConditions=lambda x: 0., lowerBoundary=lambda t: 0.,
                                          upperBoundary=lambda t: 0., method=hybrid_Neumann):
    dx = (xmax - xmin) / Nx
    x = xmin + (np.arange(Nx) + 0.5) * dx
    t = np.arange(Nt) * dt
    v = np.empty([Nx, Nt])
    v[:, 0] = initialConditions(x)
    v[0, :] = lowerBoundary(t)
    v[-1, :] = upperBoundary(t)
    mu = alpha * dt / (dx ** 2)
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

def plot(x, t, v):
    T, X = np.meshgrid(t, x)
    u = solution(X, T)
    kwargs = {'projection': '3d'}
    fig, axs = plt.subplots(1, 3, subplot_kw=kwargs, sharex=True, sharey=True)
    axs = list(axs)
    ax = axs[0]
    ax.plot_wireframe(X, T, v, color='red')
    ax.plot_wireframe(X, T, u, color='blue')
    ax.set_zlabel(r'$v_i^{<n>}$')
    ax = axs[1]
    ax.plot_wireframe(X[1:-1, :], T[1:-1, :], np.abs(v - u)[1:-1, :], color='green')
    ax.set_zlabel(r'$v_i^{<n>}-u(x_i,t^{<n>})|$')
    ax.set_xlim(*axs[0].get_xlim())
    ax.set_ylim(*axs[0].get_xlim())
    ax = axs[2]
    ax.plot_wireframe(X[1:-1, :], T[1:-1, :], np.abs(v - u)[1:-1, :] / u[1:-1, :], color='purple')
    ax.set_zlabel(r'$\left|\dfrac{v_i^{<n>}-u(x_i,t^{<n>})|$')
    ax.set_xlim(*axs[0].get_xlim())
    ax.set_ylim(*axs[0].get_xlim())
    for ax in axs:
        ax.set_xlabel(r'$x_i$')
        ax.set_ylabel(r'$t^{<n>}$')
        ax.set_zlabel(r'$v_i^{<n>}$')
    plt.show()

if __name__ == '__main__':
    print("Program Start")

    x, t, v = solveDiffusion1D_withCellCenteredGrid(initialConditions=f,
                                                    lowerBoundary=g,
                                                    upperBoundary=h)
    plot(x, t, v)

    print("Program End")