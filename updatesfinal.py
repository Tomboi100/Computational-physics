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
    Nx = v.shape[0]
    Nt = v.shape[1]
    a = np.zeros(Nx)
    b = np.ones(Nx) * (1 + theta * mu)
    c = np.zeros(Nx)
    d = np.zeros(Nx)

    # Modify coefficients according to Neumann boundary conditions
    a[1] = theta * mu
    c[-2] = theta * mu
    b[0] = b[-1] = 1 + 2 * theta * mu

    for n in range(Nt - 1):
        # Compute d based on the hybrid method and Neumann boundary conditions
        d[0] = (1 - theta) * mu * v[1, n] + (1 - 2 * (1 - theta) * mu) * v[0, n]
        d[1:-1] = (1 - theta) * mu * (v[:-2, n] + v[2:, n]) + (1 - 2 * (1 - theta) * mu) * v[1:-1, n]
        d[-1] = (1 - theta) * mu * v[-2, n] + (1 - 2 * (1 - theta) * mu) * v[-1, n]

        # Solve the tridiagonal system for the next time step
        v[:, n + 1] = thomasAlgorithm(a, b, c, d)

    return v

def f(x):
    return np.sin(np.pi*x)

def g(t):
    return f(0.)

def h(t):
    return f(1.)

def solution(x, t, alpha=0.5):
    return np.exp(-alpha * (np.pi ** 2.) * t)*f(x)

def solveDiffusion1D_withCellCenteredGrid(Nx, Nt, alpha, dt, theta=0.5, initialConditionType='sin'):
    dx = 1.0 / Nx
    x = np.linspace(dx / 2, 1 - dx / 2, Nx)
    t = np.arange(0, Nt) * dt
    v = np.zeros((Nx, Nt))

    # Set initial conditions
    if initialConditionType == 'sin':
        v[:, 0] = np.sin(np.pi * x)
    elif initialConditionType == 'step':
        v[:, 0] = np.where(x <= 0.5, 0, 1)
    elif initialConditionType == 'exp':
        v[:, 0] = np.exp(x)
    else:
        raise ValueError("Unknown initial condition type")

    # Apply Dirichlet boundary conditions
    v[0, :] = np.sin(np.pi * 0)
    v[-1, :] = np.sin(np.pi * 1)

    # Solve the diffusion equation
    mu = alpha * dt / dx ** 2
    v = hybrid_Neumann(mu, v, theta)

    # Check if total v (w) is conserved over time
    w = np.sum(v, axis=0)
    if np.allclose(w, w[0]):
        print("Total v (w) is conserved over time.")
    else:
        print("Total v (w) is not conserved over time.")

    return x, t, v


# Test the function
Nx = 20
Nt = 50
alpha = 0.5
dt = 0.01
theta = 0.5

for initial_condition in ['sin', 'step', 'exp']:
    x, t, v = solveDiffusion1D_withCellCenteredGrid(Nx, Nt, alpha, dt, theta, initial_condition)
    T, X = np.meshgrid(t, x)
    u = solution(X, T)
    kwargs = {'projection': '3d'}
    # fig, ax = plt.subplots(1, 1, subplot_kw=kwargs)
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
