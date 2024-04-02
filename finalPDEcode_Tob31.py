import numpy as np
import matplotlib.pyplot as plt

def thomasAlgorithm(a, b, c, d): # The Thomas Algorithm used for solving a tridiagonal matrix
    y = np.empty(a.size)
    x = y.copy()

    # forward sweep
    gamma = 1./b[0] # initalization
    y[0] = c[0]*gamma
    x[0] = d[0]*gamma

    for j in range(1, a.size): # iteratively adjusts the coefficients
        gamma = 1./( b[j] - a[j]*y[j-1])
        y[j] = c[j]*gamma
        x[j] = (d[j]+a[j]*x[j-1]) * gamma

    for j in reversed(range(a.size-1)): #back subtitution
        x[j] += y[j]*x[j+1]
    return x

def hybridNeumann(mu, v, theta=0.5): # modifided hybrid method for neumann boundary conditions
    N = v.shape[0]

    # setting the coefficients
    a = theta * mu * np.ones(N)
    b = (1 + theta * mu) * np.ones(N)
    c = a.copy()
    d = np.empty(N)

    a[0] = 0
    c[-1] = 0.
    b[1:-1] += theta * mu

    # Time step
    for n in range(v.shape[1]-1):
        #updating d in each time step
        d[0] = (1.-(1.-theta) * mu)*v[0, n] + (1.-theta) * mu * v[1, n]
        d[1:-1] = (1.-2. * (1.-theta) * mu)*v[1:-1, n]+(1.-theta) * mu * (v[:-2, n]+v[2:, n])
        d[-1] = (1.-theta) * mu * v[-2, n]+(1.-(1.-theta) * mu)*v[-1, n]
        v[:, n + 1] = thomasAlgorithm(a, b, c, d) # solve using thomas at each time step
    return v

def solveDiffusion1D_withCellCenteredGrid(Nx=20, Nt=50, alpha=0.5, dt=0.01, xmin=0.0, xmax=1.0, # method for solving the 1d diffusion equation with cell centered gridding
                                          initialConditions=lambda x: 0., lowerBoundary=lambda t: 0.,
                                          upperBoundary=lambda t: 0., method=hybridNeumann):
    # creating the grid
    dx = (xmax - xmin) / Nx
    x = xmin + (np.arange(Nx) + 0.5) * dx
    t = np.arange(Nt) * dt

    # solution matrix initialization
    v = np.empty([Nx, Nt])
    v[:, 0] = initialConditions(x)
    v[0, :] = lowerBoundary(t)
    v[-1, :] = upperBoundary(t)

    mu = alpha * dt / (dx ** 2) # diffusion coefficent

    v = method(mu, v) # solving

    w = v.sum(axis=0) # sum of v for each time step
    return x, t, v, w

# functions for initial and boundary conditions
def InitialConditions(x):
    return np.sin(np.pi*x)

def UpperBoundary(t):
    return InitialConditions(0.)

def LowerBoundary(t):
    return InitialConditions(1.)

def solution(x, t, alpha=0.5): # analytical solution
    return np.exp(-alpha * (np.pi ** 2.) * t)*InitialConditions(x)


def displayGraphs(x, t, v, w): # function for plotting the
    T, X = np.meshgrid(t, x)

    # setting up the plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_wireframe(X, T, v, color='red')
    ax.set_xlabel(r'$x_i$')
    ax.set_ylabel(r'$t^{<n>}$')
    ax.set_zlabel(r'$v_i^{<n>}$')

    fig.add_subplot(1, 4, 4)  # 2D subplot for total v over time
    plt.plot(t, w, color='pink')
    plt.xlabel('Time')
    plt.ylabel('Total v')
    plt.title('Conservation of Total v(w) over Time')
    plt.show() # display plots

if __name__ == '__main__':
    # calling the method
    x, t, v, w = solveDiffusion1D_withCellCenteredGrid(initialConditions=InitialConditions,
                                                    lowerBoundary=UpperBoundary,
                                                    upperBoundary=LowerBoundary)
    # plotting the data function
    displayGraphs(x, t, v, w)