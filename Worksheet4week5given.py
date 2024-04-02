import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colorsys import hsv_to_rgb

def thomasAlgorithm(a, b, c, d):
    y = np.empty(a.size)
    x = y.copy()
    gamma = 1./b[0]
    y[0] = c[0]*gamma
    x[0] = d[0]*gamma
    for j in range(1, a.size):
        gamma = 1./( b[j] - a[j]*y[j-1] )
        y[j] = c[j]*gamma
        x[j] = ( d[j] + a[j]*x[j-1] )*gamma
    for j in reversed(range(a.size-1)):
        x[j] += y[j]*x[j+1]
    return x

def explicit(mu, v):
    """
    for n in range(v.shape[1]-1): # 0,1,2,3,...,Nt-2
        for i in range(1, v.shape[0]-1): # 1,2,3,...,Nx-2
            v[i, n+1] = mu*(v[i-1, n]+v[i+1, n])+(1-2*mu)*v[i, n]
    """
    for n in range(v.shape[1]-1): # 0,1,2,3,...,Nt-2
        v[1:-1, n+1] = mu*(v[:-2, n]+v[2:, n])+(1-2*mu)*v[1:-1, n]
    return v

def implicit(mu, v):
    N = v.shape[0]-2
    a = mu*np.ones(N)
    c = a.copy()
    a[0], c[-1] = 0, 0
    b = np.ones(N)+2*mu
    for n in range(v.shape[1]-1):
        d = v[1:-1, n].copy()
        d[0] += mu*v[0, n+1]
        d[-1] += mu*v[-1, n+1]
        v[1:-1, n+1] = thomasAlgorithm(a, b, c, d)
    return v

def hybrid(mu, v, theta=0.5):
    N = v.shape[0]-2
    a = theta*mu*np.ones(N)
    c = a.copy()
    a[0], c[-1] = 0, 0
    b = np.ones(N)+2*theta*mu
    p, q = (1.-theta)*mu, (1.-2.*(1.-theta)*mu)
    for n in range(v.shape[1]-1):
        d = p*(v[:-2, n]+v[2:, n ])+q*v[1:-1, n]
        v[1:-1, n+1] = thomasAlgorithm(a, b, c, d)
    return v

# def solveDiffusion1D_withDirichletBoundaries(Nx=20, Nt=50, alpha=0.5,
#                                              dt=0.01, xmin=0.0, xmax=1.0,
#                                              initialConditions=lambda x : 0.,
#                                              lowerBoundary    =lambda t : 0.,
#                                              upperBoundary    =lambda t : 0.,
#                                              method=explicit,
#                                              **kwargs):
#     x = np.linspace(xmin, xmax, Nx)
#     dx = x[1]-x[0]
#     #dx = (xmax - xmin)/Nx
#     t = np.arange(0, Nt)*dt
#     v = np.empty([Nx, Nt])
#     v[ :, 0] = initialConditions(x)
#     v[ 0, :] = lowerBoundary(t)
#     v[-1, :] = upperBoundary(t)
#     mu = alpha*dt/(dx**2)
#     v = method(mu, v, **kwargs)
#     return x, t, v

def solveDiffusion1D_withDirichletBoundaries(Nx=20, Nt=50, alpha=0.5,
                                             dt=0.01, xmin=0.0, xmax=1.0,
                                             initialConditions=lambda x : 0.,
                                             lowerBoundary    =lambda t : 0.,
                                             upperBoundary    =lambda t : 0.,
                                             method=explicit,
                                             **kwargs):
    x = np.linspace(xmin, xmax, Nx)
    #dx = x[1]-x[0]
    dx = (xmax - xmin)/Nx
    for i in range(x.size):
        x[i] = xmin + ((i+0.5)*dx)
    t = np.arange(0, Nt)*dt
    v = np.empty([Nx, Nt])
    v[ :, 0] = initialConditions(x)
    v[ 0, :] = lowerBoundary(t)
    v[-1, :] = upperBoundary(t)
    mu = alpha*dt/(dx**2)
    v = method(mu, v, **kwargs)
    return x, t, v

def f(x):
    return np.sin(np.pi*x)

def g(t):
    return f(0.)

def h(t):
    return f(1.)

def solution(x, t, alpha=0.5):
    return np.exp(-alpha*(np.pi**2)*t)*f(x)

common_kwargs = {'initialConditions' : f,
                 'lowerBoundary'     : g,
                 'upperBoundary'     : h }

methods = [ explicit                   ,
            implicit                   ,
            hybrid                     ,
           (hybrid  , {'theta' : 0.75}),
           (implicit, {'Nx'    : 30  }),
           (hybrid  , {'theta' : 0.3 }) ]

solve = solveDiffusion1D_withDirichletBoundaries
results = []
for m in methods:
    if type(m) is tuple:
        results.append( solve(**common_kwargs, method=m[0], **m[1]) )
    else:
        results.append( solve(**common_kwargs, method=m) )

#x, t, v_exp = solveDiffusion1D(**common_kwargs)
#x, t, v_imp = solveDiffusion1D(**common_kwargs, method=implicit)
#x, t, v_hyb = solveDiffusion1D(**common_kwargs, method=hybrid)
#x, t, v_alt = solveDiffusion1D(**common_kwargs, method=hybrid, theta=0.75)
#T, X = np.meshgrid(t, x)
#u = solution(X, T)

kwargs = {'projection':'3d'}
fig, axs = plt.subplots(1, 3, subplot_kw=kwargs)
axs = list(axs)
cf = 1./len(results)
for i, xtv in enumerate(results):
    x, t, v = xtv
    T, X = np.meshgrid(t, x)
    u = solution(X, T)
    c = hsv_to_rgb(i*cf, 1., 1.)
    axs[0].plot_wireframe(X, T, u, color='black')
    axs[0].plot_wireframe(X, T, v, color=c)
    axs[1].plot_wireframe(X[1:-1, :], T[1:-1, :], np.abs(v-u)[1:-1, :],
                      color=c)
    axs[2].plot_wireframe(X[1:-1, :], T[1:-1, :],
                          np.abs(v-u)[1:-1, :]/u[1:-1, :],
                          color=c)
for ax in axs:
    ax.set_xlim(*axs[0].get_xlim())
    ax.set_ylim(*axs[0].get_ylim())
    ax.set_xlabel(r'$x_i$')
    ax.set_ylabel(r'$t^{<n>}$')
ax.set_zlabel(r'$v_i^{<n>}$')
ax.set_zlabel(r'$|v_i^{<n>}-u(x_i, t^{<n>})|$')
s = r'$\left|\dfrac{v_i^{<n>}-u(x_i, t^{<n>})}{u(x_i, t^{<n>})}\right|$'
ax.set_zlabel(s)

fig.tight_layout()

active = None
live = False

def onEnter(event):
    global active
    for ax in axs:
        if event.inaxes==ax:
            active = ax
            break

def onLeave(event):
    global active, live
    active = None
    live = False

def onPress(event):
    global live
    if active is None: return
    live = True

def onRelease(event):
    global live
    if active is None: return
    live = False

def onMotion(event):
    if live:
        for ax in axs:
            if ax is not active:
                ax.view_init(elev=active.elev, azim=active.azim)
    fig.canvas.draw()
    plt.pause(0.05)

"""
cids = [fig.canvas.mpl_connect('axes_enter_event'    , onEnter  ),
        fig.canvas.mpl_connect('axes_leave_event'    , onLeave  ),
        fig.canvas.mpl_connect('button_press_event'  , onPress  ),
        fig.canvas.mpl_connect('button_release_event', onRelease),
        fig.canvas.mpl_connect('motion_notify_event' , onMotion ) ]

plt.show(block=False)
"""
azim = 0.
while plt.fignum_exists(fig.number):
    azim += 1
    azim %= 360
    for ax in axs:
        ax.view_init(azim=azim)
    fig.canvas.draw()
    plt.pause(0.05)