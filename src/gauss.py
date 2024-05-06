## Started overall formatting here. We may want to separate functions into different files so it is easier to work on

### Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import sys
from numpy.polynomial.legendre import leggauss
from scipy.sparse import diags
from scipy.sparse.linalg import bicgstab
from numpy.linalg import norm
from matplotlib import cm
from matplotlib.colors import Normalize
import pylab
params = {
    'legend.fontsize': 'x-Large',
    'figure.figsize': (12, 6),
    'axes.labelsize': 16,
    'axes.titlesize':20,
    'xtick.labelsize':16,
    'ytick.labelsize':16,
    'text.usetex': False,  # Disable LaTeX rendering
}
pylab.rcParams.update(params)

@njit
def trapezoidal_integral(r, u):
    integral = 0.0
    for i in range(1, len(u)):
        area = np.pi * (r[i]**2 - r[i-1]**2)
        average_height = (u[i] + u[i-1]) / 2
        integral += area * average_height
    return integral

def stability_study():
    plt.figure(figsize=(12, 8))
    dts = [0.00001, 0.00005, 0.0001]
    colors = ['blue', 'green', 'red']
    for dt, color in zip(dts, colors):
        xi, times, bs, volumes, deltas = main_simulation(dt)
        plt.plot(times, bs, label=f'DT={dt}', color=color, linewidth=2)

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Scaling Factor b', fontsize=14)
    plt.title('Evolution of Scaling Factor b Over Time', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def precompute_gauss_legendre(n):
    """Precomputes Gauss-Legendre nodes and weights for given n."""
    nodes, weights = leggauss(n)
    return nodes, weights

@njit
def radial_gauss_legendre_integral(r, f_r, nodes, weights):
    """Performs radial integration using Gauss-Legendre quadrature."""
    a, b = r[0], r[-1]
    r_mapped = 0.5 * (nodes + 1) * (b - a) + a  # Remap nodes to the radial domain
    f_mapped = np.interp(r_mapped, r, f_r)      # Interpolate f at the mapped radial points

    integral = np.sum(2 * np.pi * r_mapped * f_mapped * weights * 0.5 * (b - a))
    return integral

@njit
def simpsons_rule_radial(xi, delta):
    """Calculate the radial integral of `delta` over `xi` using Simpson's rule."""
    h = (xi[-1] - xi[0]) / (len(xi) - 1)
    integral = delta[0] * xi[0]**2 + delta[-1] * xi[-1]**2  # Considera el radio en los extremos
    
    # Suma ponderada para los términos interiores
    sum_even = np.sum(delta[2:-2:2] * xi[2:-2:2]**2)
    sum_odd = np.sum(delta[1:-1:2] * xi[1:-1:2]**2)
    
    integral += 2 * sum_even + 4 * sum_odd
    integral *= h / 3 * np.pi  # np.pi factor para la integral sobre un círculo completo
    
    return integral
    
def plot_similarity_solution(xi, times, bs, deltas, V0):
    plt.figure(figsize=(10, 6))
    cmap = cm.viridis  # Choose a colormap
    norm = Normalize(vmin=min(times), vmax=max(times))  # Normalize times for color mapping

    # Create a ScalarMappable for the colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Calculate how many points to skip to reduce the plot to about 100 points
    step = max(1, len(times) // 100)

    # Plot each profile with color mapped to time
    for i in range(0, len(times), step):
        time = times[i]
        if time >= 0:  # Avoid division by zero and initial time
            nu = xi * bs[i] * (3 * np.pi**3 * C**3 / (V0**3 * G * time))**(1/8)
            f = deltas[i] / (3 * V0 * (1 / (16 * np.pi * C * G * time)))**(1/4)
            plt.plot(nu, f, color=cmap(norm(time)))

    plt.colorbar(sm, label='Time (s)')
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$f(\nu)$')
    plt.title('Self-similar Solution Over Time')
    plt.xlim([0,4])
    plt.grid(True)
    plt.show()
