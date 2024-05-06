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
    
