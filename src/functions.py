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

def plot_similarity_solution(xi, times, bs, deltas, V0):
    C = 1
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
