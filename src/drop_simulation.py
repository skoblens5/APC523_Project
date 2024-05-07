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
from functions import *
from integrators import *


class DropSimulation:
    def __init__(self, N, DT, SIMULATION_TIME, method, integration_method):
        self.N = N
        self.DT = DT
        self.SIMULATION_TIME = SIMULATION_TIME
        self.method = method
        self.integration_method = integration_method
        self.xi = np.linspace(0, 1, N)
        self.delta = np.sqrt(1 - self.xi**2)
        self.delta[0] = self.delta[1]  # Boundary conditions
        self.b = 1.0
        self.G = 9.81  # Gravity
        self.NU = 1.0  # Viscosity

        self.initial_volume = self.calculate_initial_volume()

    def calculate_initial_volume(self):
        if self.integration_method == 'gauss':
            nodes, weights = self.precompute_gauss_legendre()
            return self.radial_gauss_legendre_integral(self.xi * self.b, self.delta, nodes, weights)
        elif self.integration_method == 'trapezoidal':
            return self.trapezoidal_integral(self.xi * self.b, self.delta)
        elif self.integration_method == 'simpson':
            return self.simpsons_rule_radial(self.xi * self.b, self.delta)
        else:
            raise ValueError("Unsupported integration method")
    @staticmethod
    @njit
    def trapezoidal_integral(r, u):
        integral = 0.0
        for i in range(1, len(u)):
            area = np.pi * (r[i]**2 - r[i-1]**2)
            average_height = (u[i] + u[i-1]) / 2
            integral += area * average_height
        return integral
    
    def precompute_gauss_legendre(self):
        """Precomputes Gauss-Legendre nodes and weights for given n."""
        nodes, weights = leggauss(self.N)
        return nodes, weights
    @staticmethod
    @njit
    def radial_gauss_legendre_integral(r, f_r, nodes, weights):
        """Performs radial integration using Gauss-Legendre quadrature."""
        a, b = r[0], r[-1]
        r_mapped = 0.5 * (nodes + 1) * (b - a) + a  # Remap nodes to the radial domain
        f_mapped = np.interp(r_mapped, r, f_r)      # Interpolate f at the mapped radial points

        integral = np.sum(2 * np.pi * r_mapped * f_mapped * weights * 0.5 * (b - a))
        return integral

    @staticmethod
    @njit
    def simpsons_rule_radial(xi, delta):
        """Calculate the radial integral of `delta` over `xi` using Simpson's rule."""
        n = len(xi) - 1  # Number of intervals, ensure it's even for Simpson's rule
        if n % 2 == 1:
            raise ValueError("Simpson's rule requires an even number of intervals (n+1 must be odd)")

        h = (xi[-1] - xi[0]) / n
        integral = delta[0] * xi[0]**2 + delta[-1] * xi[-1]**2  # Weights for the endpoints
        
        sum_even = np.sum(delta[2:n:2] * xi[2:n:2]**2)  # Sum terms with even indices
        sum_odd = np.sum(delta[1:n:2] * xi[1:n:2]**2)   # Sum terms with odd indices
        
        integral += 2 * sum_even + 4 * sum_odd
        integral *= h / 3 * np.pi  # h/3 factor and pi for the radial integral
        return integral
    
    def spectral_radius(self, A_reduced):
        eig_values = np.linalg.eigvals(A_reduced)
        return np.max(np.abs(eig_values))
    
    @staticmethod
    @njit
    def step(self, delta, b, xi, dt):
        delta_new = np.zeros_like(delta)
        for i in range(1, self.N-1):
            r_factor = delta[i]**3 / b**2
            dxi = xi[i+1] - xi[i]
            derivative = (delta[i-1] - 2 * delta[i] + delta[i+1]) / dxi**2
            convective = 3 * delta[i]**2 / b**2 * ((delta[i+1] - delta[i-1]) / (2 * dxi))**2
            delta_new[i] = delta[i] + self.DT * self.G / (3 * self.NU) * (r_factor * derivative + convective)
        delta_new[0] = delta_new[1]
        delta_new[-1] = 0
        return delta_new

    def create_matrix(self, u, radius, xi, dt):
        """
        Creates a matrix A for the given method ('explicit' or 'implicit') specified by self.method.
        
        Args:
        u (np.array): Current heights of the fluid.
        radius (float): Base radius of the drop.
        xi (np.array): Spatial discretization points.
        dt (float): Time step.
        
        Returns:
        np.array: The matrix A used in the numerical solver.
        """
        N = len(xi)
        dxi = xi[1] - xi[0]
        Out = np.zeros((N, N))
        k = self.G * dt / (3 * self.NU)  # Common factor

        for i in range(1, N-1):
            a = u[i]**3 / radius**2
            b = 3 * u[i]**2 / radius**2
            if self.method == 'explicit':
                A = a * k * (1 / dxi**2 - 1 / (2 * xi[i] * dxi)) + k * b / 4 / dxi**2 * (u[i-1] - u[i+1])
                B = 1 - 2 * a * k / dxi**2
                C = a * k * (1 / dxi**2 + 1 / (2 * xi[i] * dxi)) + k * b / 4 / dxi**2 * (u[i+1] - u[i-1])
            elif self.method == 'implicit':
                A = a * k * (1 / dxi**2 - 1 / (2 * xi[i] * dxi)) + k * b / 4 / dxi**2 * (u[i-1] - u[i+1])
                B = -1 - 2 * a * k / dxi**2
                C = a * k * (1 / dxi**2 + 1 / (2 * xi[i] * dxi)) + k * b / 4 / dxi**2 * (u[i+1] - u[i-1])
            else:
                raise ValueError("Unsupported method specified. Choose 'explicit' or 'implicit'.")

            Out[i, i-1] = A
            Out[i, i] = B
            Out[i, i+1] = C

        # Boundary conditions are set here for both explicit and implicit methods
        Out[N-1, N-1] = 1
        Out[0, 0] = 1
        Out[0, 1] = -1

        return Out



    def update_profile(self, delta, b, xi, dt):
        """
        Updates the drop profile using either an explicit or implicit method.
        
        Args:
        delta (np.array): Current drop profile heights.
        b (float): Base radius of the drop.
        xi (np.array): Spatial discretization points.
        dt (float): Time step.
        
        Returns:
        np.array: Updated drop profile heights.
        float: Possibly modified time step.
        """
        delta_new = np.zeros_like(delta)
        if self.method == 'explicit':
            A = self.create_matrix(delta, b, xi, dt)
            A_reduced = A[1:-2, 1:-2]
            spectral_rad = self.spectral_radius(A_reduced)
            while spectral_rad >= 1:
                dt *= 0.9  # Reduce time step to achieve stability
                A = self.create_matrix(delta, b, xi, dt)
                A_reduced = A[1:-2, 1:-2]
                spectral_rad = self.spectral_radius(A_reduced)
            delta_new = self.step(delta, b, xi, dt)  # Update using explicit step
        elif self.method == 'implicit':
            RHS = np.zeros(self.N)
            RHS[1:-1] = -delta[1:-1]  # Setup RHS for implicit method
            A = self.create_matrix(delta, b, xi, dt)
            delta_new = np.linalg.solve(A, RHS)  # Solve the linear system A*delta_new = RHS
        else:
            raise ValueError("Unsupported method specified. Choose 'explicit' or 'implicit'.")

        return delta_new, dt
    
    def volume_shooting(self, b, delta_new):
        """
        Adjusts the base radius 'b' to conserve the volume using the specified integration method.
        """
        b_range = np.array((0.5 * b, 3 * b))
        err = 1

        while err > 1E-14:
            b_guess = np.mean(b_range)
            V_new = self.calculate_volume(b_guess, delta_new)

            if V_new > self.initial_volume:
                b_range[1] = b_guess
            elif V_new < self.initial_volume:
                b_range[0] = b_guess
            else:
                break

            err = np.abs(V_new - self.initial_volume)

        return b_guess

    def calculate_volume(self, b, delta):
        # Calculate volume based on integration method
        if self.integration_method == 'gauss':
            nodes, weights = self.precompute_gauss_legendre()
            return self.radial_gauss_legendre_integral(b * self.xi, delta, nodes, weights)
        elif self.integration_method == 'trapezoidal':
            return self.trapezoidal_integral(b * self.xi, delta)
        elif self.integration_method == 'simpson':
            return self.simpsons_rule_radial(b * self.xi, delta)
        
        
    def run_simulation(self):
        volumes = [self.initial_volume]
        bs = [self.b]
        times = [0]
        deltas = [self.delta.copy()]
        time_val = 0

        while time_val < self.SIMULATION_TIME:
            delta_new, dt = self.update_profile(self.delta, self.b, self.xi, self.DT)
            V_new = self.calculate_volume(self.b, delta_new)
            
            self.b = self.volume_shooting(self.b, delta_new)
            
            # Update and log data
            self.delta = delta_new
            volumes.append(V_new)
            bs.append(self.b)
            times.append(time_val)
            deltas.append(self.delta.copy())
            
            #print(f"Time: {time_val}, Delta New: {delta_new[:5]}, V_new: {V_new}")
            time_val += dt

        return {'xi': self.xi, 'times': times, 'bs': bs, 'volumes': volumes, 'deltas': deltas}
