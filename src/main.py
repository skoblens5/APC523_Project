## Started overall formatting here. We may want to separate functions into different files so it is easier to work on

### Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit  
import sys
from numpy.polynomial.legendre import leggauss


### Arguments
N = int(sys.argv[1])

### Funcitons
@njit
def gauss_Integral(x_l,w,a1,b1,u):
    # Calculate integral with gauss points x between a1 and b1
    # x: Gauss Points
    # w: Gauss Weights
    # a1: Lower Bound
    # b1: Upper Bound
    # u: Function to Integrate

    N = len(u)
    sum = 0.

    ## PUT SOMETHING HERE - Use trapezoidal_Integral for now

    return sum

@njit
def trapezoidal_Integral(r,u):
    # Calculate integral with gauss points x between a1 and b1
    # r: Radius
    # u: Function to Integrate

    N = len(u)
    sum = 0.
    for i in range(1,N):
        sum += np.pi*(r[i]**2-r[i-1]**2)*(u[i]+u[i-1])/2
    return sum


## This function will contain bulk of solving code
@njit
def delta_step(delta, b, dt, V0):
    g = 9.81
    nu = 0.001

    N = len(delta)
    xi = np.linspace(0, 1, N)
    dxi = xi[1]-xi[0]

    # Calculate delta(xi) at next time step
    delta_new = np.zeros(N)

    # Inner Points
    for i in range(1,N-1):
        delta_new[i] = delta[i] + g*dt/3/nu*(delta[i]**3/b**2 * ((delta[i-1]-2*delta[i]+delta[i+1])/dxi**2 + 1/xi[i]*(delta[i+1]-delta[i-1])/2/dxi) + 3*delta[i]**2/b**2 * (delta[i+1]-delta[i-1])/2/dxi)

    # Boundary Points
    delta_new[0] = delta_new[1]
    delta_new[-1] = 0
    
    # Solve for b_new through integral condition using shooting method
    err = 1
    b_range = np.array((b, 1.5*b))

    while err > 1e-6:
        b_guess = np.mean(b_range)
        V_new = trapezoidal_Integral(b_guess*xi, delta_new)

        if V_new < V0:
            b_range[0] = b_guess
        else:
            b_range[1] = b_guess

        err = np.abs(V-V0)

    b_new = b_guess

    return delta_new, b_new


### Main Script

# Initialize as a circular drop with radius b0 = 1
b0 = 1
delta = np.zeros((N))
xi = np.linspace(0, 1, N) # xi = r/b
for i in range(N):
    delta[i] = np.sqrt(b0**2 - xi[i]**2)

# Ensure BC's are met in initial condition
xi[-1] = 0 #(delta = 0 at xi = 1)
xi[1] = xi[0] # first derivative = 0 at BC (this is order 1 right now, may want to make better!)

# Calculate Initial Volume
p = 5
(x_l,w) = leggauss(p) # Find Gauss Points of Degree p
V0 = trapezoidal_Integral(b0*xi,delta)


## Run Time Stepping
dt = 0.001
t = np.arange(0, 0.1, dt)

# Initialize Stored Values
delta_history = np.zeros(len(t), N) # delta(t)
delta_history[0,:] = delta[:] # Initial Condition
V = np.zeros(len(t)) # Volume
b = np.zeros(len(t)) # Radius
b[0] = b0 # Initial Condition

for i in range(len(t)-1):
    # Calculate Current Volume
    V[i] = trapezoidal_Integral(b[i]*xi, delta_history[i,:])

    # Step
    delta_history[i+1,:], b[i+1] = delta_step(delta_history[i,:], b[i], dt, V0)


## Plotting
plt.figure()
