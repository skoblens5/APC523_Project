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
def delta_step(delta, r, dt):
    # Calculate delta at next time step

    N = len(delta)






### Main Script

# Initialize as a circular drop with radius b0
b0 = 1
r_max = 10
delta = np.zeros((N))
r = np.linspace(0, r_max, N)
for i in range(N):
    if r[i] <= b0:
        delta[i] = np.sqrt(b0**2 - r[i]**2)
    else:
        delta[i] = 0

# Calculate Initial Volume
p = 5
(x_l,w) = leggauss(p) # Find Gauss Points of Degree p
V0 = trapezoidal_Integral(r,delta)




## Run Time Stepping
dt = 0.01
t = np.arange(0, 1, dt)

# Initialize Stored Values
V = np.zeros(len(t))


for i in range(len(t)):
    # Calculate Volume
    V[i] = trapezoidal_Integral(r,delta)

    # Step
    delta_new = delta_step(delta, r, dt)

    # Update delta
    delta = delta_new
