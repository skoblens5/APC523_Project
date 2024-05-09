# Viscous Drop Spreading Simulation

## Overview
This repository contains the source code and results for a computational fluid dynamics project focusing on the behavior of an axisymmetric viscous drop spreading under the influence of gravity. The project explores the dynamics using numerical solutions to solve non-linear partial differential equations with integral constraints and specific boundary conditions.



## Project Structure

The main folder, `APC523_Project`, contains a `src` directory, along with a `README.md` file for project documentation. The `src` directory houses all the source files essential for the simulation's operation. This includes `drop_simulation.py`, which serves as the primary simulation module for the explicit method, and `drop_simulation2.py` for the implicit method. Additional support files within this directory include `functions.py` for mathematical operations, `integrators.py` for various integration schemes, `main.py` for basic setup and execution, `main_notebook.ipynb` for simulation experiments within a Jupyter notebook, and `testing.ipynb` for testing and validation of the simulation methods. Graphical results from the simulations are stored in the `Figures` directory, which includes `ProfileImplicit_1024.svg` for high-resolution results and `ProfileImplicit_32.svg` for low-resolution results from the implicit method simulations. 


## Simulation Details
The simulation investigates the spreading of a viscous fluid drop on a surface by numerically solving the governing equation:

$$
\delta^3\left[\frac{\partial^2\delta}{\partial r^2} + \frac{1}{r}\frac{\partial\delta}{\partial r}\right] + 3\delta^2\left[\frac{\partial\delta}{\partial r}\right]^2 - 3\frac{\nu}{g}\frac{\partial\delta}{\partial t} = 0
$$

where:
- $\delta(r, t)$ is the thickness of the drop,
- $r$ is the radial distance from the center,
- $\nu$ is the fluid's viscosity,
- $g$ is gravitational acceleration,
- $t$ is time.

Boundary conditions and volume conservation laws are applied to ensure the physical accuracy of the simulation.

## How to Run
To execute the simulation:
1. Navigate to the `src` directory.
2. Run the `main.py` to perform a basic simulation or main_classes.py.
3. For a detailed exploration, open and run the cells in `main_notebook.ipynb`.

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Jupyter (for .ipynb files)

## Authors
- Samuel Koblensky
- Clara Martin Blanco


## Acknowledgments
This project was developed as part of the coursework for APC523 at Princeton University.


