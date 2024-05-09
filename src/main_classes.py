import numpy as np

from drop_simulation import DropSimulation

def main():
    # Simulation parameters
    N = 100  # Number of spatial points
    DT = 0.01  # Time step
    SIMULATION_TIME = 10  # Total simulation time
    method = 'implicit'  # Choose 'explicit' or 'implicit'
    integration_method = 'trapezoidal'  # Choose 'gauss', 'trapezoidal'

    # Initialize the drop simulation
    simulator = DropSimulation(N, DT, SIMULATION_TIME, method, integration_method)

    # Run the simulation
    results = simulator.run_simulation()

    # Output the results
    print("Simulation completed successfully.")
    for key, value in results.items():
        print(f"{key}: {value[-1]}")  # Print the last elements in each list to check final state

if __name__ == "__main__":
    main()

