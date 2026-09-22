import numpy as np
import matplotlib.pyplot as plt

from src.state import AircraftState
from src.integrator import integrate_aircraft

def main():
    mass = 10.0
    g = 9.80665

    initial_state = AircraftState(
        pd = -100.0,
        u = 25.0,
        psi = np.deg2rad(45.0)
    )

    solution = integrate_aircraft(
        initial_state = initial_state,
        forces_body = np.array([0.0, 0.0, -mass * g]),
        moments_body = np.zeros(3),
        mass = mass,
        inertia = np.diag([2.0, 3.0, 4.0]),
        t_span = (0.0, 20.0),
        t_eval = np.linspace(0.0, 20.0, 201)
    )

    north = solution.y[0, :]
    east = solution.y[1, :]
    altitude = -solution.y[2, :]

    plt.figure()

    plt.plot(east, north)

    plt.xlabel("East Position [m]")
    plt.ylabel("North Position [m]")
    plt.title("Prescribed-Force UAV Trajectory")

    plt.axis("equal")
    plt.grid(True)

    plt.show()

    print(f"Final North Position: {north[-1]:.2f} m")
    print(f"Final East Position: {east[-1]:.2f} m")
    print(f"Final Altitude: {altitude[-1]:.2f} m")

if __name__ == "__main__":
    main()