import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.linearization import aircraft_state_jacobian
from src.stability import calculate_eigenvalues

def main():

    # -----------------------------------------------------
    # Define aircraft and flight conditions
    # -----------------------------------------------------

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    # -----------------------------------------------------
    # Calculate trimmed flight condition
    # -----------------------------------------------------

    trim_result = solve_full_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    if not trim_result.success:
        raise RuntimeError("Trim solver failed.")
    
    alpha, phi, delta_e, delta_a, delta_r, delta_t = (
        trim_result.x
    )

    trim_state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    # -----------------------------------------------------
    # Calculate state Jacobian
    # -----------------------------------------------------

    A = aircraft_state_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # -----------------------------------------------------
    # Calculate eigenvalues
    # -----------------------------------------------------

    eigenvalues = calculate_eigenvalues(A)

    # sort by real part for easier inspection
    eigenvalues = eigenvalues[
        np.argsort(eigenvalues.real)
    ]

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nOpen-loop Aircraft Eigenvalues")
    print("--------------------------------")

    for i, eigenvalue in enumerate(eigenvalues):

        print(
            f"Mode {i + 1:2d}: "
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
        )

    print(f"\nNumber of eigenvalues: {len(eigenvalues)}")

    # -----------------------------------------------------
    # Plot eigenvalues in complex plane
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        marker = "x",
        s = 80
    )

    ax.axvline(0.0, linestyle="--")
    ax.axhline(0.0, linestyle="--")

    ax.set_xlabel("Real Part [1/s]")
    ax.set_ylabel("Imaginary Part [rad/s]")
    ax.set_title("Open-loop Aircraft Eigenvalues")

    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


