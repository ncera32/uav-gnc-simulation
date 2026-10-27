import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.linearization import aircraft_state_jacobian
from src.stability import calculate_eigenvalues, extract_state_subsystem

def main():

    # --------------------------------------------
    # Define aircraft and trim conditions
    # --------------------------------------------

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    # ---------------------------------------------
    # Calculate trimmed flight condition
    # ---------------------------------------------

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

    # ------------------------------------------
    # Calculate full state Jacobian
    # ------------------------------------------

    A = aircraft_state_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # -----------------------------------
    # Extract dynamic-state matrix
    # -----------------------------------

    dynamic_indices = [
        3, 4, 5,
        6, 7, 8, 
        9, 10, 11
    ]

    A_dynamic = extract_state_subsystem(
        A,
        dynamic_indices
    )

    # ------------------------------------
    # Calculate eigenvalues
    # ------------------------------------

    eigenvalues_full = calculate_eigenvalues(A)
    eigenvalues_dynamic = calculate_eigenvalues(A_dynamic)

    eigenvalues_full = eigenvalues_full[
        np.argsort(eigenvalues_full.real)
    ]

    eigenvalues_dynamic = eigenvalues_dynamic[
        np.argsort(eigenvalues_dynamic.real)
    ]

    # ---------------------------------------
    # Display results
    # ---------------------------------------

    print("\nReduced Dynamic-State Matrix")
    print("------------------------------")

    np.set_printoptions(
        precision=4,
        suppress=True,
        linewidth=140
    )

    print(A_dynamic)

    print(f"\nMatrix Shape: {A_dynamic.shape}")

    print("\nFull Aircraft Eigenvalues")
    print("---------------------------")

    for eigenvalue in eigenvalues_full:
        print(
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
        )

    print("\nReduced Dynamic-State Eigenvalues")
    print("-----------------------------------")

    for eigenvalue in eigenvalues_dynamic:
        print(
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
        )
    
    print("\nDistance to Nearest Full-System Eigenvalue")
    print("------------------------------------------")

    for eigenvalue in eigenvalues_dynamic:

        distances = np.abs(
            eigenvalues_full - eigenvalue
        )

        nearest_distance = np.min(distances)

        print(
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
            f"  Distance: {nearest_distance:.6e}"
        )

    # ------------------------------------
    # Plot eigenvalue comparison
    # ------------------------------------
        
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        eigenvalues_full.real,
        eigenvalues_full.imag,
        marker="x",
        s=100,
        #s=80,
        color = "tab:blue",
        linewidth = 2,
        label="Full 12-state model",
        zorder = 3
    )

    ax.scatter(
        eigenvalues_dynamic.real,
        eigenvalues_dynamic.imag,
        marker="o",
        facecolors="none",
        #s=100,
        s = 220,
        edgecolors="tab:orange",
        linewidth = 2.5,
        label="Reduced 9-state model",
        zorder = 4
    )

    ax.axvline(0.0, linestyle="--")
    ax.axhline(0.0, linestyle="--")

    ax.set_xlabel("Real Part [1/s]")
    ax.set_ylabel("Imaginary Part [rad/s]")
    ax.set_title("Full vs Reduced Aircraft Eigenvalues")

    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

