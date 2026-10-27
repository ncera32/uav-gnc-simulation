import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.linearization import aircraft_state_jacobian
from src.stability import calculate_eigenmodes, scale_state_matrix


def main():

    # ---------------------------------------------
    # Aircraft and trim conditions
    # ---------------------------------------------

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

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

    # ---------------------------------------------
    # Calculate state Jacobian
    # ---------------------------------------------

    A = aircraft_state_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # ---------------------------------------------
    # Define state scales
    # ---------------------------------------------

    state_names = [
        "pn", "pe", "pd",
        "u", "v", "w",
        "phi", "theta", "psi",
        "p", "q", "r"
    ]

    state_scales = np.array([
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0)
    ])

    # ---------------------------------------------
    # Calculate scaled eigenmodes
    # ---------------------------------------------

    A_scaled = scale_state_matrix(
        A,
        state_scales
    )

    eigenvalues, eigenvectors = calculate_eigenmodes(A_scaled)

    # Sort eigenvalues and matching eigenvectors
    order = np.argsort(eigenvalues.real)

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # ---------------------------------------------
    # Print dominant state components
    # ---------------------------------------------

    print("\nAircraft Eigenmode Participation")
    print("----------------------------------")

    for i, eigenvalue in enumerate(eigenvalues):

        eigenvector = eigenvectors[:, i]

        magnitudes = np.abs(eigenvector)

        normalized = magnitudes / np.max(magnitudes)

        dominant_indices = np.argsort(normalized)[-5:][::-1]

        print(
            f"\nMode {i + 1}: "
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
        )

        for index in dominant_indices:

            print(
               f" {state_names[index]:>6s}: " 
               f"{normalized[index]:.4f}"
            )

    # ---------------------------------------------
    # Plot mode shapes
    # ---------------------------------------------
            
    fig, axes = plt.subplots(
        3, 4, 
        figsize = (18, 11)
    )

    axes = axes.flatten()

    for i in range(len(eigenvalues)):

        magnitudes = np.abs(eigenvectors[:, i])

        normalized = magnitudes / np.max(magnitudes)

        axes[i].bar(
            state_names,
            normalized
        )

        axes[i].set_title(
            f"Mode {i + 1}: "
            f"{eigenvalues[i].real:+.3f} "
            f"{eigenvalues[i].imag:+.3f}j"
        )

        axes[i].set_ylim(0.0, 1.1)
        axes[i].tick_params(
            axis="x",
            rotation=60
        )

    fig.suptitle(
        "Scaled Aircraft Eigenvector Magnitudes",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


