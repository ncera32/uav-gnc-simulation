import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.linearization import aircraft_state_jacobian
from src.stability import calculate_eigenvalues, extract_state_subsystem, build_modal_report

def main():

    # ------------------------------------------------
    # Define aircraft and flight condition
    # ------------------------------------------------

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    # ------------------------------------------------
    # Calculate full aircraft trim
    # ------------------------------------------------

    trim_result = solve_full_trim(
        Va=Va,
        altitude=altitude,
        aircraft=aircraft
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

    # ------------------------------------------------
    # Calculate full state jacobian
    # ------------------------------------------------

    A = aircraft_state_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # ------------------------------------------------
    # Extract longitudinal and lateral subsystems
    # ------------------------------------------------

    longitudinal_indices = [3, 5, 10, 7]
    lateral_indices = [4, 9, 11, 6]

    A_longitudinal = extract_state_subsystem(
        A,
        longitudinal_indices
    )

    A_lateral = extract_state_subsystem(
        A,
        lateral_indices
    )

    # ------------------------------------------------
    # Calculate eigenvalues
    # ------------------------------------------------

    eigenvalues_full = calculate_eigenvalues(A)

    eigenvalues_longitudinal = calculate_eigenvalues(
        A_longitudinal
    )

    eigenvalues_lateral = calculate_eigenvalues(
        A_lateral
    )

    # sort eigenvalues by real part
    eigenvalues_full = eigenvalues_full[
        np.argsort(eigenvalues_full.real)
    ]

    eigenvalues_longitudinal = eigenvalues_longitudinal[
        np.argsort(eigenvalues_longitudinal.real)
    ]

    eigenvalues_lateral = eigenvalues_lateral[
        np.argsort(eigenvalues_lateral.real)
    ]

    # ------------------------------------------------
    # Select representative eigenvalues
    # ------------------------------------------------

    # longitudinal complex modes, positive imaginary part
    longitudinal_complex = [
        eig for eig in eigenvalues_longitudinal
        if eig.imag > 1e-8
    ]

    # Sort by damped frequency
    longitudinal_complex.sort(
        key=lambda eig: abs(eig.imag),
        reverse=True
    )

    short_period = longitudinal_complex[0]
    phugoid = longitudinal_complex[1]

    # lat/dir complex mode
    lateral_complex = [
        eig for eig in eigenvalues_lateral
        if eig.imag > 1e-8
    ]

    dutch_roll = lateral_complex[0]

    # lat/dir real modes
    lateral_real = [
        eig for eig in eigenvalues_lateral
        if abs(eig.imag) < 1e-8
    ]

    lateral_real.sort(key=lambda eig: eig.real)

    roll_subsidence = lateral_real[0]
    spiral = lateral_real[1]

    # ------------------------------------------------
    # Build modal report 
    # ------------------------------------------------

    mode_eigenvalues = [
        short_period,
        phugoid,
        dutch_roll,
        roll_subsidence,
        spiral
    ]

    mode_names = [
        "Short-period",
        "Phugoid",
        "Dutch roll",
        "Roll subsidence",
        "Spiral"
    ]

    modal_report = build_modal_report(
        mode_eigenvalues,
        mode_names
    )

    # ^^^ This classification is specific to the modal structure you've observed. At other flight conditions, modes may shift or change character, so a more general automated classifier would need additional checks

    # ------------------------------------------------
    # Display results
    # ------------------------------------------------

    np.set_printoptions(
        precision=6,
        suppress=True,
        linewidth=140
    )

    print("\nLongitudinal State Matrix")
    print("---------------------------")
    print(A_longitudinal)

    print(f"\nShape: {A_longitudinal.shape}")

    print("\nLateral/Directional State Matrix")
    print("----------------------------------")
    print(A_lateral)

    print(f"\nShape: {A_lateral.shape}")

    print("\nLongitudinal Eigenvalues")
    print("--------------------------")

    for eigenvalue in eigenvalues_longitudinal:
        print(
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
        )

    print("\nLateral/Directional Eigenvalues")
    print("---------------------------------")

    for eigenvalue in eigenvalues_lateral:
        print(
            f"{eigenvalue.real:+.6f} "
            f"{eigenvalue.imag:+.6f}j"
        )

    # ------------------------------------------------
    # Print Modal Characteristics
    # ------------------------------------------------
        
    print("\nAircraft Modal Characteristics")
    print("--------------------------------")

    for mode in modal_report:

        name = mode["name"]
        eig = mode["eigenvalue"]

        if mode["stable"]:
            stability = "Stable"
        elif mode["unstable"]:
            stability = "Unstable"
        else:
            stability = "Marginal"

        print(f"\n{name}")
        print(f"  Eigenvalue: {eig.real:+.6f} {eig.imag:+.6f}j")
        print(f"  Stability: {stability}")

        if mode["period"] is not None:
            print(
                f"  Natural Frequency: "
                f"{mode['natural_frequency']:.4f} rad/s"
            )
        else:
            rate_type = (
                "Growth rate" if mode["unstable"]
                else "Decay rate" if mode["stable"]
                else "Rate"
            )

            print(
                f"  {rate_type}: "
                f"{abs(eig.real):.4f} 1/s"
            )

        if mode["period"] is not None:

            print(
                f"  Damping ratio: "
                f"{mode['damping_ratio']:.4f}"
            )

            print(
                f"  Oscillation period: "
                f"{mode['period']:.4f} s"
            )

        if mode["time_constant"] is not None:

            print(
                f"  Characteristic time: "
                f"{mode['time_constant']:.4f} s"
            )

        if mode["doubling_time"] is not None:

            print(
                f"  Doubling time: "
                f"{mode['doubling_time']:.4f} s"
            )

        print(f"  DEBUG time_constant: {mode['time_constant']}")

    # -----------------------------------------
    # Plot eigenvalue comparison
    # -----------------------------------------
        
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        eigenvalues_full.real,
        eigenvalues_full.imag,
        marker="x",
        s=80,
        color="tab:gray",
        label="Full Aircraft",
        zorder=2
    )

    ax.scatter(
        eigenvalues_longitudinal.real,
        eigenvalues_longitudinal.imag,
        marker="o",
        s=180,
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=2,
        label="Longitudinal",
        zorder=4
    )

    ax.scatter(
        eigenvalues_lateral.real,
        eigenvalues_lateral.imag,
        marker="s",
        s=160,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=2,
        label="Lateral/Directional",
        zorder=5
    )

    ax.axvline(0.0, color="gray", linestyle="--")
    ax.axhline(0.0, color="gray", linestyle="--")

    ax.set_xlabel("Real Part [1/s]")
    ax.set_ylabel("Imaginary Part [rad/s]")
    ax.set_title("Aircraft Stability Mode Comparison")

    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()




