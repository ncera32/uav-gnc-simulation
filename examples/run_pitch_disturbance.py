import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import AircraftParameters
from src.state import AircraftState
from src.trim import solve_full_trim, full_trim_state
from src.integrator import integrate_aircraft_model


def extract_flight_variables(solution):

    y = solution.y

    u = y[3]
    v = y[4]
    w = y[5]

    theta = y[7]
    pitch_rate = y[10]

    airspeed = np.sqrt(u**2 + v**2 + w**2)

    alpha = np.arctan2(w, u)

    altitude = -y[2]

    return {
        "airspeed": airspeed,
        "alpha": alpha,
        "theta": theta,
        "pitch_rate": pitch_rate,
        "altitude": altitude
    }


def main():

    aircraft = AircraftParameters

    Va_trim = 25.0
    altitude_trim = 500.0

    trim_result = solve_full_trim(
        Va = Va_trim,
        altitude = altitude_trim,
        aircraft = aircraft
    )

    if not trim_result.success:
        raise RuntimeError("Full trim solver failed.")
    
    alpha, phi, delta_e, delta_a, delta_r, delta_t = (
        trim_result.x
    )

    trim_state = full_trim_state(
        Va = Va_trim,
        altitude = altitude_trim,
        alpha = alpha,
        phi = phi
    )

    # initial pitch-rate disturbance
    pitch_rate_disturbance = 0.05 # rad/s

    # copy trim state vector
    disturbed_state_array = trim_state.to_array().copy()

    # q is index 10 in state vector
    disturbed_state_array[10] += pitch_rate_disturbance

    # construct the disturbed initial state
    disturbed_state = AircraftState.from_array(
        disturbed_state_array
    )

    t_start = 0.0
    t_end = 30.0
    dt = 0.02

    t_eval = np.linspace(
        t_start,
        t_end,
        int((t_end - t_start) / dt) + 1
    )

    baseline_solution = integrate_aircraft_model(
        initial_state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft,
        t_span = (t_start, t_end),
        t_eval = t_eval
    )

    disturbed_solution = integrate_aircraft_model(
        initial_state = disturbed_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft,
        t_span = (t_start, t_end),
        t_eval = t_eval
    )

    if not baseline_solution.success:
        raise RuntimeError(
            f"Baseline integration failed: "
            f"{baseline_solution.message}"
        )
    
    if not disturbed_solution.success:
        raise RuntimeError(
            f"Disturbed integration failed: "
            f"{disturbed_solution.message}"
        )
    
    baseline = extract_flight_variables(
        baseline_solution
    )

    disturbed = extract_flight_variables(
        disturbed_solution
    )

    t = disturbed_solution.t

    pitch_rate_error = np.rad2deg(
        disturbed["pitch_rate"]
        - baseline["pitch_rate"]
    )

    pitch_error = np.rad2deg(
        disturbed["theta"]
        - baseline["theta"]
    )

    alpha_error = np.rad2deg(
        disturbed["alpha"]
        - baseline["alpha"]
    )

    airspeed_error = (
        disturbed["airspeed"]
        - baseline["airspeed"]
    )

    altitude_error = (
        disturbed["altitude"]
        - baseline["altitude"]
    )

    # PLOTTING

    fig, axes = plt.subplots(
        5, 1,
        figsize = (10, 12),
        sharex = True
    )

    axes[0].plot(t, pitch_rate_error)
    axes[0].set_ylabel("Pitch Rate [deg/s]")
    axes[0].set_title(
        "Open-loop Response to Initial Pitch-rate Disturbance"
    )

    axes[1].plot(t, pitch_error)
    axes[1].set_ylabel("Pitch Angle [deg]")

    axes[2].plot(t, alpha_error)
    axes[2].set_ylabel("AoA [deg]")

    axes[3].plot(t, airspeed_error)
    axes[3].set_ylabel("Airspeed [m/s]")

    axes[4].plot(t, altitude_error)
    axes[4].set_ylabel("Altitude [m]")
    axes[4].set_xlabel("Time [s]")

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nPitch Rate Disturbance Simulation")
    print("-----------------------------------")

    print(
        f"Initial Disturbance: "
        f"{np.rad2deg(pitch_rate_disturbance):.3f} deg/s"
    )

    print(f"Simulation duration: {t_end:.1f} s")

    print("\nMaximum Absolute Deviations")
    print("-----------------------------")

    print(
        f"Pitch Rate: "
        f"{np.max(np.abs(pitch_rate_error)):.4f} deg/s"
    )

    print(
        f"Pitch Angle: "
        f"{np.max(np.abs(pitch_error)):.4f} deg"
    )

    print(
        f"Angle of Attack: "
        f"{np.max(np.abs(alpha_error)):.4f} deg"
    )

    print(
        f"Airspeed: "
        f"{np.max(np.abs(airspeed_error)):.4f} m/s"
    )

    print(
        f"Altitude: "
        f"{np.max(np.abs(altitude_error)):.4f} m"
    )

    print("\nFinal Deviations at 30 Seconds")
    print("--------------------------------")

    print(f"Pitch Rate: {pitch_rate_error[-1]:.4f} deg/s")
    print(f"Pitch Angle: {pitch_error[-1]:.4f} deg")
    print(f"Angle of Attack: {alpha_error[-1]:.4f} deg")
    print(f"Airspeed: {airspeed_error[-1]:.4f} m/s")
    print(f"Altitude: {altitude_error[-1]:.4f} m")

if __name__ == "__main__":
    main()


