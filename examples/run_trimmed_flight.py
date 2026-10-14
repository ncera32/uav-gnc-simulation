import numpy as np
import matplotlib.pyplot as plt

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state, full_trim_residual
from src.integrator import integrate_aircraft_model

def main():

    aircraft = AircraftParameters()

    Va = 25.0 
    altitude = 500.0

    # solve for steady, level flight
    trim_result = solve_full_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    if not trim_result.success:
        raise RuntimeError("Full trim solver failed.")
    
    alpha, phi, delta_e, delta_a, delta_r, delta_t = (
        trim_result.x
    )

    # construct the initial trimmed aircraft state
    initial_state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    residual = full_trim_residual(
        trim_variables = trim_result.x,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    # ensures that it doesn't simulate poorly converged trim condition
    if np.max(np.abs(residual)) > 1e-6:
        raise RuntimeError(
            "Trim residuals are too large for simulation."
        )
    
    # simulation settings
    t_start = 0.0
    t_end = 20.0
    dt = 0.02

    t_eval = np.linspace(
        t_start,
        t_end,
        int((t_end - t_start) / dt) + 1
    )

    # simulate with constant trimmed controls
    solution = integrate_aircraft_model(
        initial_state = initial_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft,
        t_span = (t_start, t_end),
        t_eval = t_eval
    )

    if not solution.success:
        raise RuntimeError(
            f"Integration failed: {solution.message}"
        )
    
    t = solution.t 
    y = solution.y

    pn = y[0]
    pe = y[1]
    pd = y[2]

    u = y[3]
    v = y[4]
    w = y[5]

    phi_history = y[6]
    theta_history = y[7]
    psi_history = y[8]

    p = y[9]
    q = y[10]
    r = y[11]

    # derived flight quantities
    
    # For this initial zero-wind simulation, the magnitude of the body-frame velocity is the airspeed
    # Later, when add wind, will need to distinguish air-relative velocity from ground-relative velocity
    airspeed = np.sqrt(u**2 + v**2 + w**2)
    altitude_history = -pd

    # DEVIATION FROM TRIM 
    
    airspeed_error = airspeed - Va
    altitude_error = altitude_history - altitude

    bank_error = np.rad2deg(
        phi_history - initial_state.phi
    )

    pitch_error = np.rad2deg(
        theta_history - initial_state.theta
    )

    heading_error = np.rad2deg(
        psi_history - initial_state.psi
    )

    print("\nTrimmed Flight Simulation")
    print("---------------------------")

    print(f"Duration: {t_end:.1f} s")
    print(f"Target Airspeed: {Va:.2f} m/s")
    print(f"Target Altitude: {altitude:.2f} m")

    print("\nMaximum Absolute Deviations")
    print("-----------------------------")

    print(
        f"Airspeed: {np.max(np.abs(airspeed_error)):.6e} m/s"
    )

    print(
        f"Altitude: {np.max(np.abs(altitude_error)):.6e} m"
    )

    print(
        f"Bank: {np.max(np.abs(bank_error)):.6e} deg"
    )

    print(
        f"Pitch: {np.max(np.abs(pitch_error)):.6e} deg"
    )

    print(
        f"Heading: {np.max(np.abs(heading_error)):.6e} deg"
    )


    # Airspeed and Altitude Deviation Plots

    plt.figure()

    plt.plot(t, airspeed_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Airspeed Error [m/s]")
    plt.title("Trimmed Flight: Airspeed Deviation")
    plt.grid(True)

    plt.figure()

    plt.plot(t, altitude_error)
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude Error [m]")
    plt.title("Trimmed Flight: Altitude Deviation")
    plt.grid(True)

    # Attitude Deviation Plots

    plt.figure()

    plt.plot(t, bank_error, label = "Roll")
    plt.plot(t, pitch_error, label = "Pitch")
    plt.plot(t, heading_error, label = "Yaw")

    plt.xlabel("Time [s]")
    plt.ylabel("Attitude Deviation [deg]")
    plt.title("Trimmed Flight: Attitude Deviations")
    plt.legend()
    plt.grid(True)


    # Plot body angular rates

    plt.figure()

    plt.plot(t, np.rad2deg(p), label = "p")
    plt.plot(t, np.rad2deg(q), label = "q")
    plt.plot(t, np.rad2deg(r), label = "r")

    plt.xlabel("Time [s]")
    plt.ylabel("Angular Rates [deg/s]")
    plt.title("Trimmed Flight: Body Angular Rates")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()














