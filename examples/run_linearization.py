import numpy as np

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.linearization import aircraft_state_jacobian, aircraft_control_jacobian

def main():

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

    A = aircraft_state_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    B = aircraft_control_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    print("\nAircraft State Jacobian A")
    print("---------------------------")

    np.set_printoptions(
        precision = 4,
        suppress = True,
        linewidth = 140
    )

    print(A)

    print(f"\nMatrix Shape: {A.shape}")
    print(f"All finite: {np.all(np.isfinite(A))}")

    print("\nAircraft Control Jacobian B")
    print("-----------------------------")

    np.set_printoptions(
        precision = 6,
        suppress = True,
        linewidth = 140
    )

    print(B)

    print(f"\nMatrix Shape: {B.shape}")
    print(f"All finite: {np.all(np.isfinite(B))}")

    print("\nSelected Control Derivatives")
    print("------------------------------")

    print(f"Elevator -> Pitch acceleration: {B[10, 0]:.6f}")
    print(f"Aileron -> Roll acceleration: {B[9, 1]:.6f}")
    print(f"Rudder -> Yaw acceleration: {B[11, 2]:.6f}")
    print(f"Throttle -> Forward acceleration: {B[3, 3]:.6f}")
    print(f"Throttle -> Roll acceleration: {B[9, 3]:.6f}")

if __name__ == "__main__":
    main()
