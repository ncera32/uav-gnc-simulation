import numpy as np

from src.aircraft import AircraftParameters
from src.trim import solve_longitudinal_trim, longitudinal_trim_residual, longitudinal_trim_state, solve_full_trim, full_trim_residual, full_trim_state
from src.dynamics import aircraft_state_derivative
from src.propulsion import aerosonde_propulsion_forces_moments

aircraft = AircraftParameters()

Va = 25.0
altitude = 500.0

result = solve_longitudinal_trim(
    Va = Va,
    altitude = altitude,
    aircraft = aircraft
)

alpha, delta_e, delta_t = result.x

# uses AoA from optimizer to construct whole aircraft state
trim_state = longitudinal_trim_state(
    Va = Va,
    altitude = altitude,
    alpha = alpha
)

# calls the same nonlinear dynamics function used for trim (checking aircraft model)
derivative = aircraft_state_derivative(
    state = trim_state,
    delta_e = delta_e,
    delta_a = 0.0,
    delta_r = 0.0,
    delta_t = delta_t,
    aircraft = aircraft
)

target_derivative = np.array([
    Va, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0
])

derivative_error = derivative - target_derivative

residual = longitudinal_trim_residual(
    trim_variables = result.x,
    Va = Va,
    altitude = altitude,
    aircraft = aircraft
)

prop_forces, prop_moments = aerosonde_propulsion_forces_moments(
    state = trim_state,
    delta_t = delta_t,
    aircraft = aircraft
)


print("Longitudnial trim results")
print("-------------------------")

print(f"Solver success: {result.success}")
print(f"Solver message: {result.message}")

print(f"Alpha:    {np.rad2deg(alpha):.4f} deg")
print(f"Elevator: {np.rad2deg(delta_e):.4f} deg")
print(f"Throttle: {delta_t:.4f}")

print()
print("Trim Residuals")

print(f"u_dot: {residual[0]:.6e} m/s^2")
print(f"w_dot: {residual[1]:.6e} m/s^2")
print(f"q_dot: {residual[2]:.6e} rad/s^2")

print("\nComplete State Derivatives")
print("----------------------------")

derivative_names = [
    "pn_dot",
    "pe_dot",
    "pd_dot",
    "u_dot",
    "v_dot",
    "w_dot",
    "phi_dot",
    "theta_dot",
    "psi_dot",
    "p_dot",
    "q_dot",
    "r_dot"
]

for name, value in zip(derivative_names, derivative):
    print(f"{name:12s}: {value: .6e}")

print("\nDeviation From Steady-Level Flight")
print("------------------------------------")

for name, value in zip(derivative_names, derivative_error):
    print(f"{name:12s}: {value: .6e}")

print("Propulsion reaction roll moment:", prop_moments[0])

# solve full 6-DOF trim
full_result = solve_full_trim(
    Va = Va,
    altitude = altitude,
    aircraft = aircraft
)

alpha, phi, delta_e, delta_a, delta_r, delta_t = (
    full_result.x
)

full_residual = full_trim_residual(
    trim_variables = full_result.x,
    Va = Va,
    altitude = altitude,
    aircraft = aircraft
)

print("\nFull 6-DOF Trim Results")
print("-------------------------")

print(f"Solver success: {full_result.success}")
print(f"Solver message: {full_result.message}")

print(f"Alpha:    {np.rad2deg(alpha):.4f} deg")
print(f"Bank:     {np.rad2deg(phi):.4f} deg")
print(f"Elevator: {np.rad2deg(delta_e):.4f} deg")
print(f"Aileron:  {np.rad2deg(delta_a):.4f} deg")
print(f"Rudder:   {np.rad2deg(delta_r):.4f} deg")
print(f"Throttle: {delta_t:.4f}")

print("\nFull Trim Residuals")
print("---------------------")

residual_names = [
    "u_dot",
    "v_dot",
    "w_dot",
    "p_dot",
    "q_dot",
    "r_dot"
]

for name, value in zip(residual_names, full_residual):
    print(f"{name:8s}: {value: .6e}")


