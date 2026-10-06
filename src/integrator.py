import numpy as np

from scipy.integrate import solve_ivp

from src.state import AircraftState
from src.dynamics import state_derivative, aircraft_state_derivative


#  Integrate rigid-body aircraft motion over a time interval (CONSTANT FORCES AND MOMENTS)
def integrate_aircraft(
    initial_state,
    forces_body,
    moments_body,
    mass,
    inertia,
    t_span,
    t_eval=None
):
    
    x0 = initial_state.to_array()

    def equations_of_motion(t, x):
        state = AircraftState.from_array(x)

        return state_derivative(
            state,
            forces_body,
            moments_body,
            mass,
            inertia
        )
    
    solution = solve_ivp(
        fun = equations_of_motion,
        t_span = t_span,
        y0 = x0,
        t_eval = t_eval,
        method = "RK45",
        rtol = 1e-8,
        atol = 1e-10
    )

    if not solution.success:
        raise RuntimeError(
            f"Aircraft integration failed: {solution.message}"
        )
    
    return solution 

# Integrator for varying forces/moments!!!! (controllers remain constant - for V1 not adding controller yet)
def integrate_aircraft_model(
        initial_state,
        delta_e,
        delta_a,
        delta_r,
        delta_t,
        aircraft,
        t_span,
        t_eval
):
    
    def ode_function(t, y):

        state = AircraftState.from_array(y)

        derivative = aircraft_state_derivative(
            state = state,
            delta_e = delta_e,
            delta_a = delta_a,
            delta_r = delta_r,
            delta_t = delta_t,
            aircraft = aircraft
        )

        return derivative
    
    solution = solve_ivp(
        ode_function,
        t_span = t_span,
        y0 = initial_state.to_array(),
        t_eval = t_eval,
        method = "RK45",
        rtol = 1e-8,
        atol = 1e-10
    )

    return solution