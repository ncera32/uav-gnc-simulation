import numpy as np

from scipy.integrate import solve_ivp

from src.state import AircraftState
from src.dynamics import state_derivative


#  Integrate rigid-body aircraft motion over a time interval
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