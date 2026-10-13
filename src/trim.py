import numpy as np

from src.state import AircraftState
from src.dynamics import aircraft_state_derivative
from scipy.optimize import least_squares

# Construct a candidate state for steady, straight, level longitudinal flight.
def longitudinal_trim_state(
        Va,
        altitude,
        alpha
):
    
    u = Va * np.cos(alpha)
    w = Va * np.sin(alpha)

    state = AircraftState(
        pn = 0.0,
        pe = 0.0,
        pd = -altitude,

        u = u,
        v = 0.0,
        w = w,

        phi = 0.0,
        theta = alpha,
        psi = 0.0,

        p = 0.0,
        q = 0.0,
        r = 0.0
    )

    return state 


def longitudinal_trim_residual(
        trim_variables,
        Va,
        altitude,
        aircraft
):
    
    #alpha = trim_variables[0]
    #delta_e = trim_variables[1]
    #delta_t = trim_variables[2]

    alpha, delta_e, delta_t = trim_variables

    state = longitudinal_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha
    )

    derivative = aircraft_state_derivative(
        state = state,
        delta_e = delta_e,
        delta_a = 0.0,
        delta_r = 0.0,
        delta_t = delta_t,
        aircraft = aircraft
    )

    residual = np.array([
        derivative[3],   # u_dot
        derivative[5],   # w_dot
        derivative[10]   # q_dot
    ])

    return residual

# Solve for steady, level longitudinal flight trim.
def solve_longitudinal_trim(
        Va,
        altitude,
        aircraft,
        initial_guess = None
):
    
    if Va <= 0.0:
        raise ValueError("Target airspeed must be positive.")
    
    # Initial Guess
    if initial_guess is None:
        initial_guess = np.array([
            np.deg2rad(5.0),
            np.deg2rad(-2.0),
            0.5
        ])

    # Provisional physical/search bounds
    lower_bounds = np.array([
        np.deg2rad(-10.0),
        np.deg2rad(-25.0),
        0.0
    ])
    
    upper_bounds = np.array([
        np.deg2rad(15.0),
        np.deg2rad(25.0),
        1.0
    ])

    result = least_squares(
        fun = longitudinal_trim_residual,
        x0 = initial_guess,
        bounds = (lower_bounds, upper_bounds),
        args = (Va, altitude, aircraft),
        xtol = 1e-10,
        ftol = 1e-10,
        gtol = 1e-10
    )

    return result


# Construct a candidate state for steady, level flight with zero sideslip (no elevator, aileron, rudder, throttle bc/ they are control inputs not state inputs)
def full_trim_state(
        Va,
        altitude,
        alpha,
        phi
):
    
    if Va <= 0.0:
        raise ValueError("Target airspeed must be positive.")
    
    # Body-frame velocity components
    u = Va * np.cos(alpha)
    v = 0.0
    w = Va * np.sin(alpha)

    # pitch attitude required for level flight
    theta = np.arctan2(
        np.cos(phi) * np.sin(alpha),
        np.cos(alpha)
    )

    state = AircraftState(
        pn = 0.0,
        pe = 0.0,
        pd = -altitude,

        u = u,
        v = v,
        w = w,

        phi = phi,
        theta = theta,
        psi = 0.0,

        p = 0.0,
        q = 0.0,
        r = 0.0
    )

    return state

# Evaluate the six body-axis acceleration residual for a candidate steady, level flight condition
def full_trim_residual(
        trim_variables,
        Va,
        altitude,
        aircraft
):
    
    alpha, phi, delta_e, delta_a, delta_r, delta_t = (
        trim_variables
    )

    # construct candidate state
    state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    # evaluate complete nonlinear dynamics
    derivative = aircraft_state_derivative(
        state = state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # extract translational and rotational accelerations
    residual = np.array([
        derivative[3], # u_dot
        derivative[4], # v_dot
        derivative[5], # w_dot
        derivative[9], # p_dot
        derivative[10], # q_dot
        derivative[11] # r_dot
    ])

    return residual

# Solve for steady, level, zero-angular-rate flight trim.
def solve_full_trim(
        Va,
        altitude,
        aircraft,
        initial_guess = None
):
    
    if Va <= 0.0:
        raise ValueError("Target airspeed must be positive.")
    
    # Provisional trim search bounds
    lower_bounds = np.array([
        np.deg2rad(-10.0), # alpha
        np.deg2rad(-15.0), # phi
        np.deg2rad(-25.0), # delta_e
        np.deg2rad(-25.0), # delta_a
        np.deg2rad(-25.0), # delta_r
        0.0                # delta_t
    ])

    upper_bounds = np.array([
        np.deg2rad(15.0), # alpha
        np.deg2rad(15.0), # phi
        np.deg2rad(25.0), # delta_e
        np.deg2rad(25.0), # delta_a
        np.deg2rad(25.0), # delta_r
        1.0                # delta_t
    ])

    # start from previously validated longitudinal trim
    if initial_guess is None:

        longitudinal_result = solve_longitudinal_trim(
            Va = Va,
            altitude = altitude,
            aircraft = aircraft
        )

        if not longitudinal_result.success:
            raise RuntimeError(
                "Longitudinal trim solver failed."
            )
        
        alpha, delta_e, delta_t = longitudinal_result.x

        initial_guess = np.array([
            alpha,
            0.0,     # phi
            delta_e,
            0.0,     # delta_a
            0.0,     # delta_r
            delta_t 
        ])

    # solve full trim problem
    result = least_squares(
        fun = full_trim_residual,
        x0 = initial_guess,
        bounds = (lower_bounds, upper_bounds),
        args = (Va, altitude, aircraft),
        xtol = 1e-10,
        ftol = 1e-10,
        gtol = 1e-10,
        max_nfev = 2000
    )

    return result




