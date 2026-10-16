import numpy as np

from src.state import AircraftState
from src.dynamics import aircraft_state_derivative

# Compute a numerical Jacobian using central differences

def numerical_jacobian(function, x, step_sizes):

    x = np.asarray(x, dtype=float)
    step_sizes = np.asarray(step_sizes, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input x must be one-dimensional.")
    
    if step_sizes.shape != x.shape:
        raise ValueError(
            "Step sizes must match the shape of x."
        )
    
    if not np.all(np.isfinite(step_sizes)) or np.any(step_sizes <= 0):
        raise ValueError(
            "All step sizes must be finite and positive."
        )
    
    f0 = np.asarray(function(x), dtype=float)

    if f0.ndim != 1:
        raise ValueError(
            "Function output must be one-dimensional."
        )
    
    jacobian = np.zeros((f0.size, x.size))

    for j in range(x.size):

        h = step_sizes[j] 

        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += h
        x_minus[j] -= h

        f_plus = np.asarray(
            function(x_plus),
            dtype=float
        )

        f_minus = np.asarray(
            function(x_minus),
            dtype=float
        )

        jacobian[:, j] = (
            f_plus - f_minus
        ) / (2.0 * h)

    return jacobian


# Calculate the state Jacobian A for the nonlinear aircraft model

def aircraft_state_jacobian(
        state,
        delta_e,
        delta_a,
        delta_r,
        delta_t,
        aircraft,
        step_sizes=None
):
    
    x0 = state.to_array()

    def dynamics_from_array(x):

        current_state = AircraftState.from_array(x)

        return aircraft_state_derivative(
            state = current_state,
            delta_e = delta_e,
            delta_a = delta_a,
            delta_r = delta_r,
            delta_t = delta_t,
            aircraft = aircraft
        )
    
    # perturbation sizes follow the established state order:
    # [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]

    if step_sizes is None:
        step_sizes = np.array([
        1e-3, # pn [m]
        1e-3, # pe [m]
        1e-3, # pd [m]

        1e-4, # u [m/s]
        1e-4, # v [m/s]
        1e-4, # w [m/s]

        1e-5, # phi [rad]
        1e-5, # theta [rad]
        1e-5, # psi [rad]

        1e-5, # p [rad/s]
        1e-5, # q [rad/s]
        1e-5  # r [rad/s]
    ])


    #step_sizes = np.array([
    #    1e-3, # pn [m]
    #    1e-3, # pe [m]
    #    1e-3, # pd [m]

    #    1e-4, # u [m/s]
    #    1e-4, # v [m/s]
    #    1e-4, # w [m/s]

    #    1e-5, # phi [rad]
    #    1e-5, # theta [rad]
    #    1e-5, # psi [rad]

    #    1e-5, # p [rad/s]
    #    1e-5, # q [rad/s]
    #    1e-5  # r [rad/s]
    #])

    return numerical_jacobian(
        function = dynamics_from_array,
        x = x0,
        step_sizes = step_sizes
    )







    
    

