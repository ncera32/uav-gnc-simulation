import numpy as np

# calculates airspeed, angle of attack, and sideslip (assuming still air, no wind)
def air_data(state):
    u = state.u
    v = state.v
    w = state.w

    Va = np.sqrt(u**2 + v**2 + w**2)

    if Va <= 1e-8:
        raise ValueError(
            "Airspeed is too small to define "
            "angle of attack and sideslip."
        )
    
    alpha = np.arctan2(w, u)

    beta = np.arcsin(
        np.clip(v / Va, -1.0, 1.0) # np.clip() protects against tiny floating-point rounding errors that could otherwise put the arguments to arcsin() outside valid range
    )

    return Va, alpha, beta

# return dynamics pressure [pa]
def dynamic_pressure(rho, Va):
    if rho < 0:
        raise ValueError("Air density cannot be negative.")
    
    if Va < 0:
        raise ValueError("Airspeed cannot be negative. ")
    
    return 0.5 * rho * Va**2










