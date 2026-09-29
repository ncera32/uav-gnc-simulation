import numpy as np

from src.air_data import dynamic_pressure, air_data
from src.atmosphere import density

# calculate lift and drag magnitude [N]
def lift_drag(q_bar, S, CL, CD):
    lift = q_bar * S * CL
    drag = q_bar * S * CD

    return lift, drag

# convert lift and drag to body-axis forces (assume beta = 0)
def lift_drag_to_body(lift, drag, alpha):
    Fx = -drag * np.cos(alpha) + lift * np.sin(alpha)
    Fy = 0.0
    Fz = -drag * np.sin(alpha) - lift * np.cos(alpha)

    return np.array([Fx, Fy, Fz], dtype=float)

# Calculate longitudinal aerodynamic coefficients
def longitudinal_coefficients(alpha, pitch_rate, delta_e, Va, c, coefficients):
    if Va <= 1e-8:
        raise ValueError(
            "Airspeed is too small this aerodynamic model."
        )
    
    q_hat = c * pitch_rate / (2.0 * Va)

    CL = (
        coefficients["CL0"]
        + coefficients["CL_alpha"] * alpha
        + coefficients["CL_q"] * q_hat
        + coefficients["CL_delta_e"] * delta_e
    )

    CD = (
        coefficients["CD0"]
        + coefficients["CD_alpha"] * alpha
        + coefficients["CD_q"] * q_hat
        + coefficients["CD_delta_e"] * delta_e
    )

    Cm = (
        coefficients["Cm0"]
        + coefficients["Cm_alpha"] * alpha
        + coefficients["Cm_q"] * q_hat
        + coefficients["Cm_delta_e"] * delta_e
    )

    return CL, CD, Cm

# return the aerodynamic pitching moment about CG [Nm]
def pitching_moment(q_bar, S, c, Cm):
    return q_bar * S * c * Cm

# calculate longitudinal aerodynamic forces and moments (assume zero sideslip and linear coefficient model)
def longitudinal_forces_moments(rho, Va, alpha, pitch_rate, delta_e, S, c, coefficients):
    q_bar = dynamic_pressure(rho, Va)

    CL, CD, Cm = longitudinal_coefficients(
        alpha = alpha, 
        pitch_rate = pitch_rate,
        delta_e = delta_e,
        Va = Va,
        c = c,
        coefficients = coefficients
    )

    lift, drag = lift_drag(
        q_bar = q_bar, 
        S = S,
        CL = CL,
        CD = CD
    )

    forces_body = lift_drag_to_body(
        lift = lift,
        drag = drag, 
        alpha = alpha
    )

    M = pitching_moment(
        q_bar = q_bar,
        S = S,
        c = c,
        Cm = Cm
    )

    # two other moments are zero because we are deliberately only implementing longitudinal aerodynamics
    moments_body = np.array([
        0.0, 
        M,
        0.0
    ])

    return forces_body, moments_body

# WRAPPER Calculate longitudinal aerodynamic forces and moments using the current aircraft state and Aerosonde parameters.
def aerosonde_longitudinal_forces_moments(state, delta_e, aircraft):

    # 1. calculate altitude from NED position
    altitude = -state.pd

    # 2. calculate atmospheric density at that altitude
    rho = density(altitude)

    # 3. calculate airspeed and angle of attack
    Va, alpha, beta = air_data(state)

    # This prevents from accidentally treating the longitudinal-only model as a complete aerodynamic model during a maneuver with sideslip
    # remove this restriction when we implement the full three-dimensional aerodynamic model
    if not np.isclose(beta, 0.0, atol = 1e-8):
        raise ValueError(
            "Longitudinal aerodynamic model requires zero sideslip"
        )

    # 4. Gather the aersonde longitudinal coefficients
    coefficients = {
        "CL0": aircraft.CL0,
        "CL_alpha": aircraft.CL_alpha,
        "CL_q": aircraft.CL_q,
        "CL_delta_e": aircraft.CL_delta_e,

        "CD0": aircraft.CD0,
        "CD_alpha": aircraft.CD_alpha,
        "CD_q": aircraft.CD_q,
        "CD_delta_e": aircraft.CD_delta_e,

        "Cm0": aircraft.Cm0,
        "Cm_alpha": aircraft.Cm_alpha,
        "Cm_q": aircraft.Cm_q,
        "Cm_delta_e": aircraft.Cm_delta_e
    }

    # 5. calculate longitudinal forces and moments
    return longitudinal_forces_moments(
        rho = rho,
        Va = Va,
        alpha = alpha,
        pitch_rate = state.q,
        delta_e = delta_e,
        S = aircraft.S,
        c = aircraft.c,
        coefficients = coefficients
    )

# ^^^ Later, simplify this interface by passing aircraft directly into longitudinal_coefficients()


# Calculate linear lateral-directional coefficients (angles and angular rates in radians and radians/sec)
def lateral_directional_coefficients(beta, roll_rate, yaw_rate, delta_a, delta_r, Va, b, coefficients):
    if Va <= 1e-8:
        raise ValueError(
            "Airspeed is too small for this aerodynamic model."
        )
    
    # Non-dimensional angular rates
    p_hat = b * roll_rate / (2.0 * Va)
    r_hat = b * yaw_rate / (2.0 * Va)

    # Side-force coefficient
    CY = (
        coefficients["CY0"]
        + coefficients["CY_beta"] * beta
        + coefficients["CY_p"] * p_hat
        + coefficients["CY_r"] * r_hat
        + coefficients["CY_delta_a"] * delta_a
        + coefficients["CY_delta_r"] * delta_r
    )

    # Rolling moment coefficient
    Cl = (
        coefficients["Cl0"]
        + coefficients["Cl_beta"] * beta
        + coefficients["Cl_p"] * p_hat
        + coefficients["Cl_r"] * r_hat
        + coefficients["Cl_delta_a"] * delta_a
        + coefficients["Cl_delta_r"] * delta_r
    )

    # Yawing moment coefficient
    Cn = (
        coefficients["Cn0"]
        + coefficients["Cn_beta"] * beta
        + coefficients["Cn_p"] * p_hat
        + coefficients["Cn_r"] * r_hat
        + coefficients["Cn_delta_a"] * delta_a
        + coefficients["Cn_delta_r"] * delta_r
    )

    return CY, Cl, Cn


# Calculate lateral-directional forces and moments
def lateral_directional_forces_moments(rho, Va, beta, roll_rate, yaw_rate, delta_a, delta_r, S, b, coefficients):
    q_bar = dynamic_pressure(rho, Va)

    CY, Cl, Cn = lateral_directional_coefficients(
        beta = beta,
        roll_rate = roll_rate,
        yaw_rate = yaw_rate,
        delta_a = delta_a,
        delta_r = delta_r,
        Va = Va,
        b = b,
        coefficients = coefficients
    )

    Fy = q_bar * S * CY

    L = q_bar * S * b * Cl
    N = q_bar * S * b * Cn

    forces_body = np.array([
        0.0,
        Fy,
        0.0
    ])

    moments_body = np.array([
        L,
        0.0,
        N
    ])

    return forces_body, moments_body









