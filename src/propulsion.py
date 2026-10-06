import numpy as np

from src.atmosphere import density
from src.air_data import air_data

# calculate motor input voltage from throttle command
def input_voltage(delta_t, V_max):

    if not 0.0 <= delta_t <= 1.0:
        raise ValueError(
            "Throttle command must be between 0 and 1."
        )
    
    return V_max * delta_t

# Calculate the coefficients of the quadratic equation governing steady-state propeller speed
def propeller_speed_coefficients(
        rho,
        Va,
        Vin,
        D,
        K_Q,
        K_V,
        R_motor,
        i0,
        CQ_0,
        CQ_1,
        CQ_2
):
    a = (rho * D**5 * CQ_0 / (2.0 * np.pi)**2)

    b = (rho * D**4 * CQ_1 * Va / (2.0 * np.pi) + K_Q * K_V / R_motor)

    c = (rho * D**3 * CQ_2 * Va**2 - K_Q * Vin / R_motor + K_Q * i0)

    return a, b, c

# calculate steady-state propeller angular speed [rad/s]
def propeller_speed(
        rho,
        Va,
        Vin,
        D,
        K_Q,
        K_V,
        R_motor,
        i0,
        CQ_0,
        CQ_1,
        CQ_2
):
    a, b, c = propeller_speed_coefficients(
        rho = rho,
        Va = Va,
        Vin = Vin,
        D = D,
        K_Q = K_Q,
        K_V = K_V,
        R_motor = R_motor,
        i0 = i0,
        CQ_0 = CQ_0,
        CQ_1 = CQ_1,
        CQ_2 = CQ_2
    )

    discriminant = b**2 - 4.0 * a * c

    if discriminant < 0.0:
        raise ValueError(
            "Motor/Propeller operating point has no real solution"
        )
    
    Omega_p = (-b + np.sqrt(discriminant)) / (2.0 * a)

    return Omega_p

# Calculate propeller thrust
def propeller_thrust(
        rho,
        Va,
        Omega_p,
        D,
        CT_0,
        CT_1,
        CT_2
):
    
    Tp = (
        rho * D**4 * CT_0 / (4.0 * np.pi**2) * Omega_p**2
        + rho * D**3 * CT_1 * Va / (2.0 * np.pi) * Omega_p
        + rho * D**2 * CT_2 * Va**2
    )

    return Tp

# calculate propeller torque magnitude 
def propeller_torque(
        rho,
        Va,
        Omega_p,
        D,
        CQ_0,
        CQ_1,
        CQ_2
):
    Qp = (
        rho * D**5 * CQ_0 / (4.0 * np.pi**2) * Omega_p**2
        + rho * D**4 * CQ_1 * Va / (2.0 * np.pi) * Omega_p
        + rho * D**3 * CQ_2 * Va**2
    )

    return Qp

# Calculate propeller thrust and torque from throttle command and airspeed using the steady-state motor/propeller model
def propulsion_thrust_torque(
        rho,
        Va,
        delta_t,
        V_max,
        D,
        K_Q,
        K_V,
        R_motor,
        i0,
        CT_0,
        CT_1,
        CT_2,
        CQ_0,
        CQ_1,
        CQ_2
):
    
    # Eqn. 4.22
    Vin = input_voltage(
        delta_t = delta_t,
        V_max = V_max
    )

    # Eqn. 4.21
    Omega_p = propeller_speed(
        rho = rho,
        Va = Va, 
        Vin = Vin,
        D = D,
        K_Q = K_Q,
        K_V = K_V,
        R_motor = R_motor,
        i0 = i0,
        CQ_0 = CQ_0,
        CQ_1 = CQ_1,
        CQ_2 = CQ_2
    )

    # Eqn 4.17
    Tp = propeller_thrust(
        rho = rho,
        Va = Va,
        Omega_p = Omega_p,
        D = D,
        CT_0 = CT_0,
        CT_1 = CT_1,
        CT_2 = CT_2
    )

    # Eqn 4.18
    Qp = propeller_torque(
        rho = rho,
        Va = Va,
        Omega_p = Omega_p,
        D = D,
        CQ_0 = CQ_0,
        CQ_1 = CQ_1,
        CQ_2 = CQ_2
    )

    return Tp, Qp

def advance_ratio(Va, Omega_p, D):

    if abs(Omega_p) <= 1e-8:
        raise ValueError(
            "Propeller speed is too small to calculate advance ratio"
        )
    
    return 2.0 * np.pi * Va / (Omega_p * D)

# do not need for calculations, but can be useful for plotting relationships (see book chp. 4)
def propeller_coefficients(
        J,
        CT_0,
        CT_1,
        CT_2,
        CQ_0,
        CQ_1,
        CQ_2
):
    
    CT = (CT_0 + CT_1 * J + CT_2 * J**2)

    CQ = (CQ_0 + CQ_1 * J + CQ_2 * J**2)

    return CT, CQ


# Propulsion wrapper (won't import Aerosonde parameters directly, well pass an aircraft object into wrapper MORE REUSABLE)
# calculate Aerosonde propulsion forces and moments in body axis

def aerosonde_propulsion_forces_moments(
        state,
        delta_t,
        aircraft
):
    altitude = -state.pd

    rho = density(altitude)

    # underscores communicate that propulsion only needs Va
    Va, _, _ = air_data(state)

    # motor/propeller model
    Tp, Qp = propulsion_thrust_torque(
        rho = rho,
        Va = Va,
        delta_t = delta_t,
        V_max = aircraft.V_max,
        D = aircraft.D_prop,
        K_Q = aircraft.K_Q,
        K_V = aircraft.K_V,
        R_motor = aircraft.R_motor,
        i0 = aircraft.i0,
        CT_0 = aircraft.CT_0,
        CT_1 = aircraft.CT_1,
        CT_2 = aircraft.CT_2,
        CQ_0 = aircraft.CQ_0,
        CQ_1 = aircraft.CQ_1,
        CQ_2 = aircraft.CQ_2
    )

    forces_body = np.array([
        Tp,
        0.0,
        0.0
    ])

    moments_body = np.array([
        -Qp,
        0.0,
        0.0
    ])

    return forces_body, moments_body






    





