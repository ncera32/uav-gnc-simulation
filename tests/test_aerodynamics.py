import numpy as np
import pytest

from src.aerodynamics import lift_drag, lift_drag_to_body, longitudinal_coefficients, pitching_moment, longitudinal_forces_moments, aerosonde_longitudinal_forces_moments, lateral_directional_coefficients, lateral_directional_forces_moments, aerosonde_lateral_directional_forces_moments, aerodynamic_forces_moments
from src.aircraft import AircraftParameters
from src.state import AircraftState
from src.atmosphere import density

def test_lift_drag_magnitudes():
    lift, drag = lift_drag(
        q_bar = 200.0,
        S = 0.5, 
        CL = 0.8,
        CD = 0.1
    )

    np.testing.assert_allclose(lift, 80.0)
    np.testing.assert_allclose(drag, 10.0)

def test_body_forces_zero_angle_of_attack():
    forces = lift_drag_to_body(
        lift = 80.0,
        drag = 10.0,
        alpha = 0.0
    )

    expected = np.array([
        -10.0,
        0.0,
        -80.0
    ])

    np.testing.assert_allclose(
        forces,
        expected,
        atol = 1e-12
    )
    
def test_body_forces_positive_angle_attack():
    alpha = np.deg2rad(30.0)

    forces = lift_drag_to_body(
        lift = 80.0,
        drag = 10.0,
        alpha = alpha
    )

    expected = np.array([
        -10.0 * np.cos(alpha) + 80.0 * np.sin(alpha),
        0.0,
        -10.0 * np.sin(alpha) - 80.0 * np.cos(alpha)
    ])

    np.testing.assert_allclose(
        forces,
        expected,
        atol = 1e-12
    )

def test_longitudinal_coefficients():
    coefficients = {
        "CL0": 0.2,
        "CL_alpha": 4.0,
        "CL_q": 2.0,
        "CL_delta_e": 0.5,
        "CD0": 0.03,
        "CD_alpha": 0.1,
        "CD_q": 0.0,
        "CD_delta_e": 0.02,
        "Cm0": 0.01,
        "Cm_alpha": -0.5,
        "Cm_q": -4.0,
        "Cm_delta_e": -1.0,
    }

    alpha = 0.1
    pitch_rate = 0.2
    delta_e = 0.05
    Va = 20.0
    c = 0.5

    CL, CD, Cm = longitudinal_coefficients(
        alpha = alpha,
        pitch_rate = pitch_rate, # pitch_rate is q (change???)
        delta_e = delta_e,
        Va = Va,
        c = c,
        coefficients = coefficients
    )

    # q_hat = 0.5 * 0.2 / (2 * 20) = 0.0025
    np.testing.assert_allclose(CL, 0.63)
    np.testing.assert_allclose(CD, 0.041)
    np.testing.assert_allclose(Cm, -0.10)

def test_pitching_moment():
    moment = pitching_moment(
        q_bar = 200.0,
        S = 0.5,
        c = 0.5,
        Cm = -0.1
    )

    np.testing.assert_allclose(moment, -5.0)
    
def test_longitudinal_forces_moments():
    coefficients = {
        "CL0": 0.5,
        "CL_alpha": 0.0,
        "CL_q": 0.0,
        "CL_delta_e": 0.0,
        "CD0": 0.1,
        "CD_alpha": 0.0,
        "CD_q": 0.0,
        "CD_delta_e": 0.0,
        "Cm0": -0.05,
        "Cm_alpha": 0.0,
        "Cm_q": 0.0,
        "Cm_delta_e": 0.0,
    }

    forces, moments = longitudinal_forces_moments(
        rho = 1.0,
        Va = 20.0,
        alpha = 0.0,
        pitch_rate = 0.0,
        delta_e = 0.0,
        S = 0.5,
        c = 0.4,
        coefficients = coefficients
    )

    # Dynamic pressure = 0.5 * 1.0 * 20^2 = 200 Pa
    # Lift = 200 * 0.5 * 0.5 = 50 N
    # Drag = 200 * 0.5 * 0.1 = 10 N
    # Pitching moment = 200 * 0.5 * 0.4 * (-0.05) = -2 N m

    expected_forces = np.array([
        -10.0,
        0.0,
        -50.0
    ])

    expected_moments = np.array([
        0.0,
        -2.0,
        0.0
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-12
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-12
    )

def test_aerosonde_longitudinal_forces_moments():
    aircraft = AircraftParameters

    state = AircraftState(
        pd = 0.0,
        u = 25.0,
        v = 0.0,
        w = 0.0,
        q = 0.0
    )

    delta_e = 0.0

    forces, moments = aerosonde_longitudinal_forces_moments(
        state = state, 
        delta_e = delta_e,
        aircraft = aircraft
    )

    # Independently calculate the expected results at alpha = 0, pitch_rate = 0, delta_e = 0.

    rho = density(0.0)
    q_bar = 0.5 * rho * 25.0**2

    expected_lift = q_bar * aircraft.S * aircraft.CL0

    expected_drag = q_bar * aircraft.S * aircraft.CD0

    expected_pitching_moment = q_bar * aircraft.S * aircraft.c * aircraft.Cm0

    expected_forces = np.array([
        -expected_drag,
        0.0,
        -expected_lift
    ])

    expected_moments = np.array([
        0.0, 
        expected_pitching_moment,
        0.0
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-10
    )

# confirm that aerosonde longitudinal function rejects nonzero sideslip
# def test_aerosonde_longitudinal_rejects_sideslip():
#    aircraft = AircraftParameters

#    state = AircraftState(
#        u = 25.0,
#        v = 1.0,
#        w = 0.0
#    )

#    with pytest.raises(ValueError):
#        aerosonde_longitudinal_forces_moments(
#            state = state,
#            delta_e = 0.0,
#            aircraft = aircraft
#        )

# ^^^ This is a temporary safeguard for the longitudinal-only implementation, not a restriction wanted in the eventual 6-DOF aircraft model
        
def test_lateral_directional_forces_moments():

    coefficients = {
        "CY0": 0.0,
        "CY_beta": -1.0,
        "CY_p": 0.0,
        "CY_r": 0.0,
        "CY_delta_a": 0.0,
        "CY_delta_r": 0.0,

        "Cl0": 0.0,
        "Cl_beta": -0.2,
        "Cl_p": 0.0,
        "Cl_r": 0.0,
        "Cl_delta_a": 0.0,
        "Cl_delta_r": 0.0,

        "Cn0": 0.0,
        "Cn_beta": 0.1,
        "Cn_p": 0.0,
        "Cn_r": 0.0,
        "Cn_delta_a": 0.0,
        "Cn_delta_r": 0.0
    }

    forces, moments = lateral_directional_forces_moments(
        rho = 1.0,
        Va = 20.0,
        beta = 0.1,
        roll_rate = 0.0,
        yaw_rate = 0.0,
        delta_a = 0.0,
        delta_r = 0.0,
        S = 0.5,
        b = 2.0,
        coefficients = coefficients
    )

    expected_forces = np.array([
        0.0,
        -10.0,
        0.0
    ])

    expected_moments = np.array([
        -4.0,
        0.0,
        2.0
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-10
    )

# test lateral-directional forces and moments function with nonzero yaw and roll rates!!
def test_lateral_directional_nonzero_yaw_roll_rates():

    coefficients = {
        "CY0": 0.0,
        "CY_beta": -1.0,
        "CY_p": 1.0,
        "CY_r": 1.0,
        "CY_delta_a": 0.0,
        "CY_delta_r": 0.0,

        "Cl0": 0.0,
        "Cl_beta": -0.2,
        "Cl_p": 1.0,
        "Cl_r": 1.0,
        "Cl_delta_a": 0.0,
        "Cl_delta_r": 0.0,

        "Cn0": 0.0,
        "Cn_beta": 0.1,
        "Cn_p": 1.0,
        "Cn_r": 1.0,
        "Cn_delta_a": 0.0,
        "Cn_delta_r": 0.0
    }

    forces, moments = lateral_directional_forces_moments(
        rho = 1.0,
        Va = 20.0,
        beta = 0.1,
        roll_rate = 1.0,
        yaw_rate = 1.0,
        delta_a = 0.0,
        delta_r = 0.0,
        S = 0.5,
        b = 2.0,
        coefficients = coefficients
    )

    expected_forces = np.array([
        0.0,
        0.0,
        0.0
    ])

    expected_moments = np.array([
        16.0,
        0.0,
        22.0
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-10
    )

# nonzero sideslip and no aileron and rudder deflection
def test_aerosonde_lateral_directional_forces_moments():

    aircraft = AircraftParameters()

    # moving forward with small lateral velocity
    state = AircraftState(
        pd = 0.0,
        u = 25.0,
        v = 1.0,
        w = 0.0,
        p = 0.0,
        r = 0.0
    )

    delta_a = 0.0
    delta_r = 0.0

    forces, moments = aerosonde_lateral_directional_forces_moments(
        state = state,
        delta_a = delta_a,
        delta_r = delta_r,
        aircraft = aircraft
    )

    # independently calculate expected air data
    Va = np.sqrt(25.0**2 + 1.0**2)

    beta = np.arcsin(1.0 / Va)

    # dynamic pressure at sea-level
    rho = density(0.0)
    q_bar = 0.5 * rho * Va**2

    # expected aero coefficients
    CY = aircraft.CY0 + aircraft.CY_beta * beta
    Cl = aircraft.Cl0 + aircraft.Cl_beta * beta
    Cn = aircraft.Cn0 + aircraft.Cn_beta * beta

    # expected dimensional forces and moments
    expected_Fy = q_bar * aircraft.S * CY
    expected_L = q_bar * aircraft.S * aircraft.b * Cl
    expected_N = q_bar * aircraft.S * aircraft.b * Cn

    expected_forces = np.array([
        0.0,
        expected_Fy,
        0.0
    ])

    expected_moments = np.array([
        expected_L,
        0.0,
        expected_N
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-10
    )

# nonzero control deflection and zero sideslip
def test_aersonde_lateral_directional_control_inputs():

    aircraft = AircraftParameters()

    state = AircraftState(
        pd = 0.0,
        u = 25.0,
        v = 0.0,
        w = 0.0,
        p = 0.0,
        r = 0.0
    )

    delta_a = 0.05 # rads
    delta_r = 0.03 # rads

    forces, moments = aerosonde_lateral_directional_forces_moments(
        state = state,
        delta_a = delta_a,
        delta_r = delta_r,
        aircraft = aircraft
    )

    rho = density(0.0)
    q_bar = 0.5 * rho * 25.0**2

    # expected coefficients from control deflections
    CY = (
        aircraft.CY0
        + aircraft.CY_delta_a * delta_a
        + aircraft.CY_delta_r * delta_r
    )

    Cl = (
        aircraft.Cl0
        + aircraft.Cl_delta_a * delta_a
        + aircraft.Cl_delta_r * delta_r
    )

    Cn = (
        aircraft.Cn0
        + aircraft.Cn_delta_a * delta_a
        + aircraft.Cn_delta_r * delta_r
    )

    expected_forces = np.array([
        0.0,
        q_bar * aircraft.S * CY,
        0.0
    ])

    expected_moments = np.array([
        q_bar * aircraft.S * aircraft.b * Cl,
        0.0,
        q_bar * aircraft.S * aircraft.b * Cn
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-10
    )

# test to verify wrappers roll/yaw rate mappings
def test_aerosonde_lateral_directional_yaw_roll():
    aircraft = AircraftParameters()

    state = AircraftState(
        pd = 0.0,
        u = 25.0,
        v = 0.0,
        w = 0.0,
        p = 2.0,
        r = 3.0
    )

    delta_a = 0.05 # rads
    delta_r = 0.03 # rads

    forces, moments = aerosonde_lateral_directional_forces_moments(
        state = state,
        delta_a = delta_a,
        delta_r = delta_r,
        aircraft = aircraft
    )

    rho = density(0.0)
    q_bar = 0.5 * rho * 25.0**2

    p_hat = aircraft.b * state.p / (2.0 * 25.0)
    r_hat = aircraft.b * state.r / (2.0 * 25.0)

    # expected coefficients from control deflections
    CY = (
        aircraft.CY0
        + aircraft.CY_p * p_hat
        + aircraft.CY_r * r_hat
        + aircraft.CY_delta_a * delta_a
        + aircraft.CY_delta_r * delta_r
    )

    Cl = (
        aircraft.Cl0
        + aircraft.Cl_p * p_hat
        + aircraft.Cl_r * r_hat
        + aircraft.Cl_delta_a * delta_a
        + aircraft.Cl_delta_r * delta_r
    )

    Cn = (
        aircraft.Cn0
        + aircraft.Cn_p * p_hat
        + aircraft.Cn_r * r_hat
        + aircraft.Cn_delta_a * delta_a
        + aircraft.Cn_delta_r * delta_r
    )

    expected_forces = np.array([
        0.0,
        q_bar * aircraft.S * CY,
        0.0
    ])

    expected_moments = np.array([
        q_bar * aircraft.S * aircraft.b * Cl,
        0.0,
        q_bar * aircraft.S * aircraft.b * Cn
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        atol = 1e-10
    )

# nonzero angle of attack, sideslip, deflections (elevator, aileron, rudder), zero angular rates
def test_combined_aero_forces_moments():

    aircraft = AircraftParameters()

    state = AircraftState(
        pd = 0.0,
        u = 25.0,
        v = 0.5,
        w = 1.0,
        p = 0.0,
        q = 0.0,
        r = 0.0
    )

    delta_e = 0.02
    delta_a = 0.03
    delta_r = -0.01

    forces, moments = aerodynamic_forces_moments(
        state = state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        aircraft = aircraft
    )

    # independently calc air data
    Va = np.sqrt(
        state.u**2
        + state.v**2
        + state.w**2
    )

    alpha = np.arctan2(state.w, state.u)

    beta = np.arcsin(state.v / Va)

    # atmospheric density and dynamic pressure
    rho = density(0.0)

    q_bar = 0.5 * rho * Va**2

    # ----------------------------------------------
    # Longitudinal Coefficients
    # ----------------------------------------------

    CL =  (
        aircraft.CL0
        + aircraft.CL_alpha * alpha
        + aircraft.CL_delta_e * delta_e
    )

    CD = (
        aircraft.CD0
        + aircraft.CD_alpha * alpha
        + aircraft.CD_delta_e * delta_e
    )

    Cm = (
        aircraft.Cm0
        + aircraft.Cm_alpha * alpha
        + aircraft.Cm_delta_e * delta_e
    )

    # dimensional lift and drag
    lift = q_bar * aircraft.S * CL
    drag = q_bar * aircraft.S * CD

    # Convert lift and drag to body-axis
    expected_Fx = (
        -drag * np.cos(alpha)
        + lift * np.sin(alpha)
    )

    expected_Fz = (
        -drag * np.sin(alpha)
        - lift * np.cos(alpha)
    )

    expected_M = q_bar * aircraft.S * aircraft.c * Cm

    # ---------------------------------------------------
    # Lateral-directional coefficients
    # ---------------------------------------------------

    CY = (
        aircraft.CY0
        + aircraft.CY_beta * beta
        + aircraft.CY_delta_a * delta_a
        + aircraft.CY_delta_r * delta_r
    )

    Cl = (
        aircraft.Cl0
        + aircraft.Cl_beta * beta
        + aircraft.Cl_delta_a * delta_a
        + aircraft.Cl_delta_r * delta_r
    )

    Cn = (
        aircraft.Cn0
        + aircraft.Cn_beta * beta
        + aircraft.Cn_delta_a * delta_a
        + aircraft.Cn_delta_r * delta_r
    )

    expected_Fy = q_bar * aircraft.S * CY
    expected_L = q_bar * aircraft.S * aircraft.b * Cl
    expected_N = q_bar * aircraft.S * aircraft.b * Cn

    # -----------------------------------------------------
    # Expected Combined Vectors
    # -----------------------------------------------------

    expected_forces = np.array([
        expected_Fx,
        expected_Fy,
        expected_Fz
    ])

    expected_moments = np.array([
        expected_L,
        expected_M,
        expected_N
    ])

    np.testing.assert_allclose(
        forces,
        expected_forces,
        rtol = 1e-10,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        moments,
        expected_moments,
        rtol = 1e-10,
        atol = 1e-10
    )

# ^^^^ this test checks simultaneously: 
# - correct airspeed, AoA, sideslip
# - correct elevator contribution to lift, drag, and pitching moment
# - correct aileron and rudder contributions
# - correct body-axis lift/drag conversion
# - all six aerodynamic components appear in the correct positions
    
# Because the angular rates are zero, their contributions disappear from the expected calculations. 
# previous tests have already verified the nondimensional angular-rate calculations
























