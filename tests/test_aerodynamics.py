import numpy as np
import pytest

from src.aerodynamics import lift_drag, lift_drag_to_body, longitudinal_coefficients, pitching_moment, longitudinal_forces_moments, aerosonde_longitudinal_forces_moments, lateral_directional_coefficients, lateral_directional_forces_moments
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
def test_aerosonde_longitudinal_rejects_sideslip():
    aircraft = AircraftParameters

    state = AircraftState(
        u = 25.0,
        v = 1.0,
        w = 0.0
    )

    with pytest.raises(ValueError):
        aerosonde_longitudinal_forces_moments(
            state = state,
            delta_e = 0.0,
            aircraft = aircraft
        )

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

