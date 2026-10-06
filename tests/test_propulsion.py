import numpy as np
import pytest

from src.propulsion import input_voltage, propeller_speed_coefficients, propeller_speed, propeller_thrust, propeller_torque, propulsion_thrust_torque, aerosonde_propulsion_forces_moments
from src.aircraft import AircraftParameters
from src.state import AircraftState
from src.atmosphere import density

def test_input_voltage():
    Vin = input_voltage(
        delta_t = 0.5,
        V_max = 44.4
    )

    np.testing.assert_allclose(
        Vin,
        22.2,
        atol = 1e-12
    )

def test_input_voltage_limits():

    assert input_voltage(0.0, 44.4) == 0.0
    assert input_voltage(1.0, 44.4) == 44.4

def test_input_voltage_invalid_throttle():

    with pytest.raises(ValueError):
        input_voltage(
            delta_t = 1.1,
            V_max = 44.4
        )

def test_propeller_speed():

    rho = 1.225
    Va = 20.0
    Vin = 22.2

    D = 0.508

    K_Q = 0.0659
    K_V = 0.0659
    R_motor = 0.042
    i0 = 1.5

    CQ_0 = 0.005230
    CQ_1 = 0.004970
    CQ_2 = -0.01664

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

    # independent expected calculations
    a = rho * D**5 * CQ_0 /(2.0 * np.pi)**2

    b = (rho * D**4 * CQ_1 * Va / (2.0 * np.pi) + K_Q * K_V / R_motor)

    c = (rho * D**3 * CQ_2 * Va**2 - K_Q * Vin / R_motor + K_Q * i0)

    expected_Omega_p = (-b + np.sqrt(b**2 - 4.0 * a *c)) / (2.0 * a)

    np.testing.assert_allclose(
        Omega_p,
        expected_Omega_p,
        rtol = 1e-10,
        atol = 1e-10
    )

def test_propeller_thrust_torque():

    rho = 1.225
    Va = 20.0
    Omega_p = 400.0
    D = 0.508

    CT_0 = 0.09357
    CT_1 = -0.06044
    CT_2 = -0.1079

    CQ_0 = 0.005230
    CQ_1 = 0.004970
    CQ_2 = -0.01664

    Tp = propeller_thrust(
        rho = rho,
        Va = Va,
        Omega_p = Omega_p,
        D = D,
        CT_0 = CT_0,
        CT_1 = CT_1,
        CT_2 = CT_2
    )

    Qp = propeller_torque(
        rho = rho,
        Va = Va,
        Omega_p = Omega_p,
        D = D,
        CQ_0 = CQ_0,
        CQ_1 = CQ_1,
        CQ_2 = CQ_2
    )

    expected_Tp = (
        rho * D**4 * CT_0 / (4.0 * np.pi**2) * Omega_p**2
        + rho * D**3 * CT_1 * Va / (2.0 * np.pi) * Omega_p
        + rho * D**2 * CT_2 * Va**2
    )

    expected_Qp = Qp = (
        rho * D**5 * CQ_0 / (4.0 * np.pi**2) * Omega_p**2
        + rho * D**4 * CQ_1 * Va / (2.0 * np.pi) * Omega_p
        + rho * D**3 * CQ_2 * Va**2
    )

    np.testing.assert_allclose(
        Tp,
        expected_Tp,
        rtol = 1e-10,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        Qp,
        expected_Qp,
        rtol = 1e-10,
        atol = 1e-10
    )

def test_propulsion_thrust_torque():

    rho = 1.225
    Va = 20.0
    delta_t = 0.5

    V_max = 44.4
    D = 0.508

    K_Q = 0.0659
    K_V = 0.0659
    R_motor = 0.042
    i0 = 1.5

    CT_0 = 0.09357
    CT_1 = -0.06044
    CT_2 = -0.1079

    CQ_0 = 0.005230
    CQ_1 = 0.004970
    CQ_2 = -0.01664

    Tp, Qp = propulsion_thrust_torque(
        rho = rho,
        Va = Va,
        delta_t = delta_t,
        V_max = V_max,
        D = D,
        K_Q = K_Q,
        K_V = K_V,
        R_motor = R_motor,
        i0 = i0,
        CT_0 = CT_0,
        CT_1 = CT_1,
        CT_2 = CT_2,
        CQ_0 = CQ_0,
        CQ_1 = CQ_1,
        CQ_2 = CQ_2
    )

    # -----------------------------------------
    # Independent Expected Result
    # -----------------------------------------

    Vin = V_max * delta_t

    a = (rho * D**5 * CQ_0 / (2.0 * np.pi)**2)

    b = (rho * D**4 * CQ_1 * Va / (2.0 * np.pi) + K_Q * K_V / R_motor)

    c = (rho * D**3 * CQ_2 * Va**2 - K_Q * Vin / R_motor + K_Q * i0)

    Omega_p = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)

    expected_Tp = (
        rho * D**4 * CT_0 / (4.0 * np.pi**2) * Omega_p**2
        + rho * D**3 * CT_1 * Va / (2.0 * np.pi) * Omega_p
        + rho * D**2 * CT_2 * Va**2
    )

    expected_Qp = (
        rho * D**5 * CQ_0 / (4.0 * np.pi**2) * Omega_p**2
        + rho * D**4 * CQ_1 * Va / (2.0 * np.pi) * Omega_p
        + rho * D**3 * CQ_2 * Va**2
    )

    np.testing.assert_allclose(
        Tp,
        expected_Tp,
        rtol = 1e-10,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        Qp,
        expected_Qp,
        rtol = 1e-10,
        atol = 1e-10
    )


# Is the wrapper correctly mapping the aircraft state and aircraft parameters into my already-tested propulsion model?

def test_aerosonde_propulsion_wrapper():

    aircraft = AircraftParameters()

    state = AircraftState(
        pd = 0.0,
        u = 20.0,
        v = 0.0,
        w = 0.0
    )

    delta_t = 0.5

    forces, moments = aerosonde_propulsion_forces_moments(
        state = state,
        delta_t = delta_t,
        aircraft = aircraft
    )

    rho = density(0.0)
    Va = 20.0

    expected_Tp, expected_Qp = propulsion_thrust_torque(
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

    expected_forces = np.array([
        expected_Tp,
        0.0,
        0.0
    ])

    expected_moments = np.array([
        -expected_Qp,
        0.0,
        0.0
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


def test_aerosonde_propulsion_state_mapping():

    aircraft = AircraftParameters()

    state = AircraftState(
        pd = -1000.0,
        u = 20.0,
        v = 3.0,
        w = 4.0
    )

    delta_t = 0.6

    forces, moments = aerosonde_propulsion_forces_moments(
        state = state,
        delta_t = delta_t,
        aircraft = aircraft
    )

    altitude = 1000.0
    rho = density(altitude)

    Va = np.sqrt(
        20.0**2
        + 3.0**2
        + 4.0**2
    )

    expected_Tp, expected_Qp = propulsion_thrust_torque(
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

    expected_forces = np.array([
        expected_Tp,
        0.0,
        0.0
    ])

    expected_moments = np.array([
        -expected_Qp,
        0.0,
        0.0
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













