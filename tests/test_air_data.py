import numpy as np
import pytest

from src.state import AircraftState
from src.air_data import air_data, dynamic_pressure

def test_straight_flight():
    state = AircraftState(
        u = 25.0,
        v = 0.0,
        w = 0.0
    )

    Va, alpha, beta = air_data(state)

    np.testing.assert_allclose(Va, 25.0)
    np.testing.assert_allclose(alpha, 0.0)
    np.testing.assert_allclose(beta, 0.0)

def test_positive_angle_of_attack():
    state = AircraftState(
        u = 20.0,
        v = 0.0,
        w = 2.0
    )

    Va, alpha, beta = air_data(state)

    np.testing.assert_allclose(
        Va,
        np.sqrt(404.0)
    )

    np.testing.assert_allclose(
        alpha,
        np.arctan2(2.0, 20.0)
    )

    np.testing.assert_allclose(beta, 0.0)

def test_positive_sideslip():
    state = AircraftState(
        u = 20.0,
        v = 5.0,
        w = 0.0
    )

    Va, alpha, beta = air_data(state)

    np.testing.assert_allclose(
        Va, 
        np.sqrt(425.0)
    )

    np.testing.assert_allclose(alpha, 0.0)

    np.testing.assert_allclose(
        beta,
        np.arcsin(5.0 / np.sqrt(425.0))
    )

def test_zero_airspeed_error():
    state = AircraftState()

    with pytest.raises(ValueError):
        air_data(state)

def test_dynamic_pressure():
    rho = 1.225
    Va = 20.0

    q_bar = dynamic_pressure(rho, Va)

    np.testing.assert_allclose(q_bar, 245.0)

def test_dynamic_pressure_zero_airspeed():
    np.testing.assert_allclose(
        dynamic_pressure(1.225, 0.0),
        0.0
    )