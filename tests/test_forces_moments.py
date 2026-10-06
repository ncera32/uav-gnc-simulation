import numpy as np

from src.aircraft import AircraftParameters
from src.state import AircraftState

from src.forces_moments import aircraft_forces_moments

from src.aerodynamics import aerosonde_longitudinal_forces_moments, aerosonde_lateral_directional_forces_moments

from src.propulsion import aerosonde_propulsion_forces_moments

def test_aircraft_forces_moments():

    aircraft = AircraftParameters()

    state = AircraftState(
        pd = -500.0,
        u = 25.0,
        v = 1.0,
        w = 2.0,
        p = 0.1,
        q = 0.2,
        r = -0.1
    )

    delta_e = -0.03
    delta_a = 0.02
    delta_r = -0.01
    delta_t = 0.6

    forces, moments = aircraft_forces_moments(
        state = state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    forces_long, moments_long = (
        aerosonde_longitudinal_forces_moments(
            state = state,
            delta_e = delta_e,
            aircraft = aircraft
        )
    )

    forces_lat, moments_lat = (
        aerosonde_lateral_directional_forces_moments(
            state = state,
            delta_a = delta_a,
            delta_r = delta_r,
            aircraft = aircraft
        )
    )

    forces_prop, moments_prop = (
        aerosonde_propulsion_forces_moments(
            state = state,
            delta_t = delta_t,
            aircraft = aircraft
        )
    )

    expected_forces = (
        forces_long
        + forces_lat
        + forces_prop
    )

    expected_moments = (
        moments_long
        + moments_lat
        + moments_prop
    )

    expected_X = forces_long[0] + forces_prop[0]
    expected_Y = forces_lat[1]
    expected_Z = forces_long[2]

    expected_L = moments_lat[0] + moments_prop[0]
    expected_M = moments_long[1]
    expected_N = moments_lat[2]

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

    np.testing.assert_allclose(
        forces,
        np.array([
            expected_X,
            expected_Y,
            expected_Z
        ])
    )

    np.testing.assert_allclose(
        moments,
        np.array([
            expected_L,
            expected_M,
            expected_N
        ])
    )


