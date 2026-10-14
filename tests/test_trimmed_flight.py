import numpy as np

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.integrator import integrate_aircraft_model

def test_trimmed_flight_remains_steady():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    trim_result = solve_full_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft 
    )

    assert trim_result.success

    alpha, phi, delta_e, delta_a, delta_r, delta_t = (
        trim_result.x
    )

    initial_state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    t_eval = np.linspace(0.0, 5.0, 101)

    solution = integrate_aircraft_model(
        initial_state = initial_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft,
        t_span = (0.0, 5.0),
        t_eval = t_eval
    )

    assert solution.success

    y = solution.y

    # check airspeed

    airspeed = np.sqrt(
        y[3]**2 + y[4]**2 + y[5]**2
    )

    np.testing.assert_allclose(
        airspeed,
        Va,
        atol = 1e-3
    )

    # check altitude
    np.testing.assert_allclose(
        -y[2],
        altitude,
        atol = 1e-3
    )

    expected_attitude = np.repeat(
        initial_state.to_array()[6:9, None],
        len(solution.t),
        axis = 1
    )

    # check attitude
    np.testing.assert_allclose(
        y[6:9],
        expected_attitude,
        atol = 1e-4
    )

    # check angular rates
    np.testing.assert_allclose(
        y[9:12],
        0.0,
        atol = 1e-4
    )