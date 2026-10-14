import numpy as np

from src.aircraft import AircraftParameters
from src.state import AircraftState
from src.trim import solve_full_trim, full_trim_state
from src.integrator import integrate_aircraft_model

def test_pitch_rate_disturbance():

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

    trim_state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    disturbed_array = trim_state.to_array().copy()
    disturbed_array[10] += 0.05

    disturbed_state = AircraftState.from_array(
        disturbed_array
    )

    # verify initial perturbation
    np.testing.assert_allclose(
        disturbed_state.q - trim_state.q,
        0.05
    )

    # verify all other state remain unchanged
    other_indices = [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 11
    ]

    np.testing.assert_allclose(
        disturbed_state.to_array()[other_indices],
        trim_state.to_array()[other_indices]
    )

    # short simulation
    t_eval = np.linspace(0.0, 2.0, 41)

    solution = integrate_aircraft_model(
        initial_state = disturbed_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft,
        t_span = (0.0, 2.0),
        t_eval = t_eval
    )

    assert solution.success

    assert np.all(np.isfinite(solution.y))

    # pitch rate should evolve from the disturbed value
    assert not np.isclose(
        solution.y[10, -1],
        disturbed_state.q
    )

    









