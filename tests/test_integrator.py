import numpy as np

from src.state import AircraftState
from src.aircraft import AircraftParameters
from src.integrator import integrate_aircraft, integrate_aircraft_model

# testing integrator in free-fall
def test_free_fall():
    g = 9.80665

    initial_state = AircraftState(
        pd = -100.0
    )

    solution = integrate_aircraft(
        initial_state = initial_state,
        forces_body = np.zeros(3),
        moments_body = np.zeros(3),
        mass = 10.0,
        inertia = np.diag([2.0, 3.0, 4.0]),
        t_span = (0.0, 1.0),
        t_eval = np.array([0.0, 1.0])
    )

    final_state = AircraftState.from_array(
        solution.y[:, -1]
    )

    expected_pd = -100.0 + 0.5 * g
    expected_w = g 

    np.testing.assert_allclose(
        final_state.pd,
        expected_pd,
        atol = 1e-6
    )

    np.testing.assert_allclose(
        final_state.w,
        expected_w,
        atol = 1e-6
    )

# testing integrator with constant-speed, force-balanced motion
def test_integrator_constant_speed_level_motion():
    mass = 10.0
    g = 9.80665

    initial_state = AircraftState(
        pd = -100.0,
        u = 25.0,
    )

    solution = integrate_aircraft(
        initial_state = initial_state,
        forces_body = np.array([0.0, 0.0, -mass * g]),
        moments_body = np.zeros(3),
        mass = mass,
        inertia = np.diag([2.0, 3.0, 4.0]),
        t_span = (0.0, 10.0),
        t_eval = np.linspace(0.0, 10.0, 101)
    )

    final_state = AircraftState.from_array(
        solution.y[:, -1]
    )

    np.testing.assert_allclose(
        final_state.pn,
        250.0,
        atol = 1e-6
    )

    np.testing.assert_allclose(
        final_state.pd,
        -100.0,
        atol = 1e-6
    )

    np.testing.assert_allclose(
        final_state.u,
        25.0,
        atol = 1e-6
    )

def test_integrator_constant_speed_eastward_motion():
    mass = 10.0
    g = 9.80665

    initial_state = AircraftState(
        pd = -100.0,
        u = 25.0,
        psi = np.pi / 2
    )

    solution = integrate_aircraft(
        initial_state = initial_state,
        forces_body = np.array([0.0, 0.0, -mass * g]),
        moments_body = np.zeros(3),
        mass = mass,
        inertia = np.diag([2.0, 3.0, 4.0]),
        t_span = (0.0, 10.0),
        t_eval = np.array([0.0, 10.0])
    )

    final_state = AircraftState.from_array(
        solution.y[:, -1]
    )

    np.testing.assert_allclose(
        final_state.pn,
        0.0,
        atol = 1e-6
    )

    np.testing.assert_allclose(
        final_state.pe,
        250.0,
        atol = 1e-6
    )

    np.testing.assert_allclose(
        final_state.psi,
        np.pi / 2,
        atol = 1e-10
    )

# Integration/interface test, not physical validation test (can the nonlinear model run through solve_ivp())
def test_integrate_aircraft_model_runs():

    aircraft = AircraftParameters()

    initial_state = AircraftState(
        pn = 0.0,
        pe = 0.0,
        pd = -100.0,

        u = 25.0,
        v = 0.0,
        w = 0.5,

        phi = 0.0,
        theta = 0.0,
        psi = 0.0,

        p = 0.0,
        q = 0.0,
        r = 0.0
    )

    t_eval = np.linspace(
        0.0,
        0.1,
        11
    )

    solution = integrate_aircraft_model(
        initial_state = initial_state,
        delta_e = 0.0,
        delta_a = 0.0,
        delta_r = 0.0,
        delta_t = 0.5,
        aircraft = aircraft,
        t_span = (0.0, 0.1),
        t_eval = t_eval
    )

    assert solution.success

    assert np.all(
        np.isfinite(solution.y)
    )

    assert solution.y.shape == (
        12,
        len(t_eval)
    )