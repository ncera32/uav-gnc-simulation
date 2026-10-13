import numpy as np

from src.trim import longitudinal_trim_state, longitudinal_trim_residual, solve_longitudinal_trim, full_trim_state, full_trim_residual, solve_full_trim
from src.dynamics import aircraft_state_derivative
from src.aircraft import AircraftParameters
from src.rotations import body_to_ned

def test_longitudinal_trim_state():

    Va = 25.0
    altitude = 500.0
    alpha = np.deg2rad(5.0)

    state = longitudinal_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha
    )

    expected_u = Va * np.cos(alpha)
    expected_w = Va * np.sin(alpha)

    resulting_Va = np.sqrt(
        state.u**2
        + state.v**2
        + state.w**2
    )

    np.testing.assert_allclose(
        state.u,
        expected_u
    )

    np.testing.assert_allclose(
        state.w,
        expected_w
    )

    np.testing.assert_allclose(
        state.v,
        0.0
    )

    np.testing.assert_allclose(
        state.pd,
        -altitude
    )

    np.testing.assert_allclose(
        state.theta,
        alpha
    )

    np.testing.assert_allclose(
        [state.p, state.q, state.r],
        [0.0, 0.0, 0.0]
    )

    np.testing.assert_allclose(
        resulting_Va,
        Va,
        rtol = 1e-12,
        atol = 1e-12
    )

def test_longitudinal_trim_residual():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    alpha = np.deg2rad(5.0)
    delta_e = np.deg2rad(-2.0)
    delta_t = 0.5

    trim_variables = np.array([
        alpha,
        delta_e,
        delta_t
    ])

    residual = longitudinal_trim_residual(
        trim_variables = trim_variables,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    state = longitudinal_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha
    )

    derivative = aircraft_state_derivative(
        state = state,
        delta_e = delta_e,
        delta_a = 0.0,
        delta_r = 0.0,
        delta_t = delta_t,
        aircraft = aircraft
    )

    expected_residual = np.array([
        derivative[3],
        derivative[5],
        derivative[10]
    ])

    np.testing.assert_allclose(
        residual,
        expected_residual,
        rtol = 1e-10,
        atol = 1e-10
    )

    assert np.all(np.isfinite(residual))


def test_solve_longitudinal_trim():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    result = solve_longitudinal_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    assert result.success

    alpha, delta_e, delta_t = result.x

    assert np.all(np.isfinite(result.x))

    # verify physical search bounds
    assert np.deg2rad(-10.0) <= alpha <= np.deg2rad(15.0)
    assert np.deg2rad(-25.0) <= delta_e <= np.deg2rad(25.0)
    assert 0.0 <= delta_t <= 1.0

    # re-evaluate actual trim residual 
    residual = longitudinal_trim_residual(
        trim_variables = result.x,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    np.testing.assert_allclose(
        residual,
        np.zeros(3),
        atol = 1e-4
    )
    
def test_longitudinal_trim_full_state_derivative():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    result = solve_longitudinal_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    assert result.success

    alpha, delta_e, delta_t = result.x

    state = longitudinal_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha
    )

    derivative = aircraft_state_derivative(
        state = state,
        delta_e = delta_e,
        delta_a = 0.0,
        delta_r = 0.0,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # verify level flight kinematics
    np.testing.assert_allclose(
        derivative[0:3],
        [Va, 0.0, 0.0],
        atol = 1e-8
    )

    # verify longitudinal equilibrium
    np.testing.assert_allclose(
        derivative[[3, 5, 10]],
        np.zeros(3),
        atol = 1e-4
    )

    # verify no instantaneous attitude-rate motion
    np.testing.assert_allclose(
        derivative[6:9],
        np.zeros(3),
        atol = 1e-8
    )

    # remaining lat/dir accelerations are diagnostic outputs (not yet constrained to zero)
    assert np.all(
        np.isfinite(derivative[[4, 9, 11]])
    )


def test_full_trim_state():

    Va = 25.0
    altitude = 500.0

    alpha = np.deg2rad(5.0)
    phi = np.deg2rad(3.0)

    state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    # check airspeed 
    resulting_Va = np.sqrt(
        state.u ** 2
        + state.v ** 2 
        + state.w ** 2
    )

    np.testing.assert_allclose(
        resulting_Va,
        Va,
        atol = 1e-12
    )

    # check altitude
    np.testing.assert_allclose(
        state.pd,
        -altitude
    )

    # check zero sideslip
    np.testing.assert_allclose(
        state.v,
        0.0
    )

    # check bank angle 
    np.testing.assert_allclose(
        state.phi,
        phi
    )

    # check zero angular velocity
    np.testing.assert_allclose(
        [state.p, state.q, state.r],
        [0.0, 0.0, 0.0]
    )

    velocity_body = np.array([
        state.u, 
        state.v,
        state.w
    ])

    velocity_ned = body_to_ned(
        state.phi,
        state.theta,
        state.psi
    ) @ velocity_body

    np.testing.assert_allclose(
        velocity_ned[2],
        0.0,
        atol = 1e-12
    )

def test_full_trim_state_zero_bank():

    Va = 25.0
    altitude = 500.0
    alpha = np.deg2rad(5.0)

    state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = 0.0
    )

    np.testing.assert_allclose(
        state.theta,
        alpha,
        atol = 1e-12
    )

def test_full_trim_residual():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    alpha = np.deg2rad(5.0)
    phi = np.deg2rad(3.0)

    delta_e = np.deg2rad(-5.0)
    delta_a = np.deg2rad(2.0)
    delta_r = np.deg2rad(1.0)
    delta_t = 0.5

    trim_variables = np.array([
        alpha,
        phi,
        delta_e,
        delta_a,
        delta_r,
        delta_t
    ])

    # evaluate full trim residual
    residual = full_trim_residual(
        trim_variables = trim_variables,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    # independently construct the same candidate state
    state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha,
        phi = phi
    )

    derivative = aircraft_state_derivative(
        state = state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    expected_residual = derivative[
        [3, 4, 5, 9, 10, 11]
    ]
    
    np.testing.assert_allclose(
        residual,
        expected_residual,
        rtol = 1e-10,
        atol = 1e-10
    )

    assert residual.shape == (6,)

    assert np.all(np.isfinite(residual))


def test_full_trim_matches_longitudinal_trim():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    alpha = np.deg2rad(5.0)
    delta_e = np.deg2rad(-2.0)
    delta_t = 0.5

    longitudinal_variables = np.array([
        alpha,
        delta_e,
        delta_t
    ])

    full_variables = np.array([
        alpha,
        0.0,
        delta_e,
        0.0,
        0.0,
        delta_t
    ])

    longitudinal_residual = longitudinal_trim_residual(
        trim_variables = longitudinal_variables,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    full_residual = full_trim_residual(
        trim_variables = full_variables,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    np.testing.assert_allclose(
        full_residual[[0, 2, 4]],
        longitudinal_residual,
        rtol = 1e-10,
        atol = 1e-10
    )

# Test verifies:
# - SciPy terminated successfully. 
# - The solution lies within the permitted search region.
# - All six body-axis accelerations are effectively zero.
def test_solve_full_trim():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    result = solve_full_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    assert result.success

    alpha, phi, delta_e, delta_a, delta_r, delta_t = result.x

    # check all variables are finite
    assert np.all(np.isfinite(result.x))

    # check bounds
    assert np.deg2rad(-10.0) <= alpha <= np.deg2rad(15.0)
    assert np.deg2rad(-15.0) <= phi <= np.deg2rad(15.0)
    assert np.deg2rad(-25.0) <= delta_e <= np.deg2rad(25.0)
    assert np.deg2rad(-25.0) <= delta_a <= np.deg2rad(25.0)
    assert np.deg2rad(-25.0) <= delta_r <= np.deg2rad(25.0)
    assert 0.0 <= delta_t <= 1.0

    # evaluate full equilibrium residual
    residual = full_trim_residual(
        trim_variables = result.x,
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    np.testing.assert_allclose(
        residual,
        np.zeros(6),
        atol = 1e-4
    )

# verifies that our complete aircraft model agrees with the trim solution
def test_full_trim_state_derivative():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    result = solve_full_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    assert result.success

    alpha, phi, delta_e, delta_a, delta_r, delta_t = result.x

    state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha, 
        phi = phi
    )

    derivative = aircraft_state_derivative(
        state = state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # constant altitude
    np.testing.assert_allclose(
        derivative[2],
        0.0,
        atol = 1e-8
    )

    # zero body-axis translational accelerations
    np.testing.assert_allclose(
        derivative[3:6],
        np.zeros(3),
        atol = 1e-4
    )

    # zero euler-angle rates
    np.testing.assert_allclose(
        derivative[6:9],
        np.zeros(3),
        atol = 1e-8
    )

    # zero angular accelerations
    np.testing.assert_allclose(
        derivative[9:12],
        np.zeros(3),
        atol = 1e-4
    )











