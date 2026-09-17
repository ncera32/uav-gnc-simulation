import numpy as np

from src.rotations import body_to_ned
from src.rotations import ned_to_body

# If the aircraft has zero roll, pitch, and yaw, the body-to-NED transformation shall be the identity matrix.
def test_zero_attitude():
    R = body_to_ned(0.0, 0.0, 0.0)

    np.testing.assert_allclose(
        R,
        np.eye(3),
        atol = 1e-12
    )

# Aircraft heading is pointed East, so body velocity [25, 0, 0] should return ned velocity [0, 25, 0]
def test_heading_east():
    R = body_to_ned(
        phi = 0.0,
        theta = 0.0, 
        psi = np.pi / 2
    )

    velocity_body = np.array([25.0, 0.0, 0.0])

    velocity_ned = R @ velocity_body

    expected = np.array([0.0, 25.0, 0.0])

    np.testing.assert_allclose(
        velocity_ned,
        expected,
        atol = 1e-12
    )

# Aircraft is pointed North but pitched upwards 10 degrees, purely forward velocity, so it should be moving north and gaining altitude
def test_positive_pitch_moves_upward():
    theta = np.deg2rad(10.0)

    R = body_to_ned(
        phi = 0.0,
        theta = theta,
        psi = 0.0
    )

    velocity_body = np.array([25.0, 0.0, 0.0])

    velocity_ned = R @ velocity_body

    # Positive North Velocity
    assert velocity_ned[0] > 0.0

    # Negative Down Velocity = Climbing 
    assert velocity_ned[2] < 0.0

# testing round trip transformations --> that the DCMs are inverses of eachother and when multiplied return Identity Matrix
def test_body_ned_round_trip():
    phi = np.deg2rad(10.0)
    theta = np.deg2rad(5.0)
    psi = np.deg2rad(45.0)

    R_bn = body_to_ned(phi, theta, psi)
    R_nb = ned_to_body(phi, theta, psi)

    vector_body = np.array([20.0, 3.0, -2.0])

    vector_ned = R_bn @ vector_body
    recovered_body = R_nb @ vector_ned

    np.testing.assert_allclose(
        recovered_body,
        vector_body,
        atol = 1e-12
    )

# Testing Orthogonality of DCMs
def test_rotation_matrix_orthogonal():
    R = body_to_ned(
        phi = np.deg2rad(15.0),
        theta = np.deg2rad(-8.0),
        psi = np.deg2rad(60.0)
    )

    np.testing.assert_allclose(
        R.T @ R,
        np.eye(3),
        atol = 1e-12
    )




