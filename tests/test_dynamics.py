import numpy as np

from src.state import AircraftState
from src.dynamics import kinematics, translational_dynamics, gravity_force_body, inertia_tensor, rotational_dynamics

# An aircraft pointing North, flying forward at 25 m/s, with no angular velocity should move North at 25 m/s while maintaining its current attitude.
def test_level_northward_kinematics():
    state = AircraftState(
        u = 25.0
    )

    position_dot, euler_dot = kinematics(state)

    expected_position_dot = np.array([
        25.0,
        0.0,
        0.0
    ])

    expected_euler_dot = np.array([
        0.0,
        0.0,
        0.0
    ])

    np.testing.assert_allclose(
        position_dot,
        expected_position_dot,
        atol = 1e-12
    )

    np.testing.assert_allclose(
        euler_dot,
        expected_euler_dot,
        atol = 1e-12
    )

def test_translational_dynamics_force_only():
    state = AircraftState()

    forces_body = np.array([
        20.0,
        0.0,
        0.0
    ])

    acceleration = translational_dynamics(
        state,
        forces_body,
        mass = 10.0
    )

    expected = np.array([
        2.0,
        0.0,
        0.0
    ])

    np.testing.assert_allclose(
        acceleration,
        expected,
        atol = 1e-12
    )

def test_translational_dynamics_rotating_frame():
    state = AircraftState(
        u = 20.0,
        r = 0.1
    )

    forces_body = np.zeros(3)

    acceleration = translational_dynamics(
        state,
        forces_body,
        mass = 10.0
    )

    expected = np.array([
        0.0,
        -2.0,
        0.0
    ])

    np.testing.assert_allclose(
        acceleration,
        expected,
        atol = 1e-12
    )

def test_translational_dynamics_combined():
    state = AircraftState(
        u = 20.0,
        r = 0.1
    )

    forces_body = np.array([
        20.0,
        10.0,
        0.0
    ])

    acceleration = translational_dynamics(
        state,
        forces_body,
        mass = 10.0
    )

    expected = np.array([
        2.0,
        -1.0,
        0.0
    ])

    np.testing.assert_allclose(
        acceleration,
        expected,
        atol = 1e-12
    )

# testing gravity with level forces (level attitude)
def test_gravity_force_level():
    state = AircraftState()
    
    mass = 10.0
    g = 9.80665

    force = gravity_force_body(
        state,
        mass,
        g
    )

    expected = np.array([
        0.0,
        0.0,
        mass * g
    ])

    np.testing.assert_allclose(
        force,
        expected,
        atol = 1e-12
    )


# testing gravity function at a pitched attitude
def test_gravity_positive_pitch():
    theta = np.deg2rad(30.0)

    state = AircraftState(
        theta = theta
    )

    mass = 10.0
    g = 9.80665

    force = gravity_force_body(
        state,
        mass,
        g
    )

    expected = np.array([
        -mass * g * np.sin(theta),
        0.0,
        mass * g * np.cos(theta)
    ])

    np.testing.assert_allclose(
        force,
        expected,
        atol = 1e-12
    )

def test_inertia_tensor():
    I = inertia_tensor(
        Ixx = 2.0,
        Iyy = 3.0,
        Izz = 4.0,
        Ixz = 0.5
    )

    expected = np.array([
        [2.0, 0.0, -0.5],
        [0.0, 3.0, 0.0],
        [-0.5, 0, 4.0]
    ])

    np.testing.assert_allclose(
        I,
        expected,
        atol = 1e-12
    )

# no rotation test for rotational dynamics
def test_rotational_dynamics_from_moments():
    state = AircraftState()

    inertia = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0]
    ])

    moments = np.array([
        4.0,
        6.0,
        8.0
    ])

    angular_acceleration = rotational_dynamics(
        state,
        moments,
        inertia
    )

    expected = np.array([
        2.0,
        2.0,
        2.0
    ])

    np.testing.assert_allclose(
        angular_acceleration,
        expected,
        atol = 1e-12
    )

def test_rotational_dynamics_zero():
    state = AircraftState()

    inertia = np.diag([
        2.0,
        3.0,
        4.0
    ])

    moments = np.zeros(3)

    angular_acceleration = rotational_dynamics(
        state, 
        moments,
        inertia
    )

    np.testing.assert_allclose(
        angular_acceleration,
        np.zeros(3),
        atol = 1e-12
    )
    
# testing rotational dynamics while UAV is rotating (coupling effect)
def test_rotational_dynamics_gyroscopic_coupling():
    state = AircraftState(
        p = 1.0,
        q = 2.0,
        r = 3.0
    )

    inertia = np.diag([
        2.0,
        3.0,
        4.0
    ])

    moments = np.zeros(3)

    angular_acceleration = rotational_dynamics(
        state,
        moments,
        inertia
    )

    expected = np.array([
        -3.0,
        2.0,
        -0.5
    ])

    np.testing.assert_allclose(
        angular_acceleration,
        expected,
        atol = 1e-12
    )

# testing effect of Ixz couple / roll and yaw coupling on rotational dynamics
def test_rotational_dynamics_coupling():
    state = AircraftState()

    inertia = inertia_tensor(
        Ixx = 2.0,
        Iyy = 3.0,
        Izz = 4.0,
        Ixz = 0.5
    )

    moments = np.array([
        1.0,
        0.0,
        0.0
    ])

    angular_acceleration = rotational_dynamics(
        state,
        moments,
        inertia
    )

    expected = np.linalg.solve(
        inertia,
        moments
    )

    np.testing.assert_allclose(
        angular_acceleration,
        expected,
        atol = 1e-12
    )