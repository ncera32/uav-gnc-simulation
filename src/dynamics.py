import numpy as np

from src.rotations import body_to_ned, ned_to_body, body_rates_to_euler_rates

def kinematics(state):
    # Calculate translational and rotational kinematics.

    # Returns
    # -----------------------------
    # position_dot : numpy.ndarray
    #    NED position rates [m/s]

    # euler_dot : numpy.ndarray
    #    Euler angle rates [rad/s]

    velocity_body = np.array([
        state.u,
        state.v,
        state.w
    ])

    R_bn = body_to_ned(
        state.phi,
        state.theta,
        state.psi
    )

    position_dot = R_bn @ velocity_body

    euler_dot = body_rates_to_euler_rates(
        state.phi,
        state.theta,
        state.p,
        state.q,
        state.r
    )

    return position_dot, euler_dot

def translational_dynamics(state, forces_body, mass):
    #Calculate body-frame translational acceleration.

    # Parameters
    # -----------------------------------------------
    # state : AircraftState
    #    Current aircraft state.

    # forces_body : numpy.ndarray
    #    Total body-frame forces [Fx, Fy, Fz] [N].

    # mass : float
    #    Aircraft mass [kg].

    # Returns
    # -----------------------------------------------
    # numpy.ndarray
    #    Body-frame velocity derivatives
    #    [u_dot, v_dot, w_dot] [m/s^2]

    velocity_body = np.array([
        state.u,
        state.v,
        state.w
    ])

    angular_velocity_body = np.array([
        state.p,
        state.q,
        state.r
    ])

    acceleration_body = (
        forces_body / mass - np.cross(angular_velocity_body, velocity_body)
    )

    return acceleration_body

# gravity force in body frame function
def gravity_force_body(state, mass, g = 9.80665):
    gravity_ned = np.array([
        0.0,
        0.0,
        mass * g
    ])

    R_nb = ned_to_body(
        state.phi,
        state.theta,
        state.psi
    )

    return R_nb @ gravity_ned

# function for inertia tensor
def inertia_tensor(Ixx, Iyy, Izz, Ixz):
    return np.array([
        [Ixx, 0.0, -Ixz],
        [0.0, Iyy, 0.0],
        [-Ixz, 0.0, Izz]
    ])

# defining the rotational dynamics eqn
def rotational_dynamics(state, moments_body, inertia):
    
    omega = np.array([
        state.p,
        state.q,
        state.r
    ])

    angular_momentum = inertia @ omega

    gyroscopic_term = np.cross(
        omega,
        angular_momentum
    )

    omega_dot = np.linalg.solve(
        inertia,
        moments_body - gyroscopic_term
    )

    return omega_dot