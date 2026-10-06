import numpy as np

from src.rotations import body_to_ned, ned_to_body, body_rates_to_euler_rates
from src.forces_moments import aircraft_forces_moments

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

# will have to define an inertia tensor using Aerosonde parameters for inertia (I = inertia_tensor(aircraft.Ixx, ...))

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

def state_derivative(
        state,
        forces_body,
        moments_body,
        mass,
        inertia
):
    # 1. Translational and rotational kinematics
    position_dot, euler_dot = kinematics(state)

    # 2. Calculate gravitational force in body frame
    gravity_body = gravity_force_body(
        state,
        mass
    )

    # 3. Combine non-gravitational forces with gravity
    total_forces_body = (
        np.asarray(forces_body, dtype = float) + gravity_body
    )

    # 4. Translational dynamics
    velocity_dot = translational_dynamics(
        state,
        total_forces_body,
        mass
    )

    # 5. Rotational dynamics
    angular_rates_dot = rotational_dynamics(
        state,
        moments_body,
        inertia
    )

    # 6. Assemble the complete state derivative 
    x_dot = np.concatenate([
        position_dot,
        velocity_dot,
        euler_dot,
        angular_rates_dot
    ])

    return x_dot

# Calculate the aircraft state derivative using state-dependent aerodynamic and propulsion forces.
# Control inputs are held constant for this evaluation.
def aircraft_state_derivative(
       state,
       delta_e,
       delta_a,
       delta_r,
       delta_t,
       aircraft 
):
    
    # Calc. aero. + prop. forces + moments from CURRENT aircraft state
    forces_body, moments_body = aircraft_forces_moments(
        state = state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    # Construct inertia tensor using the existing function
    inertia = inertia_tensor(
        aircraft.Ixx,
        aircraft.Iyy,
        aircraft.Izz,
        aircraft.Ixz
    )

    # Feed forces/moments into the already-tested rigid-body EOMs
    return state_derivative(
        state = state,
        forces_body = forces_body,
        moments_body = moments_body,
        mass = aircraft.m,
        inertia = inertia
    )
