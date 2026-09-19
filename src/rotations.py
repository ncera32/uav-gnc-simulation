import numpy as np

def body_to_ned(phi, theta, psi):

    # return the DCM that transforms a vector from the UAV body frame to the NED frame

    # Parameters
    # ----------------------------------------------
    # phi : float
    #   roll angle [rad]
    # theta : float
    #   pitch angle [rad]
    # psi : float
    #   Yaw/heading angle [rad]

    # Returns
    # ----------------------------------------------
    # numpy.ndarray
    #       3x3 body-NED rotation matrix

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    c_theta = np.cos(theta)
    s_theta = np.sin(theta)

    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    return np.array([
        [
            c_theta * c_psi,
            s_phi * s_theta * c_psi - c_phi * s_psi,
            c_phi * s_theta * c_psi + s_phi * s_psi
        ],
        [
            c_theta * s_psi,
            s_phi * s_theta * s_psi + c_phi * c_psi,
            c_phi * s_theta * s_psi - s_phi * c_psi
        ],
        [
            -s_theta,
            s_phi * c_theta,
            c_phi * c_theta
        ]
    ])

# returns the DCM that transforms a vector in the NED frame to Body frame
def ned_to_body(phi, theta, psi):
    return body_to_ned(phi, theta, psi).T

def euler_rate_matrix(phi, theta):
    # Return the transformation matrix from body angular rates [p, q, r] to Euler angle rates [phi_dot, theta_dot, psi_dot]

    # Parameters
    # --------------------------------------------------
    # phi : float
    #    Roll angle [rad]
    # theta : float
    #    Pitch angle [rad]

    # Returns
    # --------------------------------------------------
    # numpy.ndarray
    #    3x3 Euler-rate transformation matrix.

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    c_theta = np.cos(theta)
    t_theta = np.tan(theta)

    return np.array([
        [1.0,
         s_phi * t_theta,
         c_phi * t_theta],
         [
            0.0,
            c_phi,
            -s_phi
         ],
         [
            0.0,
            s_phi / c_theta,
            c_phi / c_theta
         ]
    ])


def body_rates_to_euler_rates(phi, theta, p, q, r):
    # Convert body angular rates to Euler angle rates.

    # Parameters
    # -----------------------------------------------
    # phi : float
    #    Roll angle [rad]
    # theta : float
    #    Pitch angle [rad]
    # p : float
    #    Body roll rate [rad/s]
    # q : float
    #    Body pitch rate [rad/s]
    # r : float
    #    Body yaw rate [rad/s]

    # Returns
    # -----------------------------------------------
    # numpy.ndarray
    #    Euler angle rates [phi_dot, theta_dot, psi_dot] [rad/s].

    T = euler_rate_matrix(phi, theta)

    body_rates = np.array([
        p,
        q,
        r
    ])

    return T @ body_rates