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

