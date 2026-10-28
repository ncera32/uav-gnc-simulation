import numpy as np

# Calculate the eigenvalues of a state matrix
def calculate_eigenvalues(A):

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("State matrix must be square.")
    
    if not np.all(np.isfinite(A)):
        raise ValueError("State matrix must contain finite values.")
    
    eigenvalues = np.linalg.eigvals(A)

    return eigenvalues


# Calculate eigenvalues and right eigenvectors of a state matrix.
# eigenvectors[:, i] corresponds to eigenvalue[i]
def calculate_eigenmodes(A):

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("State matrix must be square.")
    
    if not np.all(np.isfinite(A)):
        raise ValueError("State matrix must contain finite values.")
    
    eigenvalues, eigenvectors = np.linalg.eig(A)

    return eigenvalues, eigenvectors

# Transform a state matrix into scaled coordinates (x = S z, A_scaled = S^-1 A S)
def scale_state_matrix(A, state_scales):

    A = np.asarray(A, dtype=float)

    state_scales = np.asarray(
        state_scales,
        dtype=float
    )

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("State matrix must be square.")
    
    if state_scales.shape != (A.shape[0],):
        raise ValueError(
            "State scales must match matrix dimensions."
        )
    
    if not np.all(np.isfinite(state_scales)) or np.any(state_scales <= 0):
        raise ValueError(
            "State scales must be finite and positive."
        )
    
    return (
        A * state_scales[np.newaxis, :]
    ) / state_scales[:, np.newaxis]

# Extract a square state subsystem from a full state matrix.
def extract_state_subsystem(A, state_indices):

    A = np.asarray(A, dtype=float)
    state_indices = np.asarray(state_indices)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("State matrix must be square.")
    
    if state_indices.ndim != 1:
        raise ValueError("State indices must be one-dimensional.")
    
    if not np.issubdtype(state_indices.dtype, np.integer):
        raise ValueError("State indices must be integers.")
    
    if (
        np.any(state_indices < 0)
        or np.any(state_indices >= A.shape[0])
    ):
        raise ValueError("State indices are out of bounds.")
    
    if len(np.unique(state_indices)) != len(state_indices):
        raise ValueError("State indices must be unique.")
    
    return A[np.ix_(state_indices, state_indices)]

# calculate characteristics of a single eigenvalue
def modal_characteristics(eigenvalue):

    sigma = float(np.real(eigenvalue))
    omega_d = abs(float(np.imag(eigenvalue)))

    omega_n = np.hypot(sigma, omega_d)

    damping_ratio = (
        -sigma / omega_n if omega_n > 0 else np.nan
    )

    period = (
        2.0 * np.pi / omega_d
        if omega_d > 1e-10 else None
    )

    time_constant = (
        1 / abs(sigma)
        if abs(sigma) > 1e-10 else None
    )

    doubling_time = (
        np.log(2) / sigma
        if sigma > 1e-10 else None
    )

    return {
        "natural_frequency": omega_n,
        "damped_frequency": omega_d,
        "damping_ratio": damping_ratio,
        "period": period,
        "time_constant": time_constant,
        "doubling_time": doubling_time,
        "stable": sigma < -1e-10,
        "unstable": sigma > 1e-10
    }

# ^^^^^^^^^^^^^^^^^^^
# (1) For complex-conjugate pairs, the two eigenvalues have identical modal characteristics. You only need to report one representative from each pair.
# (2) For real eigenvalues, oscillation period is undefined, so we return None.
# (3) A zero eigenvalue is marginal rather than asymptotically stable or unstable.
# (4) For a positive real eigenvalue, doubling time describes exponential growth.
# (5) The damping ratio returned for a real eigenvalue is mathematically defined by the formula, but isn't as useful for physical interpretation as its time constant.



# Build a modal-characteristic report (doesn't automatically identify modes, assumes evals already correctly associated with physical modes)
def build_modal_report(eigenvalues, mode_names):

    eigenvalues = np.asarray(eigenvalues)

    if len(eigenvalues) != len(mode_names):
        raise ValueError(
            "Number of eigenvalues must match mode names."
        )
    
    report = []

    for name, eigenvalue in zip(mode_names, eigenvalues):

        characteristics = modal_characteristics(eigenvalue)

        mode_data = {
            "name": name,
            "eigenvalue": eigenvalue,
            **characteristics
        }

        report.append(mode_data)

    return report 
    







