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






