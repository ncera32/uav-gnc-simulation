import numpy as np

from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state
from src.linearization import aircraft_state_jacobian
from src.stability import calculate_eigenvalues, extract_state_subsystem

def test_aircraft_stability_subsystems():

    aircraft = AircraftParameters()

    Va = 25.0
    altitude = 500.0

    trim_result = solve_full_trim(
        Va = Va,
        altitude = altitude,
        aircraft = aircraft
    )

    assert trim_result.success

    alpha, phi, delta_e, delta_a, delta_r, delta_t = (
        trim_result.x
    )

    trim_state = full_trim_state(
        Va = Va,
        altitude = altitude,
        alpha = alpha, 
        phi = phi
    )

    A = aircraft_state_jacobian(
        state = trim_state,
        delta_e = delta_e,
        delta_a = delta_a,
        delta_r = delta_r,
        delta_t = delta_t,
        aircraft = aircraft
    )

    longitudinal_indices = [3, 5, 10, 7]
    lateral_indices = [4, 9, 11, 6]

    A_longitudinal = extract_state_subsystem(
        A,
        longitudinal_indices
    )

    A_lateral = extract_state_subsystem(
        A,
        lateral_indices
    )

    # Verify matrix dimensions
    assert A_longitudinal.shape == (4, 4)
    assert A_lateral.shape == (4, 4)

    # Verify matrix entries
    np.testing.assert_allclose(
        A_longitudinal,
        A[np.ix_(
            longitudinal_indices,
            longitudinal_indices
        )]
    )

    np.testing.assert_allclose(
        A_lateral,
        A[np.ix_(
            lateral_indices,
            lateral_indices
        )]
    )

    # calculate eigenvalues
    eigenvalues_longitudinal = calculate_eigenvalues(
        A_longitudinal
    )

    eigenvalues_lateral = calculate_eigenvalues(
        A_lateral
    )

    # Verify eigenvalue dimensions and finiteness
    assert eigenvalues_longitudinal.shape == (4,)
    assert eigenvalues_lateral.shape == (4,)

    assert np.all(np.isfinite(eigenvalues_longitudinal))
    assert np.all(np.isfinite(eigenvalues_lateral))

