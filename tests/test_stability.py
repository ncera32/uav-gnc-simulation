import numpy as np
import pytest

from src.stability import calculate_eigenvalues, calculate_eigenmodes, scale_state_matrix, extract_state_subsystem, modal_characteristics, build_modal_report
from src.linearization import aircraft_state_jacobian
from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state

def test_calculate_eigenvalues_real():

    A = np.array([
        [-2.0, 0.0],
        [0.0, -5.0]
    ])

    eigenvalues = calculate_eigenvalues(A)

    expected = np.array([-5.0, -2.0])

    np.testing.assert_allclose(
        np.sort(eigenvalues),
        expected,
        atol = 1e-10
    )

def test_calculate_eigenvalues_complex():

    A = np.array([
        [-1.0, -4.0],
        [4.0, -1.0]
    ])

    eigenvalues = calculate_eigenvalues(A)

    expected = np.array([
        -1.0 - 4.0j,
        -1.0 + 4.0j
    ])

    np.testing.assert_allclose(
        np.sort_complex(eigenvalues),
        np.sort_complex(expected),
        atol = 1e-10
    )

# confirms aircraft Jacobian produces a valid set of 12 finite eigenvalues
def test_aircraft_eigenvalues():

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

    eigenvalues = calculate_eigenvalues(A)

    assert eigenvalues.shape == (12,)

    assert np.all(np.isfinite(eigenvalues))

# checks the defining eigenvalue equation
def test_calculate_eigenmodes():

    A = np.array([
        [-2.0, 0.0],
        [0.0, -5.0]
    ])

    eigenvalues, eigenvectors = calculate_eigenmodes(A)

    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)

    for i in range(len(eigenvalues)):

        eigenvalue = eigenvalues[i]
        eigenvector = eigenvectors[:, i]

        left_side = A @ eigenvector
        right_side = eigenvalue * eigenvector

        np.testing.assert_allclose(
            left_side,
            right_side,
            rtol = 1e-10,
            atol = 1e-10
        )

def test_state_scaling_preserves_eigenvalues():

    A = np.array([
        [-2.0, -3.0],
        [-1.0, -4.0]
    ])

    state_scales = np.array([
        2.0,
        0.1
    ])

    A_scaled = scale_state_matrix(
        A,
        state_scales
    )

    eigenvalues_original = np.linalg.eigvals(A)
    eigenvalues_scaled = np.linalg.eigvals(A_scaled)

    np.testing.assert_allclose(
        np.sort_complex(eigenvalues_original),
        np.sort_complex(eigenvalues_scaled),
        rtol = 1e-10,
        atol = 1e-10
    )


def test_extract_state_subsystem():

    A = np.arange(16).reshape(4, 4)

    indices = [1, 3]

    A_subsystem = extract_state_subsystem(
        A,
        indices
    )

    expected = np.array([
        [5, 7],
        [13, 15]
    ])

    np.testing.assert_array_equal(
        A_subsystem,
        expected
    )

def test_extract_state_subsystem_invalid_index():

    A = np.eye(4)

    with pytest.raises(ValueError):
        extract_state_subsystem(
            A,
            [1, 5]
        )

def test_dynamic_subsystem_extraction():

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
        Va=Va,
        altitude=altitude,
        alpha=alpha,
        phi=phi
    )

    A = aircraft_state_jacobian(
        state=trim_state,
        delta_e=delta_e,
        delta_a=delta_a,
        delta_r=delta_r,
        delta_t=delta_t,
        aircraft=aircraft
    )

    indices = [3, 4, 5, 6, 7, 8, 9, 10, 11]

    A_dynamic = extract_state_subsystem(
        A,
        indices
    )

    assert A_dynamic.shape == (9, 9)
    assert np.all(np.isfinite(A_dynamic))

    np.testing.assert_allclose(
        A_dynamic,
        A[3:12, 3:12]
    )


def test_modal_characteristics_complex_mode():

    eigenvalue = -2.0 + 3.0j

    characteristics = modal_characteristics(eigenvalue)

    assert np.isclose(
        characteristics["natural_frequency"],
        np.sqrt(13)
    )

    assert np.isclose(
        characteristics["damping_ratio"],
        2 / np.sqrt(13)
    )

    assert np.isclose(
        characteristics["period"],
        2 * np.pi / 3
    )

    assert characteristics["stable"]
    assert not characteristics["unstable"]
 

def test_build_modal_report():

    eigenvalues = np.array([
        -2.0 + 3.0j,
        -5.0 + 0.0j,
        0.1 + 0.0j
    ])

    names = [
        "Oscillatory",
        "Stable real",
        "Unstable real"
    ]

    report = build_modal_report(
        eigenvalues,
        names
    )

    assert len(report) == 3

    assert report[0]["name"] == "Oscillatory"
    assert report[0]["stable"]

    assert report[1]["name"] == "Stable real"
    assert report[1]["stable"]

    assert report[2]["name"] == "Unstable real"
    assert report[2]["unstable"]

    assert np.isclose(
        report[2]["doubling_time"],
        np.log(2) / 0.1
    )


def test_build_modal_report_mismatched_lengths():

    eigenvalues = np.array([-1.0, -2.0])

    names = ["Mode 1"]

    with pytest.raises(ValueError):
        build_modal_report(
            eigenvalues,
            names
        )

def test_modal_characteristics_stable_real_mode():

    characteristics = modal_characteristics(-20.340081)

    assert characteristics["stable"]
    assert not characteristics["unstable"]

    assert characteristics["period"] is None
    assert characteristics["doubling_time"] is None

    assert np.isclose(
        characteristics["time_constant"],
        1 / 20.340081
    )



