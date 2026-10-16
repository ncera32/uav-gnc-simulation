import numpy as np

from src.linearization import numerical_jacobian, aircraft_state_jacobian
from src.aircraft import AircraftParameters
from src.trim import solve_full_trim, full_trim_state

def test_numerical_jacobian():

    def example_function(x):

        return np.array([
            x[0]**2 + 3.0*x[1],
            np.sin(x[0]) + x[1]**2
        ])
    
    x = np.array([2.0, 1.0])

    step_sizes = np.array([
        1e-5,
        1e-5
    ])

    jacobian = numerical_jacobian(
        function = example_function,
        x = x,
        step_sizes = step_sizes
    )

    expected_jacobian = np.array([
        [4.0, 3.0],
        [np.cos(2.0), 2.0]
    ])

    np.testing.assert_allclose(
        jacobian,
        expected_jacobian,
        rtol = 1e-6,
        atol = 1e-8
    )

# confirms Jacobian can be evaluated at trim condition and produce a 12x12 matrix 
def test_aircraft_state_jacobian():

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

    assert A.shape == (12, 12)

    assert np.all(np.isfinite(A))

    np.testing.assert_allclose(
        A[:, 0:2],
        0.0,
        atol = 1e-10
    )

    np.testing.assert_allclose(
        A[0, 3],
        np.cos(trim_state.theta),
        atol = 1e-6
    )

def test_aircrat_jacobian_step_size_sensitivity():

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

    base_steps = np.array([
        1e-3, 1e-3, 1e-3,
        1e-4, 1e-4, 1e-4,
        1e-5, 1e-5, 1e-5,
        1e-5, 1e-5, 1e-5
    ])

    def calculate_A(steps):
        return aircraft_state_jacobian(
            state = trim_state,
            delta_e = delta_e,
            delta_a = delta_a,
            delta_r = delta_r,
            delta_t = delta_t,
            aircraft = aircraft,
            step_sizes = steps
        )
    
    A_small = calculate_A(0.5 * base_steps)
    A_base = calculate_A(base_steps)
    A_large = calculate_A(2.0 * base_steps)

    np.testing.assert_allclose(
        A_small,
        A_base,
        rtol = 1e-4,
        atol = 1e-6
    )

    np.testing.assert_allclose(
        A_large,
        A_base,
        rtol = 1e-4,
        atol = 1e-6
    )

    # format arrays for easier inspection
    np.set_printoptions(
        precision = 6,
        suppress = True,
        linewidth = 160
    )

    print("\n--- State Jacobian: Small Step Size (0.5h) ---")
    print(A_small)

    print("\n--- State Jacobian: Base Step Size (h) ---")
    print(A_base)

    print("\n--- State Jacobian: Large Step Size (2h) ---")
    print(A_large)

    # Calculate elementwise differences
    differences_small = A_small - A_base
    differences_large = A_large - A_base

    print("\n--- Difference: Small vs Base ---")
    #print(differences_small)
    print(
        np.array2string(
            differences_small,
            formatter = {"float_kind": lambda x: f"{x:.3e}"}
        )
    )

    print("\n--- Difference: Large vs Base ---")
    #print(differences_large)
    print(
        np.array2string(
            differences_large,
            formatter = {"float_kind": lambda x: f"{x:.3e}"}
        )
    )

    # Quantify the larget difference
    print("\n--- Maximum Abosulte Differences ---")
    print(
        f"Small vs Base: "
        f"{np.max(np.abs(differences_small)):.6e}"
    )

    print(
        f"Large vs Base: "
        f"{np.max(np.abs(differences_large)):.6e}"
    )








