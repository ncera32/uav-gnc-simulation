import numpy as np

from src.state import AircraftState

def test_state_array_round_trip():
    original = AircraftState(
        pn = 100.0,
        pe = 200.0,
        pd = -300.0,
        u = 25.0,
        v = 1.0,
        w = -2.0,
        phi = 0.1,
        theta = 0.2,
        psi = 0.3,
        p = 0.01,
        q = 0.02,
        r = 0.03
    )

    x = original.to_array()

    assert x.shape == (12,)

    restored = AircraftState.from_array(x)

    np.testing.assert_allclose(
        restored.to_array(),
        original.to_array(),
        atol = 1e-12
    )

def test_array_rejects_wrong_size():
    x = np.zeros(11)

    try:
        AircraftState.from_array(x)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for an invalid state array"
        )
