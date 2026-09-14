import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nelder_mead import NelderMead

def quadratic(t):
    x, y = t
    return x**2 + y**2

def rosenbrock(t):
    x, y = t
    return (1 - x)**2 + 100 * (y - x**2)**2

def test_simplex_creation():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))

    expected = np.array([
        [1.0, 1.5],
        [1.05, 1.5],
        [1.0, 1.575]
    ])

    assert np.allclose(n.simplex, expected)

def test_order_vertices():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))

    n.order_vertices()

    expected_f_values = np.array([
        3.25,
        3.3525,
        3.480625
    ])

    assert np.allclose(n.f_values, expected_f_values)

    # values must be ordered best -> worst
    assert np.all(np.diff(n.f_values) >= 0)

def test_centroid():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))

    n.order_vertices()

    expected = np.array([1.025, 1.5])

    assert np.allclose(n.centroid, expected)


def test_reflect():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    v_R, f_R = n._reflect()

    expected_v_R = np.array([1.05, 1.425])
    expected_f_R = 3.133125

    assert np.allclose(v_R, expected_v_R)
    assert np.isclose(f_R, expected_f_R)


def test_expand():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    v_R, _ = n._reflect()
    v_E, f_E = n._expand(v_R)

    expected_v_E = np.array([1.075, 1.35])
    expected_f_E = 2.978125

    assert np.allclose(v_E, expected_v_E)
    assert np.isclose(f_E, expected_f_E)

def test_contract_inside():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    v_C, f_C = n._contract("ci")

    expected_v_C = np.array([1.0125, 1.5375])
    expected_f_C = 3.3890625

    assert np.allclose(v_C, expected_v_C)
    assert np.isclose(f_C, expected_f_C)

def test_contract_outside():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    v_C, f_C = n._contract("co")

    expected_v_C = np.array([1.0375, 1.4625])
    expected_f_C = 3.2153125

    assert np.allclose(v_C, expected_v_C)
    assert np.isclose(f_C, expected_f_C)

def test_invalid_contract_type():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    with pytest.raises(ValueError):
        n._contract("invalid")

def test_shrink():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    shrunk = n._shrink(n.simplex.copy())

    expected = np.array([
        [1.0, 1.5],
        [1.025, 1.5],
        [1.0, 1.5375]
    ])

    assert np.allclose(shrunk, expected)

def test_shrink_keeps_best_point_fixed():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))
    n.order_vertices()

    best_before = n.simplex[0].copy()

    shrunk = n._shrink(n.simplex.copy())

    assert np.allclose(shrunk[0], best_before)

def test_full_optimisation_quadratic():
    n = NelderMead(
        quadratic,
        np.array([1.0, 1.5]),
        tol=1e-6
    )

    result = n()

    best_point = result[0]
    best_value = quadratic(best_point)

    assert np.allclose(
        best_point,
        np.array([0.0, 0.0]),
        atol=1e-4
    )

    assert best_value < 1e-8

def test_full_optimisation_rosenbrock():
    def rosenbrock(t):
        x, y = t
        return (1 - x)**2 + 100 * (y - x**2)**2

    n = NelderMead(
        rosenbrock,
        np.array([-1.2, 1.0]),
        tol=1e-6
    )

    result = n()

    best_point = result[0]
    best_value = rosenbrock(best_point)

    assert np.allclose(
        best_point,
        np.array([1.0, 1.0]),
        atol=1e-3
    )

    assert best_value < 1e-6

def test_final_simplex_is_ordered():
    n = NelderMead(quadratic, np.array([1.0, 1.5]))

    n()

    assert np.all(np.diff(n.f_values) >= 0)

    assert np.isclose(
        quadratic(n.simplex[0]),
        n.f_values[0]
    )

def test_alpha_validation():
    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([1.0, 1.5]),
            alpha=0
        )

def test_gamma_validation():
    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([1.0, 1.5]),
            gamma=1
        )

def test_beta_validation():
    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([1.0, 1.5]),
            beta=0
        )

    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([1.0, 1.5]),
            beta=1
        )

def test_sigma_validation():
    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([1.0, 1.5]),
            sigma=0
        )

    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([1.0, 1.5]),
            sigma=1
        )

def test_all_zero_initial_point_rejected():
    with pytest.raises(ValueError):
        NelderMead(
            quadratic,
            np.array([0.0, 0.0])
        )