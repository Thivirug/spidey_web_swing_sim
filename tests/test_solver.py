import unittest
import sys 
from pathlib import Path 
import numpy as np

# add parent path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solver import RK4

class SolverTests(unittest.TestCase):
    def test_invalid_T(self):
        with self.assertRaises(ValueError):
            RK4(T=0, dt=0.01)

        with self.assertRaises(ValueError):
            RK4(T=-1, dt=0.01)

    def test_invalid_dt(self):
        with self.assertRaises(ValueError):
            RK4(T=5, dt=0)

        with self.assertRaises(ValueError):
            RK4(T=5, dt=-0.01)

    def test_dt_greater_than_T(self):
        with self.assertRaises(ValueError):
            RK4(T=1, dt=2)

    def test_equilibrium_remains_stationary(self):
        """
            theta = 0 and omega = 0 should remain stationary.
        """
        solver = RK4(T=2.0, dt=0.01)

        times, states = solver.simulate(
            theta0=0.0,
            omega0=0.0,
            L=15.0
        )

        np.testing.assert_allclose(
            states,
            0.0,
            atol=1e-12
        )

    def test_output_shapes_match(self):
        solver = RK4(T=1.0, dt=0.01)

        times, states = solver.simulate(
            theta0=np.radians(30),
            omega0=0.0,
            L=15.0
        )

        self.assertEqual(len(times), len(states))
        self.assertEqual(states.shape[1], 2)

    def test_initial_state_is_preserved(self):
        solver = RK4(T=1.0, dt=0.01)

        theta0 = np.radians(-45)
        omega0 = 0.5

        times, states = solver.simulate(
            theta0=theta0,
            omega0=omega0,
            L=15.0
        )

        self.assertAlmostEqual(states[0, 0], theta0)
        self.assertAlmostEqual(states[0, 1], omega0)

    def test_positive_angle_initial_acceleration_is_negative(self):
        """
            Starting at positive theta with omega=0 should initially
            accelerate back toward theta=0.
        """
        solver = RK4(T=0.01, dt=0.01)

        _, states = solver.simulate(
            theta0=np.radians(30),
            omega0=0.0,
            L=15.0
        )

        # after the first step, omega should be negative
        self.assertLess(states[1, 1], 0)

    def test_negative_angle_initial_acceleration_is_positive(self):
        """
            Starting at negative theta with omega=0 should initially
            accelerate back toward theta=0.
        """
        solver = RK4(T=0.01, dt=0.01)

        _, states = solver.simulate(
            theta0=np.radians(-30),
            omega0=0.0,
            L=15.0
        )

        self.assertGreater(states[1, 1], 0)

if __name__ == '__main__':
    unittest.main()