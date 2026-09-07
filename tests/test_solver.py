import unittest
import sys 
from pathlib import Path 
import numpy as np

# add parent path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.solver import RK4

class SolverTests(unittest.TestCase):

    ###### attached motion tests


    def test_invalid_dt(self):
        with self.assertRaises(ValueError):
            RK4(dt=0)

        with self.assertRaises(ValueError):
            RK4(dt=-0.01)

    def test_equilibrium_remains_stationary(self):
        """
            theta = 0 and omega = 0 should remain stationary.
        """
        solver = RK4(dt=0.01)

        times, states = solver.simulate(
            theta0=0.0,
            omega0=0.0,
            L=15.0,
            T=2.0
        )

        np.testing.assert_allclose(
            states,
            0.0,
            atol=1e-12
        )

    def test_output_shapes_match(self):
        solver = RK4(dt=0.01)

        times, states = solver.simulate(
            theta0=np.radians(30),
            omega0=0.0,
            L=15.0,
            T=1.0
        )

        self.assertEqual(len(times), len(states))
        self.assertEqual(states.shape[1], 2)

    def test_initial_state_is_preserved(self):
        solver = RK4(dt=0.01)

        theta0 = np.radians(-45)
        omega0 = 0.5

        times, states = solver.simulate(
            theta0=theta0,
            omega0=omega0,
            L=15.0,
            T=1.0
        )

        self.assertAlmostEqual(states[0, 0], theta0)
        self.assertAlmostEqual(states[0, 1], omega0)

    def test_positive_angle_initial_acceleration_is_negative(self):
        """
            Starting at positive theta with omega=0 should initially
            accelerate back toward theta=0.
        """
        solver = RK4(dt=0.01)

        _, states = solver.simulate(
            theta0=np.radians(30),
            omega0=0.0,
            L=15.0,
            T=0.01
        )

        # after the first step, omega should be negative
        self.assertLess(states[1, 1], 0)

    def test_negative_angle_initial_acceleration_is_positive(self):
        """
            Starting at negative theta with omega=0 should initially
            accelerate back toward theta=0.
        """
        solver = RK4(dt=0.01)

        _, states = solver.simulate(
            theta0=np.radians(-30),
            omega0=0.0,
            L=15.0,
            T=0.01
        )

        self.assertGreater(states[1, 1], 0)


    ####### projectile motion tests
    def test_released_output_shapes_match(self):
        solver = RK4(dt=0.01)

        times, states = solver.simulate_released(
            x0=0.0,
            y0=100.0,
            vx0=5.0,
            vy0=0.0,
            T=1.0
        )

        self.assertEqual(
            len(times),
            len(states)
        )

        self.assertEqual(
            states.shape[1],
            4
        )

    def test_released_initial_state_is_preserved(self):
        solver = RK4(dt=0.01)

        x0 = 2.0
        y0 = 20.0
        vx0 = 5.0
        vy0 = 3.0

        _, states = solver.simulate_released(
            x0=x0,
            y0=y0,
            vx0=vx0,
            vy0=vy0,
            T=1.0
        )

        np.testing.assert_allclose(
            states[0],
            np.array([
                x0,
                y0,
                vx0,
                vy0
            ]),
            atol=1e-12
        )

    def test_horizontal_velocity_remains_constant_without_drag(self):
        """
            With no drag, dvx/dt = 0.
        """
        solver = RK4(dt=0.01)

        vx0 = 7.5

        _, states = solver.simulate_released(
            x0=0.0,
            y0=100.0,
            vx0=vx0,
            vy0=5.0,
            T=1.0
        )

        vx_values = states[:, 2]

        np.testing.assert_allclose(
            vx_values,
            vx0,
            atol=1e-10
        )

    def test_vertical_velocity_decreases_due_to_gravity(self):
        """
            With no drag, vertical velocity should decrease
            approximately linearly with time.
        """
        solver = RK4(dt=0.01)

        times, states = solver.simulate_released(
            x0=0.0,
            y0=100.0,
            vx0=0.0,
            vy0=10.0,
            T=1.0
        )

        expected_vy = (
            10.0 - 9.81 * times
        )

        np.testing.assert_allclose(
            states[:, 3],
            expected_vy,
            atol=1e-10
        )

    def test_horizontal_position_changes_linearly_without_drag(self):
        """
            Since vx is constant without drag:
                x = x0 + vx0 * t
        """
        solver = RK4(dt=0.01)

        x0 = 2.0
        vx0 = 4.0

        times, states = solver.simulate_released(
            x0=x0,
            y0=100.0,
            vx0=vx0,
            vy0=0.0,
            T=1.0
        )

        expected_x = (
            x0 + vx0 * times
        )

        np.testing.assert_allclose(
            states[:, 0],
            expected_x,
            atol=1e-10
        )

    def test_released_simulation_stops_at_ground(self):
        """
            Projectile simulation should stop after crossing
            y_ground while travelling downward.
        """
        solver = RK4(dt=0.01)

        times, states = solver.simulate_released(
            x0=0.0,
            y0=1.0,
            vx0=0.0,
            vy0=0.0,
            T=10.0,
            y_ground=0.0
        )

        # It should stop well before full T
        self.assertLess(
            times[-1],
            10.0
        )

        # Final state should be at or below ground
        self.assertLessEqual(
            states[-1, 1],
            0.0
        )

        # And moving downward
        self.assertLess(
            states[-1, 3],
            0.0
        )

    def test_released_returns_full_duration_if_ground_not_reached(self):
        """
            If the projectile does not reach the ground within T,
            the whole requested simulation should be returned.
        """
        solver = RK4(dt=0.01)

        times, states = solver.simulate_released(
            x0=0.0,
            y0=1000.0,
            vx0=0.0,
            vy0=0.0,
            T=1.0,
            y_ground=0.0
        )

        self.assertAlmostEqual(
            times[-1],
            1.0
        )

        self.assertEqual(
            len(times),
            len(states)
        )


if __name__ == '__main__':
    unittest.main()