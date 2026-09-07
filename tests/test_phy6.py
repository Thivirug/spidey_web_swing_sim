import unittest
import sys
from pathlib import Path
import math
import numpy as np

# add parent dir to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics import t_derivatives, energy, g, t_derivatives_released

class TDerivtests(unittest.TestCase):
    def test_simple(self):
        theta = 0
        omega = 0.0
        L = 15.0

        state = np.array([theta, omega])
        result = t_derivatives(state, L)

        self.assertEqual(result[0], omega)
        self.assertEqual(result[1], 0)

    def test_horiz_release(self):
        # At theta = pi/2, sin(theta) = 1, domega = -g / L
        L = 2.0

        state = np.array([math.pi / 2, 0.0])
        result = t_derivatives(state, L=L)

        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], -g/L)

    def test_inverted_spidey(self):
        # At top equilibrium (theta = pi), domega = 0
        state = np.array([math.pi, 1.5])
        result = t_derivatives(state, L=1.0)

        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(result[1], 0.0)

    def test_negative_angle_restoring_force(self):
        # Negative angle should accelerate positively back toward equilibrium
        L = 0.5
        state = np.array([-math.pi / 2, -0.5])
        result = t_derivatives(state, L=L)
        
        self.assertAlmostEqual(result[0], -0.5)
        self.assertAlmostEqual(result[1], g / L)

    def test_nonzero_velocity_and_length(self):
        # General case check with arbitrarily chosen parameters
        theta, omega, L = 0.5, 2.5, 3.0

        state = np.array([theta, omega])
        expected_dom = -(g / L) * math.sin(theta)

        result = t_derivatives(state, L)

        self.assertAlmostEqual(result[0], omega)
        self.assertAlmostEqual(result[1], expected_dom)

class TestEnergy(unittest.TestCase):

    def test_energy_at_rest_bottom(self):
        # theta = 0 and omega = 0 -> KE = 0, PE = 0
        m = 2.0
        L = 3.0
        state = np.array([0.0, 0.0])

        res = energy(m, L, state)
        self.assertAlmostEqual(res, 0.0)

    def test_kinetic_only(self):
        # theta =0 -> PE=0
        m = 2.0
        L = 3.0
        omega = 4.0
        state = np.array([0.0, omega])

        expected = 0.5*m*np.pow(L*omega,2)
        res = energy(m, L, state)

        self.assertAlmostEqual(res, expected)

    def test_energy_potential_only(self):
        # omega = 0 -> PE only
        m = 2.0
        L = 3.0
        theta = math.pi / 2
        state = np.array([theta, 0.0])

        expected = m*g*L * (1 - math.cos(theta))
        result = energy(m, L, state)

        self.assertAlmostEqual(result, expected)

    def test_energy_at_top(self):
        # theta = pi -> maximum PE = 2mgL
        m = 1.5
        L = 2.0
        state = np.array([math.pi, 0.0])

        expected = 2*m*g*L
        result = energy(m, L, state)

        self.assertAlmostEqual(result, expected)

    def test_invalid_params(self):
        state = np.array([0.0, 0.0])

        with self.assertRaises(ValueError):
            energy(0, 2.0, state)

        with self.assertRaises(ValueError):
            energy(1.0, 0, state)

class TestRelDerivs(unittest.TestCase):
    def test_no_drag_stationary(self):
        # at rest pos should not change 
        # v_x=0, v_y=-g

        state = np.array([10.3, 20.45, 0.0, 0.0])

        result = t_derivatives_released(state)

        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], -g)

    def test_no_drag_horizontal_motion(self):
        # Horizontal velocity should remain constant without drag.
        state = np.array([0.0, 10.0, 5.0, 0.0])

        result = t_derivatives_released(state)

        self.assertAlmostEqual(result[0], 5.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], -g)

    def test_no_drag_general_velocity(self):
        # Position derivatives should equal velocity components.
        state = np.array([2.0, 8.0, 3.5, -4.0])

        result = t_derivatives_released(state)

        self.assertAlmostEqual(result[0], 3.5)
        self.assertAlmostEqual(result[1], -4.0)
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], -g)

    def test_drag_horizontal_motion(self):
        m = 80.0
        rho = 1.225
        Cd = 1.0
        A = 0.7

        vx = 4.0
        vy = 0.0

        state = np.array([0.0, 10.0, vx, vy])

        speed = 4.0
        drag_factor = 0.5 * rho * Cd * A

        expected_dvx = -(drag_factor / m) * speed * vx
        expected_dvy = -g

        result = t_derivatives_released(
            state,
            drag_params=(m, rho, Cd, A)
        )

        self.assertAlmostEqual(result[0], vx)
        self.assertAlmostEqual(result[1], vy)
        self.assertAlmostEqual(result[2], expected_dvx)
        self.assertAlmostEqual(result[3], expected_dvy)

    def test_drag_general_velocity(self):
        m = 80.0
        rho = 1.225
        Cd = 1.0
        A = 0.7

        vx = 3.0
        vy = 4.0

        state = np.array([1.0, 2.0, vx, vy])

        speed = 5.0

        drag_factor = 0.5 * rho * Cd * A

        expected_dvx = -(drag_factor / m) * speed * vx
        expected_dvy = -g - (drag_factor / m) * speed * vy

        result = t_derivatives_released(
            state,
            drag_params=(m, rho, Cd, A)
        )

        self.assertAlmostEqual(result[0], vx)
        self.assertAlmostEqual(result[1], vy)
        self.assertAlmostEqual(result[2], expected_dvx)
        self.assertAlmostEqual(result[3], expected_dvy)

    def test_drag_opposes_downward_motion(self):
        m = 80.0
        rho = 1.225
        Cd = 1.0
        A = 0.7

        vx = 0.0
        vy = -4.0

        state = np.array([0.0, 10.0, vx, vy])

        speed = 4.0
        drag_factor = 0.5 * rho * Cd * A

        expected_dvy = (
            -g
            - (drag_factor / m) * speed * vy
        )

        result = t_derivatives_released(
            state,
            drag_params=(m, rho, Cd, A)
        )

        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], expected_dvy)



if __name__ == '__main__':
    unittest.main()