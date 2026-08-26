import unittest
import sys
from pathlib import Path
import math
import numpy as np

# add parent dir to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics import t_derivatives, energy, g

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

if __name__ == '__main__':
    unittest.main()