import unittest
import sys
from pathlib import Path
import math

# add parent dir to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.physics import t_derivatives, g

class TDerivtests(unittest.TestCase):
    def test_simple(self):
        theta = 0
        omega = 0.0
        L = 15.0

        state = theta, omega
        result = t_derivatives(state, L)

        self.assertEqual(result[0], omega)
        self.assertEqual(result[1], 0)

    def test_horiz_release(self):
        # At theta = pi/2, sin(theta) = 1, domega = -g / L
        L = 2.0

        state = math.pi / 2, 0.0
        result = t_derivatives(state, L=L)

        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], -g/L)

    def test_inverted_spidey(self):
        # At top equilibrium (theta = pi), domega = 0
        state = math.pi, 1.5
        result = t_derivatives(state, L=1.0)

        self.assertAlmostEqual(result[0], 1.5)
        self.assertAlmostEqual(result[1], 0.0)

    def test_negative_angle_restoring_force(self):
        # Negative angle should accelerate positively back toward equilibrium
        L = 0.5
        state = -math.pi / 2, -0.5
        result = t_derivatives(state, L=L)
        
        self.assertAlmostEqual(result[0], -0.5)
        self.assertAlmostEqual(result[1], g / L)

    def test_nonzero_velocity_and_length(self):
        # General case check with arbitrarily chosen parameters
        theta, omega, L = 0.5, 2.5, 3.0

        state = theta, omega
        expected_dom = -(g / L) * math.sin(theta)

        result = t_derivatives(state, L)

        self.assertAlmostEqual(result[0], omega)
        self.assertAlmostEqual(result[1], expected_dom)

if __name__ == '__main__':
    unittest.main()