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

        dth, dom = t_derivatives(theta, omega, L)

        self.assertEqual(dth, omega)
        self.assertEqual(dom, 0)

    def test_horiz_release(self):
        # At theta = pi/2, sin(theta) = 1, domega = -g / L
        L = 2.0
        dth, dom = t_derivatives(theta=math.pi / 2, omega=0.0, L=L)

        self.assertAlmostEqual(dth, 0.0)
        self.assertAlmostEqual(dom, -g/L)

    def test_inverted_spidey(self):
        # At top equilibrium (theta = pi), domega = 0
        dth, dom = t_derivatives(theta=math.pi, omega=1.5, L=1.0)

        self.assertAlmostEqual(dth, 1.5)
        self.assertAlmostEqual(dom, 0.0)

    def test_negative_angle_restoring_force(self):
        # Negative angle should accelerate positively back toward equilibrium
        L = 0.5
        dth, dom = t_derivatives(theta=-math.pi / 2, omega=-0.5, L=L)
        
        self.assertAlmostEqual(dth, -0.5)
        self.assertAlmostEqual(dom, g / L)

    def test_nonzero_velocity_and_length(self):
        # General case check with arbitrarily chosen parameters
        theta, omega, L = 0.5, 2.5, 3.0
        expected_dom = -(g / L) * math.sin(theta)

        dth, dom = t_derivatives(theta, omega, L)

        self.assertAlmostEqual(dth, omega)
        self.assertAlmostEqual(dom, expected_dom)

if __name__ == '__main__':
    unittest.main()