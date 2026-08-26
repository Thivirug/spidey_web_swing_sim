import unittest
import sys
from pathlib import Path

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

if __name__ == '__main__':
    unittest.main()