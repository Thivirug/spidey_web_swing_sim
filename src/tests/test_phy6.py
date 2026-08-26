import unittest

from physics import t_derivatives, g

class TDerivtests(unittest.TestCase):
    def test_simple(self, theta, omega, L):
        dth, dom = t_derivatives(theta, omega, L)

        self.assertEqual(dth, omega)
        self.assertEqual(dom, 0)

if __name__ == '__main__':
    unittest.main()