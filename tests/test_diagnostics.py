import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.diagnostics import energy_diagnostic

class TestEnergyDiagnostic(unittest.TestCase):

    def setUp(self):
        self.T = 5.0
        self.dt = 0.01
        self.theta0 = 45.0   # degrees
        self.omega0 = 0.0
        self.L = 15.0
        self.m = 1.0

        self.times, self.energies, self.rel_errors = energy_diagnostic(
            self.T,
            self.dt,
            self.theta0,
            self.omega0,
            self.L,
            self.m
        )

    def test_output_lengths_match(self):
        self.assertEqual(len(self.times), len(self.energies))
        self.assertEqual(len(self.times), len(self.rel_errors))

    def test_initial_relative_error_is_zero(self):
        self.assertAlmostEqual(
            self.rel_errors[0],
            0.0,
            places=12
        )

    def test_energy_values_are_finite(self):
        self.assertTrue(np.isfinite(self.energies).all())

    def test_relative_errors_are_finite(self):
        self.assertTrue(np.isfinite(self.rel_errors).all())

    def test_energy_is_positive(self):
        self.assertTrue(np.all(self.energies >= 0))

    def test_rk4_conserves_energy(self):
        max_error = np.max(np.abs(self.rel_errors))

        self.assertLess(
            max_error,
            1e-5
        )


if __name__ == "__main__":
    unittest.main()