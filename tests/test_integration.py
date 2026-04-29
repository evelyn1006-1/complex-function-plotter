import math
import unittest

from complex_plotter.exact_integration import attempt_exact_integral
from complex_plotter.integration import integrate_path


RAY_POSITIVE_REAL = [{"type": "ray", "start": [0, 0], "through": [1, 0]}]
BOUNDS = (-1, 3, -1, 1)


class IntegrationTests(unittest.TestCase):
    def test_exact_mode_handles_convergent_ray_with_elementary_antiderivative(self) -> None:
        result = integrate_path("exp(-z)", RAY_POSITIVE_REAL, BOUNDS, method_mode="auto")

        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["exact_value"], "1")
        self.assertAlmostEqual(result["value"][0], 1.0)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_exact_mode_handles_convergent_ray_with_special_antiderivative_limit(self) -> None:
        result = integrate_path("exp(-z**2)", RAY_POSITIVE_REAL, BOUNDS, method_mode="auto")

        self.assertEqual(result["method"], "exact")
        self.assertAlmostEqual(result["value"][0], math.sqrt(math.pi) / 2)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_divergent_ray_antiderivative_limit_is_not_reported_as_exact(self) -> None:
        exact = attempt_exact_integral("exp(z)", RAY_POSITIVE_REAL, BOUNDS)

        self.assertIsNone(exact)


if __name__ == "__main__":
    unittest.main()
