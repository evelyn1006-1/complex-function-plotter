import math
import unittest

from complex_plotter.exact_integration import attempt_exact_integral
from complex_plotter.integration import integrate_path
from complex_plotter.webapp import path_from_payload


RAY_POSITIVE_REAL = [{"type": "ray", "start": [0, 0], "through": [1, 0]}]
FULL_REAL_LINE = [{"type": "full_line", "start": [0, 0], "through": [1, 0]}]
REVERSED_FULL_REAL_LINE = [{"type": "full_line", "start": [0, 0], "through": [-1, 0]}]
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

    def test_meromorphic_ray_with_antiderivative_can_be_exact(self) -> None:
        result = integrate_path("1/(z+1)**2", RAY_POSITIVE_REAL, BOUNDS, method_mode="auto")

        self.assertIn(result["method"], {"exact", "residue-derivation"})
        self.assertEqual(result["exact_value"], "1")
        self.assertAlmostEqual(result["value"][0], 1.0)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_even_rational_half_line_uses_residue_derivation(self) -> None:
        result = integrate_path("1/(z**2+1)", RAY_POSITIVE_REAL, (-1, 3, -1, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi/2")
        self.assertTrue(any("upper half-plane" in note for note in result["notes"]))
        self.assertAlmostEqual(result["value"][0], math.pi / 2)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_full_real_line_uses_residue_derivation(self) -> None:
        result = integrate_path("1/(z**2+1)", FULL_REAL_LINE, (-3, 3, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi")
        self.assertTrue(any("full real line" in note for note in result["notes"]))
        self.assertAlmostEqual(result["value"][0], math.pi)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_full_line_with_entire_antiderivative_limits_can_be_exact(self) -> None:
        result = integrate_path("exp(-z**2)", FULL_REAL_LINE, (-3, 3, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "exact")
        self.assertAlmostEqual(result["value"][0], math.sqrt(math.pi))
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_repeated_upper_pole_is_counted_once_in_residue_derivation(self) -> None:
        result = integrate_path("1/(z**2+1)**2", FULL_REAL_LINE, (-3, 3, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi/2")
        self.assertEqual(len(result["residues"]), 1)

    def test_full_line_cannot_be_combined_with_other_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "full-line path cannot be combined"):
            path_from_payload({
                "path": [
                    {"type": "full_line", "start": [0, 0], "through": [1, 0]},
                    {"type": "line", "start": [0, 0], "end": [1, 0]},
                ]
            })

    def test_cosine_rational_half_line_uses_jordan_residue_derivation(self) -> None:
        result = integrate_path("cos(z)/(1+z**2)", RAY_POSITIVE_REAL, (-1, 4, -1, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi*exp(-1)/2")
        self.assertTrue(any("Jordan's lemma" in note for note in result["notes"]))
        self.assertAlmostEqual(result["value"][0], math.pi / (2 * math.e))
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_sine_times_odd_rational_half_line_uses_jordan_residue_derivation(self) -> None:
        result = integrate_path("z*sin(z)/(1+z**2)", RAY_POSITIVE_REAL, (-1, 4, -1, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi*exp(-1)/2")
        self.assertTrue(any("imaginary part" in note for note in result["notes"]))
        self.assertAlmostEqual(result["value"][0], math.pi / (2 * math.e))
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_exponential_rational_full_line_uses_jordan_residue_derivation(self) -> None:
        result = integrate_path("exp(i*z)/(1+z**2)", FULL_REAL_LINE, (-4, 4, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi*exp(-1)")
        self.assertAlmostEqual(result["value"][0], math.pi / math.e)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_noneven_rational_half_line_uses_keyhole_residue_derivation(self) -> None:
        result = integrate_path("1/(z**3+1)", RAY_POSITIVE_REAL, (-1, 4, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "2*sqrt(3)*pi/9")
        self.assertTrue(any("keyhole contour" in note for note in result["notes"]))
        self.assertAlmostEqual(result["value"][0], 2 * math.pi / (3 * math.sqrt(3)))
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_higher_degree_even_rational_half_line_residue_derivation(self) -> None:
        result = integrate_path("1/(z**4+1)", RAY_POSITIVE_REAL, (-1, 4, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "sqrt(2)*pi/4")
        self.assertAlmostEqual(result["value"][0], math.pi / (2 * math.sqrt(2)))
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_shifted_noneven_rational_full_line_residue_derivation(self) -> None:
        result = integrate_path("1/(z**2+2*z+2)", FULL_REAL_LINE, (-5, 5, -3, 3), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi")
        self.assertAlmostEqual(result["value"][0], math.pi)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_cosine_rational_with_frequency_and_scaled_pole(self) -> None:
        result = integrate_path("cos(3*z)/(z**2+4)", RAY_POSITIVE_REAL, (-1, 5, -3, 3), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi*exp(-6)/4")
        self.assertAlmostEqual(result["value"][0], math.pi * math.exp(-6) / 4)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_negative_frequency_exponential_uses_lower_half_plane(self) -> None:
        result = integrate_path("exp(-i*z)/(z**2+1)", FULL_REAL_LINE, (-4, 4, -2, 2), method_mode="auto")

        self.assertEqual(result["method"], "residue-derivation")
        self.assertEqual(result["exact_value"], "pi*exp(-1)")
        self.assertTrue(any("lower half-plane" in note for note in result["notes"]))
        self.assertAlmostEqual(result["value"][0], math.pi / math.e)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_reversed_full_line_orientation_is_preserved(self) -> None:
        result = integrate_path("1/(z**2+1)", REVERSED_FULL_REAL_LINE, (-3, 3, -2, 2), method_mode="auto")

        self.assertEqual(result["exact_value"], "-pi")
        self.assertAlmostEqual(result["value"][0], -math.pi)
        self.assertAlmostEqual(result["value"][1], 0.0)

    def test_pole_beyond_viewport_on_ray_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lies on the ray to infinity"):
            integrate_path("1/(z-5)**2", RAY_POSITIVE_REAL, (-1, 1, -1, 1), method_mode="auto")

    def test_real_axis_pole_is_rejected_instead_of_using_residue_shortcut(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite value"):
            integrate_path("1/(z**2-1)", RAY_POSITIVE_REAL, (-0.5, 0.5, -1, 1), method_mode="auto")


if __name__ == "__main__":
    unittest.main()
