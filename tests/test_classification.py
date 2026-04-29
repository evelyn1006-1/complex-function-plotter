import unittest

from complex_plotter.expressions import classify_expression, mobius_analysis, singularity_points_in_bounds
from complex_plotter.plotting import compute_plot_cached


def classification_kinds(expr: str, *, deep: bool = True) -> list[str]:
    return [item["kind"] for item in classify_expression(expr, deep=deep)["singularities"]]


def singularities_by_exact_point(expr: str) -> dict[str, dict]:
    return {
        item["exact_point"]: item
        for item in singularity_points_in_bounds(expr, (-4, 4, -4, 4), max_points=16)
    }


class ClassificationTests(unittest.TestCase):
    def test_classification_labels_cover_supported_categories(self) -> None:
        cases = [
            ("exp(z)", True, "entire"),
            ("sqrt(z)**2", True, "entire after symbolic simplification"),
            ("x+i*y", True, "locally holomorphic / analytic"),
            ("zeta(2)", False, "probably holomorphic (unclassified singularities)"),
            ("log(z)", True, "holomorphic on a branch domain"),
            ("1/(z-1)", True, "meromorphic"),
            ("sin(z)/z", True, "meromorphic with removable singularities"),
            ("exp(1/z)", True, "holomorphic except at isolated singularities"),
            ("jv(z,1)", True, "special-function analytic status uncertain"),
            ("1/(z-z)", True, "undefined / singular everywhere"),
            ("conj(z)", True, "non-holomorphic"),
            ("where(x>0,z,z**2)", True, "piecewise analytic (not globally holomorphic)"),
        ]

        for expr, deep, expected in cases:
            with self.subTest(expr=expr, deep=deep):
                self.assertEqual(classify_expression(expr, deep=deep)["analytic_label"], expected)

    def test_fast_constant_special_functions_are_not_active_pole_families(self) -> None:
        for expr in ("gamma(1)", "tan(1)", "zeta(2)"):
            with self.subTest(expr=expr):
                result = classify_expression(expr, deep=False)

                self.assertEqual(result["label"], "probably holomorphic (unclassified singularities)")
                self.assertFalse(result["flags"]["known_poles"])

    def test_comparison_and_modulo_syntax_are_non_holomorphic(self) -> None:
        for expr in ("z < 1", "z % 2"):
            with self.subTest(expr=expr):
                result = classify_expression(expr, deep=True)

                self.assertEqual(result["label"], "non-holomorphic")
                self.assertTrue(result["flags"]["non_holomorphic"])

    def test_nonzero_denominator_does_not_create_possible_pole(self) -> None:
        result = classify_expression("1/exp(z)", deep=True)

        self.assertEqual(result["label"], "entire")
        self.assertFalse(result["flags"]["denominators"])
        self.assertEqual(result["singularities"], [])

    def test_identically_zero_denominator_is_reported_as_undefined(self) -> None:
        result = classify_expression("1/(z-z)", deep=True)

        self.assertEqual(result["label"], "undefined / singular everywhere")
        self.assertTrue(result["flags"]["undefined_denominators"])
        self.assertEqual(result["singularities"][0]["kind"], "undefined_everywhere")

    def test_branch_domain_does_not_hide_local_pole_or_removable_point(self) -> None:
        sqrt_points = singularities_by_exact_point("sqrt(z)/(z-1)")
        self.assertEqual(sqrt_points["0"]["kind"], "symbolic_branch_point_candidate")
        self.assertEqual(sqrt_points["1"]["kind"], "symbolic_pole")
        self.assertEqual(sqrt_points["1"]["pole_order"], 1)

        log_points = singularities_by_exact_point("log(z)/(z-1)")
        self.assertEqual(log_points["0"]["kind"], "symbolic_branch_point_candidate")
        self.assertEqual(log_points["1"]["kind"], "symbolic_removable_singularity")

    def test_transcendental_over_pole_is_not_downgraded_to_removable(self) -> None:
        points = singularities_by_exact_point("(-1)**(1/z)")

        self.assertEqual(points["0"]["kind"], "symbolic_essential_candidate")

    def test_fast_singularity_notes_cover_feature_categories(self) -> None:
        cases = [
            ("1/(z-z)", "undefined_everywhere"),
            ("exp(1/z)", "essential_or_nonisolated"),
            ("z**0.5", "branch_point"),
            ("log(z)", "branch_cut"),
            ("1/(z-1)", "possible_pole"),
            ("gamma(z)", "known_pole_family"),
        ]

        for expr, expected_kind in cases:
            with self.subTest(expr=expr):
                self.assertIn(expected_kind, classification_kinds(expr, deep=True))

    def test_located_singularities_cover_symbolic_categories(self) -> None:
        cases = [
            ("1/(z-1)", "1", "symbolic_pole"),
            ("sin(z)/z", "0", "symbolic_removable_singularity"),
            ("log(z)", "0", "symbolic_branch_point_candidate"),
            ("exp(1/z)", "0", "symbolic_essential_candidate"),
            ("gamma(z)", "-1", "symbolic_pole"),
        ]

        for expr, point, expected_kind in cases:
            with self.subTest(expr=expr, point=point):
                points = singularities_by_exact_point(expr)
                self.assertEqual(points[point]["kind"], expected_kind)

    def test_mobius_analysis_detects_lft_data(self) -> None:
        result = mobius_analysis("(z+1)/(z-1)")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["label"], "elliptic")
        self.assertEqual(result["a"], "1")
        self.assertEqual(result["b"], "1")
        self.assertEqual(result["c"], "1")
        self.assertEqual(result["d"], "-1")
        self.assertEqual(result["determinant"], "-2")
        self.assertEqual(result["zero"], "-1")
        self.assertEqual(result["pole"], "1")
        self.assertEqual(set(result["fixed_points"]), {"1 - sqrt(2)", "1 + sqrt(2)"})

    def test_classification_includes_mobius_payload(self) -> None:
        result = classify_expression("(z+1)/(z-1)", deep=True)

        self.assertEqual(result["analytic_label"], "meromorphic")
        self.assertEqual(result["label"], "elliptic Möbius/LFT (meromorphic)")
        self.assertEqual(result["mobius"]["kind"], "mobius")
        self.assertEqual(result["mobius"]["pole"], "1")

    def test_transform_highlight_adds_animated_curve_trace(self) -> None:
        result = compute_plot_cached(
            "1/z",
            "transform",
            -2,
            2,
            -2,
            2,
            350,
            18,
            12,
            0.72,
            5,
            40,
            False,
            False,
            "circle",
            0.0,
            0.0,
            1.0,
        )

        self.assertEqual(result["kind"], "transform")
        self.assertEqual(len(result["traces"]), 2)
        self.assertEqual(result["traces"][1]["name"], "|z| = 1")
        self.assertEqual(len(result["frames"][0]["data"]), 2)

    def test_transform_highlight_supports_diagonal_and_off_center_circle(self) -> None:
        diagonal = compute_plot_cached(
            "z^2",
            "transform",
            -2,
            2,
            -2,
            2,
            350,
            18,
            12,
            0.72,
            5,
            40,
            False,
            False,
            "diagonal",
            -1.0,
            0.5,
            1.0,
        )
        circle = compute_plot_cached(
            "z^2",
            "transform",
            -2,
            2,
            -2,
            2,
            350,
            18,
            12,
            0.72,
            5,
            40,
            False,
            False,
            "circle",
            0.5,
            -0.25,
            0.75,
        )

        self.assertEqual(diagonal["traces"][1]["name"], "y = -1x +0.5")
        self.assertEqual(circle["traces"][1]["name"], "|z - (0.5-0.25i)| = 0.75")


if __name__ == "__main__":
    unittest.main()
