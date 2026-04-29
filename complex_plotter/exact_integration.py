from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import sympy as sp

from .expressions import ENTIRE_FUNCTIONS, SAFE_CONSTANTS, analyze_expression, _parsed_expression
from .number_labels import complex_component_labels
from .paths import distance_to_path, finite_endpoints, is_closed_path, segment_end, segment_start, to_complex, winding_number


Z = sp.Symbol("z")
MAX_EXACT_SINGULARITIES = 240


SYMPY_FUNCTIONS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "asinh": sp.asinh,
    "acosh": sp.acosh,
    "atanh": sp.atanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "sech": sp.sech,
    "csch": sp.csch,
    "coth": sp.coth,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "gamma": sp.gamma,
    "digamma": getattr(sp, "digamma", None),
    "psi": getattr(sp, "digamma", None),
}

EXACT_MEROMORPHIC_FUNCTIONS = ENTIRE_FUNCTIONS | {
    "tan",
    "sec",
    "cot",
    "csc",
    "sech",
    "csch",
    "coth",
}


def _sympy_number(value: Any) -> sp.Expr | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float) and math.isfinite(value):
        return sp.Rational(repr(value))
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            return None
        return sp.Rational(repr(float(value.real))) + sp.I * sp.Rational(repr(float(value.imag)))
    return None


def _integer_exponent(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _integer_exponent(node.operand)
        return -value if value is not None else None
    return None


def _to_sympy(node: ast.AST) -> sp.Expr | None:
    if isinstance(node, ast.Constant):
        return _sympy_number(node.value)
    if isinstance(node, ast.Name):
        if node.id == "z":
            return Z
        if node.id in {"i", "j"}:
            return sp.I
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
        if node.id == "tau":
            return 2 * sp.pi
        return None
    if isinstance(node, ast.UnaryOp):
        operand = _to_sympy(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        return None
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            left = _to_sympy(node.left)
            exponent = _integer_exponent(node.right)
            if left is None:
                return None
            if exponent is not None:
                return left ** exponent
            right = _to_sympy(node.right)
            return left ** right if right is not None else None
        left = _to_sympy(node.left)
        right = _to_sympy(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return None
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return None
        func = SYMPY_FUNCTIONS.get(node.func.id)
        if func is None:
            return None
        args = [_to_sympy(arg) for arg in node.args]
        if any(arg is None for arg in args):
            return None
        return func(*args)
    return None


def _sympy_expr(expr: str) -> sp.Expr | None:
    try:
        tree, _code = _parsed_expression(expr)
        return _to_sympy(tree.body)
    except Exception:
        return None


def _complex_from_sympy(value: sp.Expr) -> complex | None:
    try:
        numeric = complex(sp.N(value, 18))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric.real) or not math.isfinite(numeric.imag):
        return None
    return numeric


def _exact_text(value: sp.Expr) -> str:
    simplified = sp.trigsimp(sp.simplify(value.rewrite(sp.sin)))
    return sp.sstr(simplified).replace("I", "i")


def _exact_latex(value: sp.Expr) -> str:
    simplified = sp.trigsimp(sp.simplify(value.rewrite(sp.sin)))
    return sp.latex(simplified)


def _sanitize_winding(value: float) -> int | None:
    if not np.isfinite(value):
        return None
    rounded = int(round(value))
    if abs(value - rounded) > 2e-2:
        return None
    return rounded


def _diag(bounds: tuple[float, float, float, float]) -> float:
    xmin, xmax, ymin, ymax = bounds
    return float(math.hypot(xmax - xmin, ymax - ymin))


def _is_zero_expr(value: sp.Expr) -> bool:
    simplified = sp.simplify(value)
    if simplified == 0:
        return True
    return bool(simplified.is_zero)


def _zero_order(expr: sp.Expr, point: sp.Expr, max_order: int = 16) -> int | None:
    for order in range(max_order + 1):
        try:
            coefficient = sp.diff(expr, Z, order).subs(Z, point)
        except Exception:
            return None
        if not _is_zero_expr(coefficient):
            return order
    return None


def _pole_order(sym_expr: sp.Expr, point: sp.Expr, max_order: int = 16) -> int | None:
    for order in range(max_order + 1):
        try:
            scaled = sp.simplify((Z - point) ** order * sym_expr)
            value = sp.limit(scaled, Z, point)
        except Exception:
            continue
        if value in (sp.oo, -sp.oo, sp.zoo) or value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            continue
        if _is_zero_expr(value):
            return None
        return order
    return None


def _series_text(series_expr: sp.Expr) -> str:
    return sp.sstr(series_expr).replace("I", "i")


def _series_latex(series_expr: sp.Expr) -> str:
    return sp.latex(series_expr)


def _local_residue_observability(sym_expr: sp.Expr, point: sp.Expr, pole_order: int | None) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if pole_order is not None:
        details["pole_order"] = pole_order
        shifted = f"(z - ({_exact_text(point)}))" if point != 0 else "z"
        if pole_order == 1:
            details["residue_note"] = f"Simple pole: the residue is the coefficient of {shifted}^-1 in the Laurent expansion."
        else:
            details["residue_note"] = (
                f"Pole order {pole_order}: the residue is the coefficient of {shifted}^-1 in the Laurent expansion, "
                f"equivalently the coefficient of {shifted}^{pole_order - 1} in {shifted}^{pole_order} f(z)."
            )

    try:
        series_order = max(4, (pole_order or 1) + 3)
        laurent_series = sp.series(sym_expr, Z, point, series_order)
        details["laurent_series"] = _series_text(laurent_series)
        details["laurent_series_latex"] = _series_latex(laurent_series)
    except Exception:
        pass

    try:
        _numerator, denominator = sp.fraction(sp.together(sym_expr))
        denominator_order = _zero_order(denominator, point)
        if denominator_order and denominator_order > 0:
            details["denominator_zero_order"] = denominator_order
            denominator_series = sp.series(denominator, Z, point, denominator_order + 3)
            details["denominator_series"] = _series_text(denominator_series)
            details["denominator_series_latex"] = _series_latex(denominator_series)
    except Exception:
        pass

    return details


def _inside_bounds(z: complex, bounds: tuple[float, float, float, float], pad: float) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return xmin - pad <= z.real <= xmax + pad and ymin - pad <= z.imag <= ymax + pad


def _singularity_search_pad(bounds: tuple[float, float, float, float]) -> float:
    return max(1e-6, 0.02 * _diag(bounds))


def _add_singularity_candidate(
    candidates: list[sp.Expr],
    seen: list[complex],
    point: sp.Expr,
    bounds: tuple[float, float, float, float],
    pad: float,
) -> bool:
    numeric = _complex_from_sympy(point)
    if numeric is None or not _inside_bounds(numeric, bounds, pad):
        return True
    if any(abs(numeric - existing) <= 1e-7 for existing in seen):
        return True
    candidates.append(point)
    seen.append(numeric)
    return len(candidates) <= MAX_EXACT_SINGULARITIES


def _n_range_for_linear_imageset(expr: sp.Expr, var: sp.Symbol, bounds: tuple[float, float, float, float], pad: float) -> range | None:
    coeff = sp.diff(expr, var)
    offset = sp.simplify(expr.subs(var, 0))
    if coeff.has(var) or offset.has(var):
        return None

    coeff_numeric = _complex_from_sympy(coeff)
    offset_numeric = _complex_from_sympy(offset)
    if coeff_numeric is None or offset_numeric is None or abs(coeff_numeric) < 1e-12:
        return None

    xmin, xmax, ymin, ymax = bounds
    intervals: list[tuple[float, float]] = []
    if abs(coeff_numeric.real) > 1e-12:
        a = (xmin - pad - offset_numeric.real) / coeff_numeric.real
        b = (xmax + pad - offset_numeric.real) / coeff_numeric.real
        intervals.append((min(a, b), max(a, b)))
    elif not xmin - pad <= offset_numeric.real <= xmax + pad:
        return range(0)

    if abs(coeff_numeric.imag) > 1e-12:
        a = (ymin - pad - offset_numeric.imag) / coeff_numeric.imag
        b = (ymax + pad - offset_numeric.imag) / coeff_numeric.imag
        intervals.append((min(a, b), max(a, b)))
    elif not ymin - pad <= offset_numeric.imag <= ymax + pad:
        return range(0)

    if not intervals:
        return None
    low = max(item[0] for item in intervals)
    high = min(item[1] for item in intervals)
    if low > high:
        return range(0)
    start = math.floor(low) - 1
    stop = math.ceil(high) + 2
    if stop - start > MAX_EXACT_SINGULARITIES:
        return None
    return range(start, stop)


def _candidates_from_imageset(
    image_set: sp.ImageSet,
    bounds: tuple[float, float, float, float],
    pad: float,
    candidates: list[sp.Expr],
    seen: list[complex],
) -> bool:
    if not isinstance(image_set.lamda, sp.Lambda) or image_set.base_set != sp.S.Integers:
        return False
    variables = image_set.lamda.variables
    if len(variables) != 1:
        return False
    var = variables[0]
    n_range = _n_range_for_linear_imageset(image_set.lamda.expr, var, bounds, pad)
    if n_range is None:
        return False
    for n in n_range:
        point = sp.simplify(image_set.lamda.expr.subs(var, n))
        if not _add_singularity_candidate(candidates, seen, point, bounds, pad):
            return False
    return True


def _collect_singularity_candidates(
    singularities: sp.Set,
    bounds: tuple[float, float, float, float],
    pad: float,
    candidates: list[sp.Expr],
    seen: list[complex],
) -> bool:
    if singularities in (sp.S.EmptySet, sp.EmptySet):
        return True
    if isinstance(singularities, sp.FiniteSet):
        for point in singularities:
            if not _add_singularity_candidate(candidates, seen, point, bounds, pad):
                return False
        return True
    if isinstance(singularities, sp.Union):
        for subset in singularities.args:
            if not _collect_singularity_candidates(subset, bounds, pad, candidates, seen):
                return False
        return True
    if isinstance(singularities, sp.ImageSet):
        return _candidates_from_imageset(singularities, bounds, pad, candidates, seen)
    return False


def _singularity_candidates(sym_expr: sp.Expr, bounds: tuple[float, float, float, float]) -> list[sp.Expr] | None:
    try:
        singularities = sp.singularities(sym_expr, Z)
    except Exception:
        return None
    candidates: list[sp.Expr] = []
    seen: list[complex] = []
    pad = _singularity_search_pad(bounds)
    if not _collect_singularity_candidates(singularities, bounds, pad, candidates, seen):
        return None
    return candidates


def _result(
    *,
    value: sp.Expr,
    notes: list[str],
    residues: list[dict[str, Any]] | None = None,
    method: str = "exact",
) -> dict[str, Any] | None:
    value = sp.trigsimp(sp.simplify(value.rewrite(sp.sin)))
    numeric = _complex_from_sympy(value)
    if numeric is None:
        return None
    return {
        "method": method,
        "value": numeric,
        "exact_value": _exact_text(value),
        "exact_latex": _exact_latex(value),
        "notes": notes,
        "residues": residues or [],
    }


def _attempt_exact_residues(
    expr: str,
    sym_expr: sp.Expr,
    path: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    if not is_closed_path(path):
        return None
    features = analyze_expression(expr)
    function_names = set(features.used_names) - SAFE_CONSTANTS
    if features.has_branchy or features.has_nonanalytic or features.has_piecewise:
        return None
    if not function_names <= EXACT_MEROMORPHIC_FUNCTIONS:
        return None

    singularities = _singularity_candidates(sym_expr, bounds)
    if singularities is None:
        return None
    if not singularities:
        return _result(
            value=sp.Integer(0),
            notes=["Exact mode: the integrand is holomorphic on and inside the closed contour."],
        )

    contour_tol = max(1e-4, 1e-3 * _diag(bounds))
    total = sp.Integer(0)
    residue_markers: list[dict[str, Any]] = []
    used_higher_order_laurent = False

    for singularity in singularities:
        root_numeric = _complex_from_sympy(singularity)
        if root_numeric is None:
            return None
        winding = _sanitize_winding(winding_number(path, root_numeric, bounds))
        if winding in (None, 0):
            continue
        if distance_to_path(path, root_numeric, bounds) <= contour_tol:
            return None
        residue = sp.residue(sym_expr, Z, singularity)
        residue = sp.trigsimp(sp.simplify(residue.rewrite(sp.sin)))
        residue_numeric = _complex_from_sympy(residue)
        if residue_numeric is None:
            return None
        pole_order = _pole_order(sym_expr, singularity)
        if pole_order is not None and pole_order > 1:
            used_higher_order_laurent = True
        total += winding * residue
        residue_markers.append({
            "point": [float(root_numeric.real), float(root_numeric.imag)],
            "point_labels": complex_component_labels(root_numeric),
            "winding": winding,
            "residue": [float(residue_numeric.real), float(residue_numeric.imag)],
            "residue_labels": complex_component_labels(residue_numeric),
            "exact_residue": _exact_text(residue),
            "exact_residue_latex": _exact_latex(residue),
            "exact_point": _exact_text(singularity),
            "exact_point_latex": _exact_latex(singularity),
            "radius": 0.0,
            **_local_residue_observability(sym_expr, singularity, pole_order),
        })

    if not residue_markers:
        return _result(
            value=sp.Integer(0),
            notes=["Exact mode: found no singularities enclosed by the contour, so the residue theorem gives 0."],
            residues=[],
        )

    note = (
        "Exact mode: used the residue theorem with Laurent-series coefficient extraction for higher-order enclosed poles."
        if used_higher_order_laurent
        else "Exact mode: used the symbolic residue theorem for enclosed simple poles."
    )

    return _result(
        value=sp.simplify(2 * sp.pi * sp.I * total),
        notes=[note],
        residues=residue_markers,
    )


def _sympy_point(point: complex) -> sp.Expr:
    return sp.Rational(repr(float(np.real(point)))) + sp.I * sp.Rational(repr(float(np.imag(point))))


def _exact_meromorphic_features(features: Any) -> bool:
    if features.has_branchy or features.has_nonanalytic or features.has_piecewise:
        return False
    function_names = set(features.used_names) - SAFE_CONSTANTS
    return bool(function_names <= EXACT_MEROMORPHIC_FUNCTIONS)


def _sympy_ray_direction(segment: dict[str, Any]) -> sp.Expr | None:
    start = to_complex(segment["start"])
    through = to_complex(segment["through"])
    direction = through - start
    if abs(direction) < 1e-12:
        return None

    if abs(direction.imag) <= 1e-12:
        return sp.Integer(1) if direction.real > 0 else sp.Integer(-1)
    if abs(direction.real) <= 1e-12:
        return sp.I if direction.imag > 0 else -sp.I

    unit = direction / abs(direction)
    return _sympy_point(unit)


def _ray_parameter_for_point(segment: dict[str, Any], point: complex, tol: float = 1e-8) -> float | None:
    start = to_complex(segment["start"])
    through = to_complex(segment["through"])
    direction = through - start
    if abs(direction) < 1e-12:
        return None
    direction /= abs(direction)
    relative = point - start
    parameter = float(relative.real * direction.real + relative.imag * direction.imag)
    perpendicular = abs(relative - parameter * direction)
    scale = max(1.0, abs(start), abs(point))
    if perpendicular <= tol * scale and parameter >= -tol * scale:
        return parameter
    return None


def _line_parameter_for_point(segment: dict[str, Any], point: complex, tol: float = 1e-8) -> float | None:
    start = to_complex(segment["start"])
    through = to_complex(segment["through"])
    direction = through - start
    if abs(direction) < 1e-12:
        return None
    direction /= abs(direction)
    relative = point - start
    parameter = float(relative.real * direction.real + relative.imag * direction.imag)
    perpendicular = abs(relative - parameter * direction)
    scale = max(1.0, abs(start), abs(point))
    if perpendicular <= tol * scale:
        return parameter
    return None


def _is_positive_real_ray_from_origin(path: list[dict[str, Any]], tol: float = 1e-8) -> bool:
    if len(path) != 1 or path[0]["type"] != "ray":
        return False
    return _is_positive_real_improper_segment_from_origin(path[0], tol)


def _is_positive_real_full_line_from_origin(path: list[dict[str, Any]], tol: float = 1e-8) -> bool:
    if len(path) != 1 or path[0]["type"] != "full_line":
        return False
    return _is_positive_real_improper_segment_from_origin(path[0], tol)


def _is_positive_real_improper_segment_from_origin(segment: dict[str, Any], tol: float = 1e-8) -> bool:
    start = to_complex(segment["start"])
    through = to_complex(segment["through"])
    direction = through - start
    if abs(start) > tol or abs(direction) < tol:
        return False
    direction /= abs(direction)
    return bool(abs(direction.imag) <= tol and direction.real > 0)


def _finite_singularity_points(sym_expr: sp.Expr) -> list[sp.Expr] | None:
    try:
        singularities = sp.singularities(sym_expr, Z)
    except Exception:
        return None
    if singularities in (sp.S.EmptySet, sp.EmptySet):
        return []
    if isinstance(singularities, sp.FiniteSet):
        return list(singularities)
    if isinstance(singularities, sp.Union):
        points: list[sp.Expr] = []
        for subset in singularities.args:
            subset_points = _finite_singularity_points_from_set(subset)
            if subset_points is None:
                return None
            points.extend(subset_points)
        return points
    return None


def _finite_singularity_points_from_set(singularities: sp.Set) -> list[sp.Expr] | None:
    if singularities in (sp.S.EmptySet, sp.EmptySet):
        return []
    if isinstance(singularities, sp.FiniteSet):
        return list(singularities)
    if isinstance(singularities, sp.Union):
        points: list[sp.Expr] = []
        for subset in singularities.args:
            subset_points = _finite_singularity_points_from_set(subset)
            if subset_points is None:
                return None
            points.extend(subset_points)
        return points
    return None


def _validate_ray_avoids_finite_singularities(sym_expr: sp.Expr, segment: dict[str, Any]) -> bool:
    singularities = _finite_singularity_points(sym_expr)
    if singularities is None:
        return False
    for singularity in singularities:
        numeric = _complex_from_sympy(singularity)
        if numeric is None:
            return False
        parameter = _ray_parameter_for_point(segment, numeric)
        if parameter is not None:
            raise ValueError(
                f"A detected singularity at {numeric} lies on the ray to infinity, "
                "so the integral is undefined for this app's ordinary contour integral mode."
            )
    return True


def _validate_full_line_avoids_finite_singularities(sym_expr: sp.Expr, segment: dict[str, Any]) -> bool:
    singularities = _finite_singularity_points(sym_expr)
    if singularities is None:
        return False
    for singularity in singularities:
        numeric = _complex_from_sympy(singularity)
        if numeric is None:
            return False
        parameter = _line_parameter_for_point(segment, numeric)
        if parameter is not None:
            raise ValueError(
                f"A detected singularity at {numeric} lies on the full line, "
                "so the integral is undefined for this app's ordinary contour integral mode."
            )
    return True


def _clean_finite_limit(value: sp.Expr) -> sp.Expr | None:
    value = sp.trigsimp(sp.simplify(value.rewrite(sp.sin)))
    if value in (sp.oo, -sp.oo, sp.zoo, sp.nan) or value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None
    if value.has(sp.Limit, sp.Integral):
        return None
    if _complex_from_sympy(value) is None:
        return None
    return value


def _rational_polys(sym_expr: sp.Expr) -> tuple[sp.Poly, sp.Poly] | None:
    if not sym_expr.is_rational_function(Z):
        return None
    try:
        numerator, denominator = sp.fraction(sp.together(sym_expr))
        numerator_poly = sp.Poly(numerator, Z)
        denominator_poly = sp.Poly(denominator, Z)
    except Exception:
        return None
    if denominator_poly.is_zero:
        return None
    return numerator_poly, denominator_poly


def _poly_has_real_coefficients(poly: sp.Poly) -> bool:
    for coefficient in poly.all_coeffs():
        if not _is_zero_expr(sp.im(coefficient)):
            return False
    return True


def _polynomial_roots_exact(poly: sp.Poly) -> list[sp.Expr] | None:
    try:
        roots = sp.roots(poly.as_expr(), Z)
    except Exception:
        return None
    if sum(roots.values()) != poly.degree():
        return None
    return list(roots)


def _residue_marker(sym_expr: sp.Expr, pole: sp.Expr, residue: sp.Expr) -> dict[str, Any] | None:
    pole_numeric = _complex_from_sympy(pole)
    residue_numeric = _complex_from_sympy(residue)
    if pole_numeric is None or residue_numeric is None:
        return None
    return {
        "point": [float(pole_numeric.real), float(pole_numeric.imag)],
        "point_labels": complex_component_labels(pole_numeric),
        "winding": 1,
        "residue": [float(residue_numeric.real), float(residue_numeric.imag)],
        "residue_labels": complex_component_labels(residue_numeric),
        "exact_residue": _exact_text(residue),
        "exact_residue_latex": _exact_latex(residue),
        "exact_point": _exact_text(pole),
        "exact_point_latex": _exact_latex(pole),
        "radius": 0.0,
        **_local_residue_observability(sym_expr, pole, _pole_order(sym_expr, pole)),
    }


def _attempt_real_axis_rational_residue(
    expr: str,
    sym_expr: sp.Expr,
    path: list[dict[str, Any]],
) -> dict[str, Any] | None:
    is_full_line = _is_positive_real_full_line_from_origin(path)
    is_half_line = _is_positive_real_ray_from_origin(path)
    if not (is_full_line or is_half_line):
        return None

    features = analyze_expression(expr)
    if not _exact_meromorphic_features(features):
        return None

    rational = _rational_polys(sym_expr)
    if rational is None:
        return None
    numerator_poly, denominator_poly = rational
    if not (_poly_has_real_coefficients(numerator_poly) and _poly_has_real_coefficients(denominator_poly)):
        return None
    if is_half_line and not _is_zero_expr(sp.simplify(sym_expr.subs(Z, -Z) - sym_expr)):
        return None
    if denominator_poly.degree() - numerator_poly.degree() < 2:
        return None

    poles = _polynomial_roots_exact(denominator_poly)
    if poles is None:
        return None

    upper_poles: list[sp.Expr] = []
    for pole in poles:
        pole_numeric = _complex_from_sympy(pole)
        if pole_numeric is None:
            return None
        if abs(pole_numeric.imag) <= 1e-8:
            return None
        if pole_numeric.imag > 0:
            upper_poles.append(pole)
    if not upper_poles:
        return None

    total_residue = sp.Integer(0)
    residue_markers: list[dict[str, Any]] = []
    for pole in upper_poles:
        residue = sp.trigsimp(sp.simplify(sp.residue(sym_expr, Z, pole).rewrite(sp.sin)))
        marker = _residue_marker(sym_expr, pole, residue)
        if marker is None:
            return None
        total_residue += residue
        residue_markers.append(marker)

    pole_list = ", ".join(_exact_text(pole) for pole in upper_poles)
    multiplier = 2 * sp.pi * sp.I if is_full_line else sp.pi * sp.I
    leading_note = (
        "Residue derivation: integrate over the full real line from -infinity to infinity."
        if is_full_line
        else "Residue derivation: the integrand is an even rational function, so the half-line integral is one half of the real-line integral."
    )
    conclusion = (
        "Therefore the full-line integral equals 2*pi*i times the sum of those residues."
        if is_full_line
        else "Therefore the ray integral equals pi*i times the sum of those residues."
    )
    return _result(
        value=multiplier * total_residue,
        method="residue-derivation",
        residues=residue_markers,
        notes=[
            leading_note,
            "Use an upper half-plane semicircle. The denominator degree exceeds the numerator degree by at least 2, so the arc contribution tends to 0.",
            f"Enclosed upper half-plane poles: {pole_list}.",
            conclusion,
        ],
    )


def _linear_real_frequency(arg: sp.Expr) -> sp.Expr | None:
    coeff = sp.simplify(sp.diff(arg, Z))
    offset = sp.simplify(arg.subs(Z, 0))
    if coeff.has(Z) or not _is_zero_expr(offset):
        return None
    coeff_numeric = _complex_from_sympy(coeff)
    if coeff_numeric is None or abs(coeff_numeric.imag) > 1e-10:
        return None
    if abs(coeff_numeric.real) <= 1e-12:
        return None
    return sp.simplify(coeff if coeff_numeric.real > 0 else -coeff)


def _linear_imaginary_frequency(arg: sp.Expr) -> sp.Expr | None:
    coeff = sp.simplify(sp.diff(arg, Z))
    offset = sp.simplify(arg.subs(Z, 0))
    if coeff.has(Z) or not _is_zero_expr(offset):
        return None
    frequency = sp.simplify(coeff / sp.I)
    frequency_numeric = _complex_from_sympy(frequency)
    if frequency_numeric is None or abs(frequency_numeric.imag) > 1e-10:
        return None
    if abs(frequency_numeric.real) <= 1e-12:
        return None
    return frequency


def _trig_rational_parts(sym_expr: sp.Expr) -> tuple[str, sp.Expr, sp.Expr] | None:
    cos_terms = list(sym_expr.atoms(sp.cos))
    sin_terms = list(sym_expr.atoms(sp.sin))
    exp_terms = [term for term in sym_expr.atoms(sp.exp) if _linear_imaginary_frequency(term.args[0]) is not None]
    if sum(bool(items) for items in (cos_terms, sin_terms, exp_terms)) != 1:
        return None

    projection = "identity"
    if cos_terms:
        if len(cos_terms) != 1:
            return None
        active_term = cos_terms[0]
        frequency = _linear_real_frequency(active_term.args[0])
        projection = "real"
    elif sin_terms:
        if len(sin_terms) != 1:
            return None
        active_term = sin_terms[0]
        frequency = _linear_real_frequency(active_term.args[0])
        projection = "imag"
    else:
        if len(exp_terms) != 1:
            return None
        active_term = exp_terms[0]
        frequency = _linear_imaginary_frequency(active_term.args[0])

    if frequency is None:
        return None
    rational_part = sp.simplify(sym_expr / active_term)
    if rational_part.has(sp.sin, sp.cos, sp.exp, sp.log):
        return None
    if _rational_polys(rational_part) is None:
        return None
    return projection, frequency, rational_part


def _project_fourier_value(value: sp.Expr, projection: str) -> sp.Expr:
    if projection == "real":
        return sp.simplify(sp.re(value))
    if projection == "imag":
        return sp.simplify(sp.im(value))
    return sp.simplify(value)


def _attempt_real_axis_fourier_rational_residue(
    expr: str,
    sym_expr: sp.Expr,
    path: list[dict[str, Any]],
) -> dict[str, Any] | None:
    is_full_line = _is_positive_real_full_line_from_origin(path)
    is_half_line = _is_positive_real_ray_from_origin(path)
    if not (is_full_line or is_half_line):
        return None

    features = analyze_expression(expr)
    if not _exact_meromorphic_features(features):
        return None
    if is_half_line and not _is_zero_expr(sp.simplify(sym_expr.subs(Z, -Z) - sym_expr)):
        return None

    parts = _trig_rational_parts(sym_expr)
    if parts is None:
        return None
    projection, frequency, rational_part = parts
    rational = _rational_polys(rational_part)
    if rational is None:
        return None
    numerator_poly, denominator_poly = rational
    if not (_poly_has_real_coefficients(numerator_poly) and _poly_has_real_coefficients(denominator_poly)):
        return None
    if denominator_poly.degree() <= numerator_poly.degree():
        return None

    poles = _polynomial_roots_exact(denominator_poly)
    if poles is None:
        return None

    frequency_numeric = _complex_from_sympy(frequency)
    if frequency_numeric is None or abs(frequency_numeric.imag) > 1e-10:
        return None
    use_upper_half_plane = frequency_numeric.real > 0
    enclosed_poles: list[sp.Expr] = []
    for pole in poles:
        pole_numeric = _complex_from_sympy(pole)
        if pole_numeric is None:
            return None
        if abs(pole_numeric.imag) <= 1e-8:
            return None
        if (use_upper_half_plane and pole_numeric.imag > 0) or (not use_upper_half_plane and pole_numeric.imag < 0):
            enclosed_poles.append(pole)
    if not enclosed_poles:
        return None

    contour_expr = sp.exp(sp.I * frequency * Z) * rational_part
    total_residue = sp.Integer(0)
    residue_markers: list[dict[str, Any]] = []
    for pole in enclosed_poles:
        residue = sp.trigsimp(sp.simplify(sp.residue(contour_expr, Z, pole).rewrite(sp.sin)))
        marker = _residue_marker(contour_expr, pole, residue)
        if marker is None:
            return None
        total_residue += residue
        residue_markers.append(marker)

    contour_factor = 2 * sp.pi * sp.I if use_upper_half_plane else -2 * sp.pi * sp.I
    full_line_value = _project_fourier_value(contour_factor * total_residue, projection)
    value = sp.simplify(full_line_value if is_full_line else full_line_value / 2)
    pole_list = ", ".join(_exact_text(pole) for pole in enclosed_poles)
    half_plane = "upper" if use_upper_half_plane else "lower"
    projection_text = {
        "real": "real part",
        "imag": "imaginary part",
        "identity": "value",
    }[projection]
    leading_note = (
        f"Residue derivation: rewrite the integrand using an exp(i*a*z) rational contour integral and take its {projection_text}."
        if is_full_line
        else f"Residue derivation: the integrand is even, so the half-line integral is one half of the full real-line {projection_text}."
    )
    conclusion = (
        f"Take the {projection_text} of the residue-theorem value to recover the full-line integral."
        if is_full_line
        else f"Take one half of the {projection_text} of the residue-theorem value to recover the ray integral."
    )
    return _result(
        value=value,
        method="residue-derivation",
        residues=residue_markers,
        notes=[
            leading_note,
            f"Use the {half_plane} half-plane contour for exp(i*({_exact_text(frequency)})*z) times the rational factor. Jordan's lemma applies because the exponential decays there and the rational factor tends to 0.",
            f"Enclosed {half_plane} half-plane poles: {pole_list}.",
            conclusion,
        ],
    )


def _branch_log_offset_for_positive_keyhole(point: sp.Expr) -> sp.Expr | None:
    numeric = _complex_from_sympy(point)
    if numeric is None:
        return None
    if abs(numeric.imag) <= 1e-8 and numeric.real > 0:
        return None
    return 2 * sp.pi * sp.I if numeric.imag < -1e-8 else sp.Integer(0)


def _attempt_keyhole_rational_half_line_residue(
    expr: str,
    sym_expr: sp.Expr,
    path: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_positive_real_ray_from_origin(path):
        return None

    features = analyze_expression(expr)
    if not _exact_meromorphic_features(features):
        return None

    rational = _rational_polys(sym_expr)
    if rational is None:
        return None
    numerator_poly, denominator_poly = rational
    if not (_poly_has_real_coefficients(numerator_poly) and _poly_has_real_coefficients(denominator_poly)):
        return None
    if denominator_poly.degree() - numerator_poly.degree() < 2:
        return None
    if _is_zero_expr(denominator_poly.as_expr().subs(Z, 0)):
        return None

    poles = _polynomial_roots_exact(denominator_poly)
    if poles is None:
        return None

    total = sp.Integer(0)
    residue_markers: list[dict[str, Any]] = []
    for pole in poles:
        pole_numeric = _complex_from_sympy(pole)
        if pole_numeric is None:
            return None
        if abs(pole_numeric.imag) <= 1e-8 and pole_numeric.real > 0:
            return None
        branch_offset = _branch_log_offset_for_positive_keyhole(pole)
        if branch_offset is None:
            return None
        weighted_residue = sp.trigsimp(sp.simplify(sp.residue(sym_expr * (sp.log(Z) + branch_offset), Z, pole).rewrite(sp.sin)))
        marker = _residue_marker(sym_expr, pole, weighted_residue)
        if marker is None:
            return None
        total += weighted_residue
        residue_markers.append(marker)

    pole_list = ", ".join(_exact_text(pole) for pole in poles)
    return _result(
        value=-total,
        method="residue-derivation",
        residues=residue_markers,
        notes=[
            "Residue derivation: use a keyhole contour around the positive real axis with the branch 0 < arg(z) < 2*pi.",
            "The large and small circular arcs vanish because the rational integrand decays faster than 1/z at infinity and is finite at the origin.",
            f"Poles inside the keyhole contour: {pole_list}.",
            "The jump in log(z) across the positive real axis isolates the desired half-line integral, giving minus the sum of the log-weighted residues.",
        ],
    )


def _antiderivative_delta_for_segment(antiderivative: sp.Expr, segment: dict[str, Any]) -> sp.Expr | None:
    start = to_complex(segment["start"]) if segment["type"] == "full_line" else segment_start(segment)
    if start is None:
        return None
    start_expr = _sympy_point(start)

    if segment["type"] == "ray":
        direction = _sympy_ray_direction(segment)
        if direction is None:
            return None
        s = sp.Symbol("s", positive=True, real=True)
        ray_expr = start_expr + direction * s
        try:
            tail_value = sp.limit(antiderivative.subs(Z, ray_expr), s, sp.oo)
        except Exception:
            return None
        tail_value = _clean_finite_limit(tail_value)
        if tail_value is None:
            return None
        return sp.simplify(tail_value - antiderivative.subs(Z, start_expr))

    if segment["type"] == "full_line":
        direction = _sympy_ray_direction(segment)
        if direction is None:
            return None
        s = sp.Symbol("s", real=True)
        line_expr = start_expr + direction * s
        try:
            positive_tail = sp.limit(antiderivative.subs(Z, line_expr), s, sp.oo)
            negative_tail = sp.limit(antiderivative.subs(Z, line_expr), s, -sp.oo)
        except Exception:
            return None
        positive_tail = _clean_finite_limit(positive_tail)
        negative_tail = _clean_finite_limit(negative_tail)
        if positive_tail is None or negative_tail is None:
            return None
        return sp.simplify(positive_tail - negative_tail)

    end = segment_end(segment)
    if end is None:
        return None
    return sp.simplify(antiderivative.subs(Z, _sympy_point(end)) - antiderivative.subs(Z, start_expr))


def _attempt_exact_antiderivative(
    expr: str,
    sym_expr: sp.Expr,
    path: list[dict[str, Any]],
) -> dict[str, Any] | None:
    features = analyze_expression(expr)
    has_ray = any(segment["type"] == "ray" for segment in path)
    has_full_line = any(segment["type"] == "full_line" for segment in path)
    has_improper = has_ray or has_full_line
    meromorphic_ray = (
        has_improper
        and len(path) == 1
        and path[0]["type"] in {"ray", "full_line"}
        and _exact_meromorphic_features(features)
        and not features.proven_entire
    )
    if not features.proven_entire and not meromorphic_ray:
        return None
    if meromorphic_ray:
        if path[0]["type"] == "ray" and not _validate_ray_avoids_finite_singularities(sym_expr, path[0]):
            return None
        if path[0]["type"] == "full_line" and not _validate_full_line_avoids_finite_singularities(sym_expr, path[0]):
            return None

    start, end = finite_endpoints(path)
    if (start is None and not has_full_line) or (end is None and not has_improper):
        return None
    if not has_improper and start is not None and end is not None and abs(start - end) <= 1e-12:
        return _result(
            value=sp.Integer(0),
            notes=["Exact mode: the contour is closed and the integrand is in the app's conservative entire-function class."],
        )

    antiderivative = sp.integrate(sym_expr, Z)
    if antiderivative.has(sp.Integral):
        return None

    deltas: list[sp.Expr] = []
    for segment in path:
        delta = _antiderivative_delta_for_segment(antiderivative, segment)
        if delta is None:
            return None
        deltas.append(delta)

    value = sp.simplify(sum(deltas, sp.Integer(0)))
    if has_improper:
        note = (
            "Exact mode: used a SymPy antiderivative of a meromorphic integrand and convergent symbolic limits at infinity."
            if meromorphic_ray
            else "Exact mode: used a SymPy antiderivative and convergent symbolic limits at infinity."
        )
        return _result(
            value=value,
            notes=[note],
        )
    return _result(
        value=value,
        notes=["Exact mode: used a SymPy antiderivative of an entire integrand, so the value depends only on endpoints."],
    )


def attempt_exact_integral(
    expr: str,
    path: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    sym_expr = _sympy_expr(expr)
    if sym_expr is None:
        return None

    residue_result = _attempt_exact_residues(expr, sym_expr, path, bounds)
    if residue_result is not None:
        return residue_result
    residue_ray_result = _attempt_real_axis_rational_residue(expr, sym_expr, path)
    if residue_ray_result is not None:
        return residue_ray_result
    fourier_residue_result = _attempt_real_axis_fourier_rational_residue(expr, sym_expr, path)
    if fourier_residue_result is not None:
        return fourier_residue_result
    keyhole_residue_result = _attempt_keyhole_rational_half_line_residue(expr, sym_expr, path)
    if keyhole_residue_result is not None:
        return keyhole_residue_result
    return _attempt_exact_antiderivative(expr, sym_expr, path)
