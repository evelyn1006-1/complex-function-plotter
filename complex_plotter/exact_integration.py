from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import sympy as sp

from .expressions import ENTIRE_FUNCTIONS, SAFE_CONSTANTS, analyze_expression, _parsed_expression
from .number_labels import complex_component_labels
from .paths import distance_to_path, finite_endpoints, is_closed_path, winding_number


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
    numeric = complex(sp.N(value, 18))
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
) -> dict[str, Any] | None:
    value = sp.trigsimp(sp.simplify(value.rewrite(sp.sin)))
    numeric = _complex_from_sympy(value)
    if numeric is None:
        return None
    return {
        "method": "exact",
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


def _attempt_exact_antiderivative(
    expr: str,
    sym_expr: sp.Expr,
    path: list[dict[str, Any]],
) -> dict[str, Any] | None:
    features = analyze_expression(expr)
    if not features.proven_entire:
        return None

    start, end = finite_endpoints(path)
    if start is None or end is None:
        return None
    if abs(start - end) <= 1e-12:
        return _result(
            value=sp.Integer(0),
            notes=["Exact mode: the contour is closed and the integrand is in the app's conservative entire-function class."],
        )

    antiderivative = sp.integrate(sym_expr, Z)
    if antiderivative.has(sp.Integral):
        return None
    value = sp.simplify(antiderivative.subs(Z, _sympy_point(end)) - antiderivative.subs(Z, _sympy_point(start)))
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
    return _attempt_exact_antiderivative(expr, sym_expr, path)
