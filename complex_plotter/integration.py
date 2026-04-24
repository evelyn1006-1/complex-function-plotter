from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import integrate, optimize
from scipy.integrate import IntegrationWarning

from .exact_integration import attempt_exact_integral
from .expressions import analyze_expression, evaluate, evaluate_scalar
from .number_labels import complex_component_labels
from .paths import (
    distance_to_path,
    finite_endpoints,
    is_closed_path,
    path_summary,
    path_xy_for_plot,
    sample_path,
    segment_end,
    segment_start,
    to_complex,
    winding_number,
)


@dataclass
class NumericIntegrationResult:
    value: complex
    abs_error: float
    warnings: list[str]
    converged: bool


@dataclass
class ResidueInfo:
    point: complex
    winding: int
    residue: complex
    radius: float


def _complex_quad(func, a: float, b: float, *, points: list[float] | None = None, limit: int = 200) -> NumericIntegrationResult:
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        real_val, real_err = integrate.quad(lambda t: float(np.real(func(t))), a, b, epsabs=1e-9, epsrel=1e-8, limit=limit, points=points)
        imag_val, imag_err = integrate.quad(lambda t: float(np.imag(func(t))), a, b, epsabs=1e-9, epsrel=1e-8, limit=limit, points=points)
    for item in caught:
        warning_messages.append(str(item.message))
    return NumericIntegrationResult(
        value=complex(real_val, imag_val),
        abs_error=float(real_err + imag_err),
        warnings=warning_messages,
        converged=len(warning_messages) == 0,
    )


def _integrate_line_like(expr: str, z_of_t, dz_of_t, *, points: list[float] | None = None) -> NumericIntegrationResult:
    def integrand(t: float) -> complex:
        z = complex(z_of_t(t))
        dz = complex(dz_of_t(t))
        return evaluate_scalar(expr, z) * dz

    return _complex_quad(integrand, 0.0, 1.0, points=points)


def _integrate_ray(expr: str, start: complex, through: complex) -> NumericIntegrationResult:
    direction = through - start
    if abs(direction) < 1e-12:
        raise ValueError("Ray direction cannot be zero")
    direction /= abs(direction)

    def integrand(s: float) -> complex:
        z = start + direction * s
        return evaluate_scalar(expr, z) * direction

    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        real_val, real_err = integrate.quad(lambda s: float(np.real(integrand(s))), 0.0, np.inf, epsabs=1e-8, epsrel=1e-7, limit=250)
        imag_val, imag_err = integrate.quad(lambda s: float(np.imag(integrand(s))), 0.0, np.inf, epsabs=1e-8, epsrel=1e-7, limit=250)
    for item in caught:
        warning_messages.append(str(item.message))
    return NumericIntegrationResult(
        value=complex(real_val, imag_val),
        abs_error=float(real_err + imag_err),
        warnings=warning_messages,
        converged=len(warning_messages) == 0,
    )


def integrate_segment(expr: str, segment: dict[str, Any]) -> NumericIntegrationResult:
    kind = segment["type"]
    if kind == "line":
        start = to_complex(segment["start"])
        end = to_complex(segment["end"])
        return _integrate_line_like(expr, lambda t: start + t * (end - start), lambda _t: end - start)
    if kind == "arc":
        center = to_complex(segment["center"])
        start = to_complex(segment["start"])
        end = to_complex(segment["end"])
        radius = abs(start - center)
        if radius < 1e-12:
            raise ValueError("Arc radius cannot be zero")
        theta0 = np.angle(start - center)
        theta1 = np.angle(end - center)
        if bool(segment.get("ccw", True)):
            delta = (theta1 - theta0) % (2 * math.pi)
            delta = delta if delta > 1e-12 else 2 * math.pi
        else:
            delta = -((theta0 - theta1) % (2 * math.pi))
            delta = delta if delta < -1e-12 else -2 * math.pi
        return _integrate_line_like(
            expr,
            lambda t: center + radius * np.exp(1j * (theta0 + delta * t)),
            lambda t: 1j * delta * radius * np.exp(1j * (theta0 + delta * t)),
        )
    if kind == "circle":
        center = to_complex(segment["center"])
        start = to_complex(segment["start"])
        radius = abs(start - center)
        if radius < 1e-12:
            raise ValueError("Circle radius cannot be zero")
        theta0 = np.angle(start - center)
        delta = 2 * math.pi if bool(segment.get("ccw", True)) else -2 * math.pi
        return _integrate_line_like(
            expr,
            lambda t: center + radius * np.exp(1j * (theta0 + delta * t)),
            lambda t: 1j * delta * radius * np.exp(1j * (theta0 + delta * t)),
        )
    if kind == "quadratic":
        p0 = to_complex(segment["start"])
        p1 = to_complex(segment["control"])
        p2 = to_complex(segment["end"])
        return _integrate_line_like(
            expr,
            lambda t: (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2,
            lambda t: 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1),
        )
    if kind == "cubic":
        p0 = to_complex(segment["start"])
        p1 = to_complex(segment["control1"])
        p2 = to_complex(segment["control2"])
        p3 = to_complex(segment["end"])
        return _integrate_line_like(
            expr,
            lambda t: (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3,
            lambda t: 3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t * t * (p3 - p2),
        )
    if kind == "polyline":
        total = 0j
        error = 0.0
        warnings_out: list[str] = []
        points = [to_complex(p) for p in segment["points"]]
        for a, b in zip(points[:-1], points[1:]):
            res = _integrate_line_like(expr, lambda t, a=a, b=b: a + t * (b - a), lambda _t, a=a, b=b: b - a)
            total += res.value
            error += res.abs_error
            warnings_out.extend(res.warnings)
        return NumericIntegrationResult(total, error, warnings_out, len(warnings_out) == 0)
    if kind == "ray":
        return _integrate_ray(expr, to_complex(segment["start"]), to_complex(segment["through"]))
    raise ValueError(f"Unknown segment type: {kind}")


def _diag(bounds: tuple[float, float, float, float]) -> float:
    xmin, xmax, ymin, ymax = bounds
    return float(math.hypot(xmax - xmin, ymax - ymin))


def _known_family_candidates(features, bounds: tuple[float, float, float, float]) -> list[complex]:
    xmin, xmax, ymin, ymax = bounds
    candidates: list[complex] = []
    names = set(features.used_names)

    def add_if_inside(z: complex) -> None:
        if xmin - 1e-9 <= np.real(z) <= xmax + 1e-9 and ymin - 1e-9 <= np.imag(z) <= ymax + 1e-9:
            candidates.append(z)

    if {"gamma", "digamma", "psi"} & names:
        k_min = math.floor(min(0.0, xmin)) - 1
        k_max = math.ceil(max(0.0, xmax)) + 1
        for k in range(k_min, k_max + 1):
            if k <= 0:
                add_if_inside(complex(k, 0.0))
    if "zeta" in names:
        add_if_inside(1 + 0j)
    if {"tan", "sec"} & names:
        n_min = math.floor((xmin - math.pi / 2) / math.pi) - 1
        n_max = math.ceil((xmax - math.pi / 2) / math.pi) + 1
        for n in range(n_min, n_max + 1):
            add_if_inside(complex(math.pi / 2 + n * math.pi, 0.0))
    if {"cot", "csc"} & names:
        n_min = math.floor(xmin / math.pi) - 1
        n_max = math.ceil(xmax / math.pi) + 1
        for n in range(n_min, n_max + 1):
            add_if_inside(complex(n * math.pi, 0.0))

    deduped: list[complex] = []
    for z in candidates:
        if not any(abs(z - w) < 1e-7 for w in deduped):
            deduped.append(z)
    return deduped


def _denominator_roots(den_expr: str, bounds: tuple[float, float, float, float]) -> list[complex]:
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, 21)
    ys = np.linspace(ymin, ymax, 21)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    vals = evaluate(den_expr, Z)
    score = np.abs(vals)
    score[~np.isfinite(score)] = np.inf
    if not np.any(np.isfinite(score)):
        return []

    flat_order = np.argsort(score, axis=None)
    seeds: list[complex] = []
    for idx in flat_order[:18]:
        iy, ix = np.unravel_index(int(idx), score.shape)
        z0 = complex(X[iy, ix], Y[iy, ix])
        if not any(abs(z0 - s) < 0.2 * max((xmax - xmin) / 20, (ymax - ymin) / 20) for s in seeds):
            seeds.append(z0)

    roots: list[complex] = []
    for seed in seeds:
        def f_xy(v):
            z = complex(float(v[0]), float(v[1]))
            val = evaluate_scalar(den_expr, z)
            return [np.real(val), np.imag(val)]

        try:
            sol = optimize.root(f_xy, [np.real(seed), np.imag(seed)], method="hybr")
        except Exception:
            continue
        if not sol.success:
            continue
        z = complex(float(sol.x[0]), float(sol.x[1]))
        if not (xmin - 1e-6 <= np.real(z) <= xmax + 1e-6 and ymin - 1e-6 <= np.imag(z) <= ymax + 1e-6):
            continue
        val = evaluate_scalar(den_expr, z)
        if not np.isfinite(val) or abs(val) > 1e-5:
            continue
        if not any(abs(z - r) < 1e-5 for r in roots):
            roots.append(z)
    return roots


def _candidate_singularities(expr: str, bounds: tuple[float, float, float, float]) -> tuple[list[complex], Any]:
    features = analyze_expression(expr)
    candidates = _known_family_candidates(features, bounds)
    for den_expr in features.denominator_exprs:
        for root in _denominator_roots(den_expr, bounds):
            if not any(abs(root - existing) < 1e-5 for existing in candidates):
                candidates.append(root)
    return candidates, features


def _residue_by_small_circle(expr: str, center: complex, radius: float, samples: int = 720) -> complex | None:
    theta = np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    z = center + radius * np.exp(1j * theta)
    dz_dtheta = 1j * radius * np.exp(1j * theta)
    vals = evaluate(expr, z)
    integrand = vals * dz_dtheta
    if not np.all(np.isfinite(integrand)):
        return None
    dtheta = 2 * math.pi / samples
    integral = dtheta * np.sum(integrand)
    return complex(integral / (2j * math.pi))


def _sanitize_winding(w: float) -> int | None:
    if not np.isfinite(w):
        return None
    rounded = int(round(w))
    if abs(w - rounded) > 2e-2:
        return None
    return rounded


def _attempt_theorem(expr: str, path: list[dict[str, Any]], bounds: tuple[float, float, float, float], direct_value: complex | None = None):
    if not is_closed_path(path):
        return None

    candidates, features = _candidate_singularities(expr, bounds)
    if not features.theorem_eligible:
        return None

    if features.proven_entire:
        return {
            "method": "cauchy",
            "value": 0j,
            "residues": [],
            "notes": ["The expression is in the app's conservative entire-function class, so the closed-contour integral is 0."],
        }

    contour_tol = max(1e-3, 1e-3 * _diag(bounds))
    residue_items: list[ResidueInfo] = []
    notes: list[str] = []

    for point in candidates:
        winding = _sanitize_winding(winding_number(path, point, bounds))
        if winding in (None, 0):
            continue
        dist = distance_to_path(path, point, bounds)
        if dist <= contour_tol:
            notes.append(f"A candidate singularity at {point} lies on or too close to the contour, so theorem mode was skipped.")
            return None

        sep = [abs(point - other) for other in candidates if abs(point - other) > 1e-8]
        nearest_other = min(sep) if sep else 0.25 * _diag(bounds)
        radius = min(0.35 * dist, 0.3 * nearest_other, 0.08 * _diag(bounds))
        if radius <= 1e-5:
            return None
        residue = _residue_by_small_circle(expr, point, radius)
        if residue is None or not np.isfinite(residue):
            return None
        if abs(residue) < 1e-10:
            continue
        residue_items.append(ResidueInfo(point=point, winding=winding, residue=residue, radius=radius))

    if not residue_items:
        return None

    value = 2j * math.pi * sum(item.winding * item.residue for item in residue_items)
    if direct_value is not None and np.isfinite(direct_value):
        scale = max(1.0, abs(value), abs(direct_value))
        if abs(value - direct_value) > 5e-4 * scale:
            notes.append("Residue sum and direct quadrature disagreed beyond tolerance, so direct quadrature was kept.")
            return None

    return {
        "method": "residue",
        "value": value,
        "residues": [
            {
                "point": [float(np.real(item.point)), float(np.imag(item.point))],
                "point_labels": complex_component_labels(item.point),
                "winding": item.winding,
                "residue": [float(np.real(item.residue)), float(np.imag(item.residue))],
                "residue_labels": complex_component_labels(item.residue),
                "radius": item.radius,
            }
            for item in residue_items
        ],
        "notes": notes or ["Used the residue theorem with numerically estimated residues on small circles around enclosed singularities."],
    }


def _plot_traces(path: list[dict[str, Any]], bounds: tuple[float, float, float, float], residue_markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    xs, ys = path_xy_for_plot(path, bounds, points_per_segment=140)
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"width": 3},
            "name": "Path",
            "hoverinfo": "skip",
        }
    ]

    start, end = finite_endpoints(path)
    if start is not None:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [float(np.real(start))],
            "y": [float(np.imag(start))],
            "marker": {"size": 11, "symbol": "circle"},
            "name": "Start",
            "hovertemplate": "start<extra></extra>",
        })
    if end is not None:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [float(np.real(end))],
            "y": [float(np.imag(end))],
            "marker": {"size": 11, "symbol": "diamond"},
            "name": "End",
            "hovertemplate": "end<extra></extra>",
        })

    if residue_markers:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": [item["point"][0] for item in residue_markers],
            "y": [item["point"][1] for item in residue_markers],
            "marker": {"size": 11, "symbol": "x"},
            "text": [f"Res={item['residue'][0]:.4g}{item['residue'][1]:+.4g}i" for item in residue_markers],
            "textposition": "top center",
            "name": "Residues",
            "hoverinfo": "skip",
        })
    return traces


def _integration_result(
    *,
    title: str,
    message: str,
    summary: str,
    path: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    method: str,
    status: str,
    value: complex,
    abs_error: float,
    warning_messages: list[str],
    notes: list[str],
    residue_markers: list[dict[str, Any]],
    exact_value: str | None = None,
    exact_latex: str | None = None,
) -> dict[str, Any]:
    result = {
        "kind": "integration",
        "title": title,
        "message": message,
        "summary": summary,
        "closed": is_closed_path(path),
        "method": method,
        "status": status,
        "value": [float(np.real(value)), float(np.imag(value))],
        "value_labels": complex_component_labels(value),
        "abs_error": float(abs_error),
        "warnings": warning_messages,
        "notes": notes,
        "residues": residue_markers,
        "traces": _plot_traces(path, bounds, residue_markers),
        "xrange": [bounds[0], bounds[1]],
        "yrange": [bounds[2], bounds[3]],
    }
    if exact_value is not None:
        result["exact_value"] = exact_value
    if exact_latex is not None:
        result["exact_latex"] = exact_latex
    return result


def integrate_path(
    expr: str,
    path: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    use_theorem: bool = True,
    method_mode: str = "auto",
) -> dict[str, Any]:
    if not path:
        raise ValueError("Draw a path first.")
    if method_mode not in {"auto", "theorem", "numeric"}:
        raise ValueError("Integration method must be auto, theorem, or numeric.")

    summary = path_summary(path)
    path_points = sample_path(path, bounds, points_per_segment=180)
    if len(path_points) < 2:
        raise ValueError("The path is too short to integrate.")

    candidate_points, _features = _candidate_singularities(expr, bounds)
    contour_tol = max(1e-4, 1e-3 * _diag(bounds))
    for point in candidate_points:
        if distance_to_path(path, point, bounds) <= contour_tol:
            raise ValueError(f"A detected singularity at {point} lies on or too close to the path, so the integral is undefined for this app's ordinary contour integral mode.")

    sampled_values = evaluate(expr, path_points)
    if np.any(~np.isfinite(sampled_values)):
        raise ValueError("The integrand hits a non-finite value somewhere on the sampled path. The integral is undefined or needs a principal-value treatment, which this app does not do automatically.")

    if method_mode == "auto":
        exact = attempt_exact_integral(expr, path, bounds)
        if exact is not None:
            notes = exact["notes"]
            residue_markers = exact["residues"]
            return _integration_result(
                title="Contour / path integral",
                message=notes[0],
                summary=summary,
                path=path,
                bounds=bounds,
                method=exact["method"],
                status="ok",
                value=exact["value"],
                abs_error=0.0,
                warning_messages=[],
                notes=notes,
                residue_markers=residue_markers,
                exact_value=exact["exact_value"],
                exact_latex=exact.get("exact_latex"),
            )

    total = 0j
    total_err = 0.0
    warning_messages: list[str] = []
    for segment in path:
        seg_result = integrate_segment(expr, segment)
        total += seg_result.value
        total_err += seg_result.abs_error
        warning_messages.extend(seg_result.warnings)

    method = "numerical"
    theorem = None
    if use_theorem and method_mode in {"auto", "theorem"}:
        theorem = _attempt_theorem(expr, path, bounds, direct_value=total)
        if theorem is not None:
            method = theorem["method"]
            total = theorem["value"]

    status = "ok"
    if warning_messages:
        status = "warning"
    if not np.isfinite(total):
        status = "nonfinite"

    theorem_notes = theorem["notes"] if theorem else []
    residue_markers = theorem["residues"] if theorem and theorem["method"] == "residue" else []

    return _integration_result(
        title="Contour / path integral",
        message=theorem_notes[0] if theorem_notes else ("Integrated numerically along the supplied parameterized path."),
        summary=summary,
        path=path,
        bounds=bounds,
        method=method,
        status=status,
        value=total,
        abs_error=total_err,
        warning_messages=warning_messages,
        notes=theorem_notes,
        residue_markers=residue_markers,
    )
