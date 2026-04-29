from __future__ import annotations

import base64
import io
import math
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image
from scipy import optimize

from .expressions import evaluate, evaluate_scalar, mobius_analysis, singularity_points_in_bounds
from .number_labels import complex_component_labels


def make_grid(xmin: float, xmax: float, ymin: float, ymax: float, n: int) -> np.ndarray:
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return x_grid + 1j * y_grid


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    rgb = np.empty(h.shape + (3,), dtype=np.float64)
    conditions = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    choices = [
        np.stack([v, t, p], axis=-1),
        np.stack([q, v, p], axis=-1),
        np.stack([p, v, t], axis=-1),
        np.stack([p, q, v], axis=-1),
        np.stack([t, p, v], axis=-1),
        np.stack([v, p, q], axis=-1),
    ]
    for cond, choice in zip(conditions, choices):
        rgb[cond] = choice[cond]
    return rgb


def domain_coloring(w: np.ndarray) -> Image.Image:
    finite = np.isfinite(w)
    safe_w = np.where(finite, w, 0)
    hue = (np.angle(safe_w) + np.pi) / (2 * np.pi)
    mag = np.abs(safe_w)

    with np.errstate(all="ignore"):
        log_mag = np.log1p(np.clip(mag, 0.0, 1e300))
        bands = 0.72 + 0.28 * (0.5 + 0.5 * np.cos(2 * np.pi * log_mag))
        value = np.clip(1.0 - 1.0 / (1.0 + mag**0.35), 0, 1)
    value = (0.35 + 0.65 * value) * bands
    sat = np.full_like(value, 0.9)

    rgb = hsv_to_rgb(hue, sat, np.clip(value, 0, 1))
    rgb[~finite] = np.array([0.95, 0.97, 1.0])
    arr = np.clip(255 * rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def padded_range(xs: np.ndarray, ys: np.ndarray) -> tuple[list[float], list[float]]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return [-2, 2], [-2, 2]

    xf = xs[finite]
    yf = ys[finite]
    xmin, xmax = np.percentile(xf, [1, 99])
    ymin, ymax = np.percentile(yf, [1, 99])

    if xmin == xmax:
        xmin -= 1
        xmax += 1
    if ymin == ymax:
        ymin -= 1
        ymax += 1

    xpad = 0.08 * (xmax - xmin)
    ypad = 0.08 * (ymax - ymin)
    return [float(xmin - xpad), float(xmax + xpad)], [float(ymin - ypad), float(ymax + ypad)]


def _angle_delta(a: float, b: float) -> float:
    return float((b - a + math.pi) % (2 * math.pi) - math.pi)


def _cell_winding(values: list[complex]) -> int:
    if any(not np.isfinite(value) or abs(value) < 1e-14 for value in values):
        return 0
    angles = [float(np.angle(value)) for value in values]
    total = sum(_angle_delta(a, b) for a, b in zip(angles, angles[1:] + angles[:1]))
    winding = int(round(total / (2 * math.pi)))
    return winding if abs(total / (2 * math.pi) - winding) < 0.25 else 0


def _add_seed(seeds: list[complex], seed: complex, min_distance: float) -> None:
    if not np.isfinite(seed):
        return
    if any(abs(seed - existing) < min_distance for existing in seeds):
        return
    seeds.append(seed)


def _zero_seed_candidates(
    expr: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n: int,
) -> tuple[list[complex], float]:
    sample_nx = min(121, max(45, n // 3))
    sample_ny = min(121, max(45, n // 3))
    if sample_nx % 2 == 0:
        sample_nx += 1
    if sample_ny % 2 == 0:
        sample_ny += 1
    xs = np.linspace(xmin, xmax, sample_nx)
    ys = np.linspace(ymin, ymax, sample_ny)
    x_grid, y_grid = np.meshgrid(xs, ys)
    z_grid = x_grid + 1j * y_grid
    values = evaluate(expr, z_grid)
    score = np.abs(values)
    score[~np.isfinite(score)] = np.inf
    if not np.any(np.isfinite(score)):
        return [], 1e-8

    finite_scores = score[np.isfinite(score)]
    value_scale = max(1.0, float(np.percentile(finite_scores, 50)))
    root_tol = max(1e-8, 1e-6 * value_scale)
    dx = (xmax - xmin) / max(sample_nx - 1, 1)
    dy = (ymax - ymin) / max(sample_ny - 1, 1)
    seed_spacing = 0.55 * max(dx, dy)

    seeds: list[complex] = []

    # Argument-principle scan: positive phase winding around a cell is evidence of a zero inside.
    for iy in range(sample_ny - 1):
        for ix in range(sample_nx - 1):
            corners = [
                values[iy, ix],
                values[iy, ix + 1],
                values[iy + 1, ix + 1],
                values[iy + 1, ix],
            ]
            if _cell_winding(corners) > 0:
                _add_seed(seeds, complex(0.5 * (xs[ix] + xs[ix + 1]), 0.5 * (ys[iy] + ys[iy + 1])), seed_spacing)

    # Boundary and exact-grid zeros can make winding ambiguous, so keep exact hits too.
    for iy, ix in np.argwhere(score <= root_tol):
        _add_seed(seeds, complex(float(x_grid[iy, ix]), float(y_grid[iy, ix])), seed_spacing)

    # Add local minima of |f| across the whole grid, including boundaries.
    local_minima: list[tuple[float, complex]] = []
    for iy in range(sample_ny):
        y0 = max(0, iy - 1)
        y1 = min(sample_ny, iy + 2)
        for ix in range(sample_nx):
            current = score[iy, ix]
            if not np.isfinite(current):
                continue
            x0 = max(0, ix - 1)
            x1 = min(sample_nx, ix + 2)
            if current <= float(np.min(score[y0:y1, x0:x1])):
                local_minima.append((float(current), complex(float(x_grid[iy, ix]), float(y_grid[iy, ix]))))

    local_minima.sort(key=lambda item: item[0])
    for _score_value, seed in local_minima[:220]:
        _add_seed(seeds, seed, seed_spacing)

    return seeds, root_tol


def zero_marker_traces(expr: str, xmin: float, xmax: float, ymin: float, ymax: float, n: int) -> list[dict[str, Any]]:
    seeds, root_tol = _zero_seed_candidates(expr, xmin, xmax, ymin, ymax, n)
    if not seeds:
        return []

    bounds_tol = 0.02 * max(xmax - xmin, ymax - ymin, 1e-9)
    root_items: list[tuple[complex, complex, float]] = []

    for seed in seeds:
        try:
            seed_val = evaluate_scalar(expr, seed)
        except Exception:
            seed_val = np.inf
        if np.isfinite(seed_val) and abs(seed_val) <= root_tol:
            if not any(abs(seed - existing[0]) < 1e-5 for existing in root_items):
                root_items.append((seed, complex(seed_val), float(abs(seed_val))))
            continue

        def f_xy(v):
            z = complex(float(v[0]), float(v[1]))
            val = evaluate_scalar(expr, z)
            return [float(np.real(val)), float(np.imag(val))]

        try:
            sol = optimize.root(f_xy, [np.real(seed), np.imag(seed)], method="hybr")
        except Exception:
            continue
        if not sol.success:
            continue
        root = complex(float(sol.x[0]), float(sol.x[1]))
        if not (xmin - bounds_tol <= np.real(root) <= xmax + bounds_tol and ymin - bounds_tol <= np.imag(root) <= ymax + bounds_tol):
            continue
        try:
            val = evaluate_scalar(expr, root)
        except Exception:
            continue
        if not np.isfinite(val) or abs(val) > root_tol:
            continue
        if not any(abs(root - existing[0]) < 1e-5 for existing in root_items):
            root_items.append((root, complex(val), float(abs(val))))

    if not root_items:
        return []

    root_items.sort(key=lambda item: (float(np.imag(item[0])), float(np.real(item[0]))))
    root_labels = [complex_component_labels(root) for root, _value, _residual in root_items]
    return [{
        "type": "scatter",
        "mode": "markers",
        "x": [float(np.real(root)) for root, _value, _residual in root_items],
        "y": [float(np.imag(root)) for root, _value, _residual in root_items],
        "customdata": [
            [
                float(np.real(value)),
                float(np.imag(value)),
                residual,
                labels["re"]["label"] or "none",
                labels["im"]["label"] or "none",
            ]
            for labels, (_root, value, residual) in zip(root_labels, root_items)
        ],
        "marker": {
            "size": 15,
            "symbol": "circle-open-dot",
            "color": "#ff8f8f",
            "opacity": 0.98,
            "line": {"width": 2.6, "color": "#ff8f8f"},
        },
        "name": "zeros",
        "hoverinfo": "text",
        "hovertemplate": (
            "zero<br>"
            "z = %{x:.12g}%{y:+.12g}i<br>"
            "labels: Re %{customdata[3]}, Im %{customdata[4]}<br>"
            "f(z) = %{customdata[0]:.3e}%{customdata[1]:+.3e}i<br>"
            "|f(z)| = %{customdata[2]:.3e}<extra></extra>"
        ),
    }]


SINGULARITY_STYLES: dict[str, dict[str, Any]] = {
    "symbolic_pole": {"name": "poles", "label": "pole", "symbol": "x", "color": "#ff5d73", "size": 15},
    "symbolic_pole_candidate": {"name": "pole candidates", "label": "pole?", "symbol": "x-open", "color": "#fb7185", "size": 15},
    "symbolic_removable_singularity": {"name": "removable", "label": "rem", "symbol": "diamond-open", "color": "#7dd3fc", "size": 14},
    "symbolic_essential_candidate": {"name": "essential", "label": "ess", "symbol": "star-diamond", "color": "#f59e0b", "size": 16},
    "symbolic_branch_point_candidate": {"name": "branch points", "label": "branch", "symbol": "triangle-up", "color": "#c084fc", "size": 15},
}
DEFAULT_SINGULARITY_STYLE = {"name": "singularities", "label": "sing", "symbol": "cross", "color": "#f3c76b", "size": 14}


def singularity_marker_traces(expr: str, xmin: float, xmax: float, ymin: float, ymax: float) -> list[dict[str, Any]]:
    items = singularity_points_in_bounds(expr, (xmin, xmax, ymin, ymax))
    if not items:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item["kind"]), []).append(item)

    traces: list[dict[str, Any]] = []
    for kind, group in grouped.items():
        style = SINGULARITY_STYLES.get(kind, DEFAULT_SINGULARITY_STYLE)
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": [float(item["point"][0]) for item in group],
            "y": [float(item["point"][1]) for item in group],
            "text": [style["label"] for _item in group],
            "textposition": "bottom center",
            "customdata": [
                [
                    item.get("exact_point") or "",
                    item.get("kind") or "",
                    item.get("source") or "",
                    item.get("reason") or "",
                    item.get("pole_order") or "",
                ]
                for item in group
            ],
            "marker": {
                "size": style["size"],
                "symbol": style["symbol"],
                "color": style["color"],
                "opacity": 0.98,
                "line": {"width": 2.4, "color": style["color"]},
            },
            "textfont": {"color": style["color"], "size": 11, "family": "IBM Plex Mono, monospace"},
            "name": style["name"],
            "hovertemplate": (
                "singularity<br>"
                "z = %{x:.12g}%{y:+.12g}i<br>"
                "exact: %{customdata[0]}<br>"
                "kind: %{customdata[1]}<br>"
                "source: %{customdata[2]}<br>"
                "order: %{customdata[4]}<br>"
                "%{customdata[3]}<extra></extra>"
            ),
        })
    return traces


VECTOR_COLORS = ["#5f7cff", "#5aa9e6", "#78e0cf", "#9cf0bd", "#f3c76b", "#f59e5b", "#ff8f8f"]
VECTOR_COLORSCALE = [[i / (len(VECTOR_COLORS) - 1), color] for i, color in enumerate(VECTOR_COLORS)]


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _vector_color(log_length: float, low: float, high: float) -> str:
    if not math.isfinite(log_length) or high <= low:
        return VECTOR_COLORS[len(VECTOR_COLORS) // 2]
    t = np.clip((log_length - low) / (high - low), 0.0, 1.0)
    return VECTOR_COLORS[int(round(t * (len(VECTOR_COLORS) - 1)))]


def vector_segments(expr: str, xmin: float, xmax: float, ymin: float, ymax: float, stride: int, n: int, cap_scale: float):
    xs = np.linspace(xmin, xmax, max(2, n // stride))
    ys = np.linspace(ymin, ymax, max(2, n // stride))
    x_grid, y_grid = np.meshgrid(xs, ys)
    z = x_grid + 1j * y_grid
    w = evaluate(expr, z)

    dx = (xmax - xmin) / max(len(xs) - 1, 1)
    dy = (ymax - ymin) / max(len(ys) - 1, 1)
    cell = min(dx, dy) if dx > 0 and dy > 0 else max(dx, dy, 1.0)
    cap = max(1e-12, cap_scale * cell)

    finite = np.isfinite(w)
    raw_vectors = w - z
    raw_lengths = np.abs(raw_vectors)
    finite_lengths = raw_lengths[finite]
    if finite_lengths.size:
        log_lengths = np.log10(np.maximum(finite_lengths, 1e-12))
        color_low = float(np.percentile(log_lengths, 5))
        color_high = float(np.percentile(log_lengths, 95))
        if color_high <= color_low:
            color_high = color_low + 1.0
    else:
        color_low = -12.0
        color_high = 0.0

    line_bins: list[dict[str, Any]] = [
        {"x": [], "y": [], "color": color}
        for color in VECTOR_COLORS
    ]
    tip_x: list[float] = []
    tip_y: list[float] = []
    tip_angle: list[float] = []
    tip_color_value: list[float] = []
    tip_customdata: list[list[Any]] = []

    zero_x: list[float] = []
    zero_y: list[float] = []
    capped_count = 0

    for z0, w0, raw_vec, raw_len in zip(z.ravel(), w.ravel(), raw_vectors.ravel(), raw_lengths.ravel()):
        if not np.isfinite(w0) or not math.isfinite(float(raw_len)):
            continue
        if raw_len <= 1e-12:
            zero_x.append(float(np.real(z0)))
            zero_y.append(float(np.imag(z0)))
            continue

        display_len = min(float(raw_len), cap)
        capped = raw_len > cap
        if capped:
            capped_count += 1
        display_vec = raw_vec / raw_len * display_len
        tip = z0 + display_vec
        log_len = float(np.log10(max(float(raw_len), 1e-12)))
        bin_index = int(round(np.clip((log_len - color_low) / (color_high - color_low), 0.0, 1.0) * (len(VECTOR_COLORS) - 1)))
        line_bins[bin_index]["x"].extend([float(np.real(z0)), float(np.real(tip)), None])
        line_bins[bin_index]["y"].extend([float(np.imag(z0)), float(np.imag(tip)), None])

        tip_x.append(float(np.real(tip)))
        tip_y.append(float(np.imag(tip)))
        tip_angle.append(float(np.degrees(np.angle(display_vec)) - 90.0))
        tip_color_value.append(log_len)
        tip_customdata.append([
            float(np.real(z0)),
            float(np.imag(z0)),
            _finite_or_none(np.real(w0)),
            _finite_or_none(np.imag(w0)),
            float(raw_len),
            float(display_len),
            bool(capped),
        ])

    traces: list[dict[str, Any]] = []
    for item in line_bins:
        if not item["x"]:
            continue
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": item["x"],
            "y": item["y"],
            "line": {"width": 1.35, "color": item["color"]},
            "hoverinfo": "skip",
            "showlegend": False,
        })

    if tip_x:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": tip_x,
            "y": tip_y,
            "customdata": tip_customdata,
            "marker": {
                "size": 9,
                "symbol": "triangle-up",
                "angle": tip_angle,
                "color": tip_color_value,
                "colorscale": VECTOR_COLORSCALE,
                "cmin": color_low,
                "cmax": color_high,
                "line": {"color": "rgba(4,12,13,0.85)", "width": 0.75},
                "colorbar": {"title": "log10 |f(z)-z|", "len": 0.72},
            },
            "name": "capped arrows",
            "hovertemplate": (
                "z = %{customdata[0]:.5g}%{customdata[1]:+.5g}i<br>"
                "f(z) = %{customdata[2]:.5g}%{customdata[3]:+.5g}i<br>"
                "|f(z)-z| = %{customdata[4]:.5g}<br>"
                "drawn length = %{customdata[5]:.5g}<br>"
                "capped = %{customdata[6]}<extra></extra>"
            ),
        })

    if zero_x:
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": zero_x,
            "y": zero_y,
            "marker": {"size": 4, "color": "rgba(214,235,229,0.72)"},
            "name": "fixed points",
            "hovertemplate": "f(z) = z<extra></extra>",
        })

    return traces, [xmin, xmax], [ymin, ymax], capped_count


def make_line_segments(points: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    x: list[float | None] = []
    y: list[float | None] = []
    for line in points:
        finite = np.isfinite(line)
        cur_x: list[float] = []
        cur_y: list[float] = []
        for ok, z0 in zip(finite, line):
            if ok:
                cur_x.append(float(np.real(z0)))
                cur_y.append(float(np.imag(z0)))
            elif cur_x:
                x.extend(cur_x + [None])
                y.extend(cur_y + [None])
                cur_x = []
                cur_y = []
        if cur_x:
            x.extend(cur_x + [None])
            y.extend(cur_y + [None])
    return x, y


HIGHLIGHT_CURVE_COLORS = ["#f3c76b", "#ff8f8f", "#7dd3fc"]


def _highlight_curve_points(
    kind: str,
    a: float,
    b: float,
    c: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    points_per_curve: int,
) -> list[tuple[str, np.ndarray]]:
    sample_count = max(60, points_per_curve)
    if kind == "none":
        return []
    if kind == "vertical":
        y_vals = np.linspace(ymin, ymax, sample_count)
        return [(f"x = {a:.6g}", a + 1j * y_vals)]
    if kind == "horizontal":
        x_vals = np.linspace(xmin, xmax, sample_count)
        return [(f"y = {a:.6g}", x_vals + 1j * a)]
    if kind == "diagonal":
        slope = a
        intercept = b
        points: list[tuple[float, float]] = []
        for x in (xmin, xmax):
            y = slope * x + intercept
            if ymin <= y <= ymax:
                points.append((x, y))
        if abs(slope) > 1e-12:
            for y in (ymin, ymax):
                x = (y - intercept) / slope
                if xmin <= x <= xmax:
                    points.append((x, y))
        unique_points: list[tuple[float, float]] = []
        for point in points:
            if not any(math.hypot(point[0] - existing[0], point[1] - existing[1]) < 1e-9 for existing in unique_points):
                unique_points.append(point)
        if len(unique_points) >= 2:
            unique_points.sort(key=lambda point: (point[0], point[1]))
            start, end = unique_points[0], unique_points[-1]
            x_vals = np.linspace(start[0], end[0], sample_count)
            y_vals = np.linspace(start[1], end[1], sample_count)
        else:
            x_vals = np.linspace(xmin, xmax, sample_count)
            y_vals = slope * x_vals + intercept
        return [(f"y = {slope:.6g}x {intercept:+.6g}", x_vals + 1j * y_vals)]
    if kind == "circle":
        center = complex(a, b)
        radius = max(abs(c), 1e-9)
        angles = np.linspace(0, 2 * math.pi, sample_count)
        label = f"|z| = {radius:.6g}" if abs(center) < 1e-12 else f"|z - ({a:.6g}{b:+.6g}i)| = {radius:.6g}"
        return [(label, center + radius * np.exp(1j * angles))]
    if kind == "axes":
        x_vals = np.linspace(xmin, xmax, sample_count)
        y_vals = np.linspace(ymin, ymax, sample_count)
        return [
            ("real axis", x_vals + 0j),
            ("imaginary axis", 1j * y_vals),
        ]
    return []


def _line_trace(points: np.ndarray, *, name: str, color: str, width: float, hover: str = "skip") -> dict[str, Any]:
    x, y = make_line_segments(np.asarray([points], dtype=np.complex128))
    return {
        "type": "scatter",
        "mode": "lines",
        "x": x,
        "y": y,
        "line": {"width": width, "color": color},
        "name": name,
        "hoverinfo": hover,
    }


def transform_frames(
    expr: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    frame_count: int,
    line_count: int,
    points_per_line: int,
    highlight_kind: str = "none",
    highlight_a: float = 0.0,
    highlight_b: float = 0.0,
    highlight_c: float = 1.0,
):
    xs = np.linspace(xmin, xmax, line_count)
    ys = np.linspace(ymin, ymax, line_count)

    lines = []
    for x in xs:
        y_vals = np.linspace(ymin, ymax, points_per_line)
        lines.append(x + 1j * y_vals)
    for y in ys:
        x_vals = np.linspace(xmin, xmax, points_per_line)
        lines.append(x_vals + 1j * y)

    z = np.array(lines, dtype=np.complex128)
    w = evaluate(expr, z)
    finite_w = np.isfinite(w)
    highlight_curves = _highlight_curve_points(highlight_kind, highlight_a, highlight_b, highlight_c, xmin, xmax, ymin, ymax, points_per_line)
    highlighted: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name, curve_z in highlight_curves:
        curve_w = evaluate(expr, curve_z)
        highlighted.append((name, curve_z, curve_w, np.isfinite(curve_w)))

    if np.any(finite_w):
        x_parts = [np.real(z).ravel(), np.real(w[finite_w]).ravel()]
        y_parts = [np.imag(z).ravel(), np.imag(w[finite_w]).ravel()]
    else:
        x_parts = [np.real(z).ravel()]
        y_parts = [np.imag(z).ravel()]
    for _name, curve_z, curve_w, finite_curve_w in highlighted:
        x_parts.append(np.real(curve_z).ravel())
        y_parts.append(np.imag(curve_z).ravel())
        if np.any(finite_curve_w):
            x_parts.append(np.real(curve_w[finite_curve_w]).ravel())
            y_parts.append(np.imag(curve_w[finite_curve_w]).ravel())

    all_x = np.concatenate(x_parts)
    all_y = np.concatenate(y_parts)
    xrange, yrange = padded_range(all_x, all_y)
    base_x, base_y = make_line_segments(z)
    traces = [{
        "type": "scatter",
        "mode": "lines",
        "x": base_x,
        "y": base_y,
        "line": {"width": 1.35, "color": "#78e0cf"},
        "name": "input grid",
        "hoverinfo": "skip",
    }]
    for index, (name, curve_z, _curve_w, _finite_curve_w) in enumerate(highlighted):
        traces.append(_line_trace(
            curve_z,
            name=name,
            color=HIGHLIGHT_CURVE_COLORS[index % len(HIGHLIGHT_CURVE_COLORS)],
            width=4.0,
        ))

    frames = []
    for t in np.linspace(0.0, 1.0, frame_count):
        if t == 0.0:
            p = z
        else:
            p = np.where(finite_w, (1 - t) * z + t * w, np.nan + 1j * np.nan)
        px, py = make_line_segments(p)
        frame_data = [{
            "type": "scatter",
            "mode": "lines",
            "x": px,
            "y": py,
            "line": {"width": 1.35, "color": "#78e0cf"},
            "name": "input grid",
            "hoverinfo": "skip",
        }]
        for index, (name, curve_z, curve_w, finite_curve_w) in enumerate(highlighted):
            if t == 0.0:
                curve_p = curve_z
            else:
                curve_p = np.where(finite_curve_w, (1 - t) * curve_z + t * curve_w, np.nan + 1j * np.nan)
            frame_data.append(_line_trace(
                curve_p,
                name=name,
                color=HIGHLIGHT_CURVE_COLORS[index % len(HIGHLIGHT_CURVE_COLORS)],
                width=4.0,
            ))
        frames.append({
            "name": f"{t:.2f}",
            "data": frame_data,
        })
    return traces, frames, xrange, yrange


@lru_cache(maxsize=24)
def compute_plot_cached(
    expr: str,
    mode: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n: int,
    stride: int,
    frame_count: int,
    vector_cap: float,
    grid_lines: int,
    grid_samples: int,
    highlight_zeros: bool,
    show_singularities: bool,
    transform_highlight: str = "none",
    transform_highlight_a: float = 0.0,
    transform_highlight_b: float = 0.0,
    transform_highlight_c: float = 1.0,
) -> dict[str, Any]:
    if not all(math.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
        raise ValueError("Bounds must be finite numbers")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Bounds must satisfy xmin < xmax and ymin < ymax")
    if not (100 <= n <= 800):
        raise ValueError("Resolution must be between 100 and 800")
    if not (6 <= stride <= 80):
        raise ValueError("Vector stride must be between 6 and 80")
    if not (10 <= frame_count <= 120):
        raise ValueError("Animation frames must be between 10 and 120")
    if not (0.15 <= vector_cap <= 2.5):
        raise ValueError("Vector cap must be between 0.15 and 2.5")
    if not (3 <= grid_lines <= 41):
        raise ValueError("Transform grid lines must be between 3 and 41")
    if not (20 <= grid_samples <= 500):
        raise ValueError("Transform samples per line must be between 20 and 500")
    if transform_highlight not in {"none", "vertical", "horizontal", "diagonal", "circle", "axes"}:
        raise ValueError("Unknown transform highlight curve")
    if not all(math.isfinite(v) for v in (transform_highlight_a, transform_highlight_b, transform_highlight_c)):
        raise ValueError("Transform highlight values must be finite")

    if mode == "colors":
        z = make_grid(xmin, xmax, ymax, ymin, n)
        w = evaluate(expr, z)
        img = domain_coloring(w)
        finite_ratio = float(np.mean(np.isfinite(w)))
        message = "Domain coloring: hue = argument, brightness varies with magnitude."
        if finite_ratio < 0.995:
            message += f" Non-finite values are shown in pale blue ({finite_ratio:.1%} finite samples)."
        return {
            "kind": "image",
            "data_uri": image_to_data_uri(img),
            "zero_traces": zero_marker_traces(expr, xmin, xmax, ymin, ymax, n) if highlight_zeros else [],
            "singularity_traces": singularity_marker_traces(expr, xmin, xmax, ymin, ymax) if show_singularities else [],
            "xrange": [xmin, xmax],
            "yrange": [ymin, ymax],
            "title": "Domain coloring",
            "message": message,
        }

    if mode == "vectors":
        traces, xrange, yrange, capped_count = vector_segments(expr, xmin, xmax, ymin, ymax, stride, n, vector_cap)
        if highlight_zeros:
            traces.extend(zero_marker_traces(expr, xmin, xmax, ymin, ymax, n))
        if show_singularities:
            traces.extend(singularity_marker_traces(expr, xmin, xmax, ymin, ymax))
        message = "Arrows show z -> f(z), capped to keep the field readable; color encodes original displacement length."
        if capped_count:
            message += f" {capped_count} arrows were capped."
        return {
            "kind": "vectors",
            "traces": traces,
            "xrange": xrange,
            "yrange": yrange,
            "title": "Capped vector map: z -> f(z)",
            "message": message,
        }

    if mode == "transform":
        traces, frames, xrange, yrange = transform_frames(
            expr,
            xmin,
            xmax,
            ymin,
            ymax,
            frame_count,
            grid_lines,
            grid_samples,
            transform_highlight,
            transform_highlight_a,
            transform_highlight_b,
            transform_highlight_c,
        )
        mobius = mobius_analysis(expr)
        message = f"Press Play to move the {grid_lines}x{grid_lines} input grid to its image."
        if transform_highlight != "none":
            message += " Highlighted curves are drawn thicker."
        if mobius:
            message += f" Detected {mobius['label']} LFT."
        return {
            "kind": "transform",
            "traces": traces,
            "frames": frames,
            "xrange": xrange,
            "yrange": yrange,
            "title": "Grid homotopy: points move from z to its image.",
            "message": message,
            "mobius": mobius,
        }

    raise ValueError(f"Unknown mode: {mode}")
