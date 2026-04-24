from __future__ import annotations

import math
from typing import Any

import numpy as np

from .number_parsing import parse_real


def to_complex(point: Any) -> complex:
    if isinstance(point, complex):
        return point
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return complex(parse_real(point[0]), parse_real(point[1]))
    if isinstance(point, dict):
        return complex(parse_real(point["x"]), parse_real(point["y"]))
    raise ValueError(f"Cannot convert {point!r} to complex point")


def point_to_json(z: complex) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def project_arc_end(center: complex, start: complex, end: complex) -> complex:
    radius = abs(start - center)
    direction = end - center
    if radius < 1e-12:
        raise ValueError("Arc radius cannot be zero")
    if abs(direction) < 1e-12:
        raise ValueError("Arc end point cannot equal the center")
    return center + radius * direction / abs(direction)


def normalize_segment(segment: dict[str, Any]) -> dict[str, Any]:
    if segment.get("type") != "arc":
        return segment
    center = to_complex(segment["center"])
    start = to_complex(segment["start"])
    end = to_complex(segment["end"])
    normalized = dict(segment)
    normalized["end"] = point_to_json(project_arc_end(center, start, end))
    return normalized


def _angle(z: complex) -> float:
    return float(np.angle(z))


def _ccw_delta(theta0: float, theta1: float) -> float:
    delta = (theta1 - theta0) % (2 * math.pi)
    return float(delta if delta > 1e-12 else 2 * math.pi)


def _cw_delta(theta0: float, theta1: float) -> float:
    delta = -((theta0 - theta1) % (2 * math.pi))
    return float(delta if delta < -1e-12 else -2 * math.pi)


def segment_start(segment: dict[str, Any]) -> complex | None:
    kind = segment["type"]
    if kind in {"line", "ray", "arc", "circle", "quadratic", "cubic"}:
        return to_complex(segment["start"])
    if kind == "polyline":
        pts = [to_complex(p) for p in segment["points"]]
        return pts[0] if pts else None
    raise ValueError(f"Unknown segment type: {kind}")


def segment_end(segment: dict[str, Any]) -> complex | None:
    kind = segment["type"]
    if kind == "line":
        return to_complex(segment["end"])
    if kind == "ray":
        return None
    if kind == "arc":
        center = to_complex(segment["center"])
        start = to_complex(segment["start"])
        end = to_complex(segment["end"])
        return project_arc_end(center, start, end)
    if kind == "circle":
        return to_complex(segment["start"])
    if kind == "quadratic":
        return to_complex(segment["end"])
    if kind == "cubic":
        return to_complex(segment["end"])
    if kind == "polyline":
        pts = [to_complex(p) for p in segment["points"]]
        return pts[-1] if pts else None
    raise ValueError(f"Unknown segment type: {kind}")


def _normalize_direction(start: complex, through: complex) -> complex:
    direction = through - start
    if abs(direction) < 1e-12:
        raise ValueError("Ray direction cannot be zero")
    return direction / abs(direction)


def reverse_segment(segment: dict[str, Any]) -> dict[str, Any]:
    kind = segment["type"]
    if kind == "line":
        return {"type": "line", "start": segment["end"], "end": segment["start"]}
    if kind == "arc":
        center = to_complex(segment["center"])
        start = to_complex(segment["start"])
        end = segment_end(segment)
        return {
            "type": "arc",
            "center": segment["center"],
            "start": point_to_json(end if end is not None else to_complex(segment["end"])),
            "end": point_to_json(start),
            "ccw": not bool(segment.get("ccw", True)),
        }
    if kind == "circle":
        return {
            "type": "circle",
            "center": segment["center"],
            "start": segment["start"],
            "ccw": not bool(segment.get("ccw", True)),
        }
    if kind == "quadratic":
        return {
            "type": "quadratic",
            "start": segment["end"],
            "control": segment["control"],
            "end": segment["start"],
        }
    if kind == "cubic":
        return {
            "type": "cubic",
            "start": segment["end"],
            "control1": segment["control2"],
            "control2": segment["control1"],
            "end": segment["start"],
        }
    if kind == "polyline":
        return {"type": "polyline", "points": list(reversed(segment["points"]))}
    if kind == "ray":
        start = to_complex(segment["start"])
        through = to_complex(segment["through"])
        direction = _normalize_direction(start, through)
        fake_far = start - direction
        return {"type": "ray", "start": point_to_json(start), "through": point_to_json(fake_far)}
    raise ValueError(f"Unknown segment type: {kind}")


def reverse_path(path: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [reverse_segment(seg) for seg in reversed(path)]


def is_closed_path(path: list[dict[str, Any]], tol: float = 1e-6) -> bool:
    if not path:
        return False
    if len(path) == 1 and path[0]["type"] == "circle":
        return True
    start = segment_start(path[0])
    end = segment_end(path[-1])
    return bool(start is not None and end is not None and abs(start - end) <= tol)


def close_path_with_line(path: list[dict[str, Any]], tol: float = 1e-9) -> list[dict[str, Any]]:
    if not path:
        return path
    start = segment_start(path[0])
    end = segment_end(path[-1])
    if start is None or end is None:
        raise ValueError("Cannot close a path that ends at infinity")
    if abs(start - end) <= tol:
        return path
    return [*path, {"type": "line", "start": point_to_json(end), "end": point_to_json(start)}]


def _eval_line(segment: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start = to_complex(segment["start"])
    end = to_complex(segment["end"])
    z = start + t * (end - start)
    dz = np.full_like(t, end - start, dtype=np.complex128)
    return z, dz


def _eval_arc(segment: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = to_complex(segment["center"])
    start = to_complex(segment["start"])
    end = project_arc_end(center, start, to_complex(segment["end"]))
    theta0 = _angle(start - center)
    theta1 = _angle(end - center)
    radius = abs(start - center)
    if radius < 1e-12:
        raise ValueError("Arc radius cannot be zero")
    delta = _ccw_delta(theta0, theta1) if bool(segment.get("ccw", True)) else _cw_delta(theta0, theta1)
    theta = theta0 + delta * t
    exp_theta = np.exp(1j * theta)
    z = center + radius * exp_theta
    dz = 1j * delta * radius * exp_theta
    return z, dz


def _eval_circle(segment: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = to_complex(segment["center"])
    start = to_complex(segment["start"])
    radius = abs(start - center)
    if radius < 1e-12:
        raise ValueError("Circle radius cannot be zero")
    theta0 = _angle(start - center)
    delta = 2 * math.pi if bool(segment.get("ccw", True)) else -2 * math.pi
    theta = theta0 + delta * t
    exp_theta = np.exp(1j * theta)
    z = center + radius * exp_theta
    dz = 1j * delta * radius * exp_theta
    return z, dz


def _eval_quadratic(segment: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = to_complex(segment["start"])
    p1 = to_complex(segment["control"])
    p2 = to_complex(segment["end"])
    omt = 1 - t
    z = omt * omt * p0 + 2 * omt * t * p1 + t * t * p2
    dz = 2 * omt * (p1 - p0) + 2 * t * (p2 - p1)
    return z, dz


def _eval_cubic(segment: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = to_complex(segment["start"])
    p1 = to_complex(segment["control1"])
    p2 = to_complex(segment["control2"])
    p3 = to_complex(segment["end"])
    omt = 1 - t
    z = (
        omt**3 * p0
        + 3 * omt * omt * t * p1
        + 3 * omt * t * t * p2
        + t**3 * p3
    )
    dz = (
        3 * omt * omt * (p1 - p0)
        + 6 * omt * t * (p2 - p1)
        + 3 * t * t * (p3 - p2)
    )
    return z, dz


def _eval_polyline(segment: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = [to_complex(p) for p in segment["points"]]
    if len(points) < 2:
        raise ValueError("Polyline needs at least two points")
    subsegments = len(points) - 1
    scaled = np.clip(t * subsegments, 0, np.nextafter(subsegments, -np.inf))
    idx = np.floor(scaled).astype(int)
    local_t = scaled - idx
    start = np.asarray([points[i] for i in idx], dtype=np.complex128)
    end = np.asarray([points[i + 1] for i in idx], dtype=np.complex128)
    z = start + local_t * (end - start)
    dz = (end - start) * subsegments
    return z, dz


def _eval_ray_finite_preview(segment: dict[str, Any], t: np.ndarray, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    start = to_complex(segment["start"])
    through = to_complex(segment["through"])
    direction = _normalize_direction(start, through)
    xmin, xmax, ymin, ymax = bounds

    candidates: list[float] = []
    if abs(np.real(direction)) > 1e-12:
        candidates.extend([(xmin - np.real(start)) / np.real(direction), (xmax - np.real(start)) / np.real(direction)])
    if abs(np.imag(direction)) > 1e-12:
        candidates.extend([(ymin - np.imag(start)) / np.imag(direction), (ymax - np.imag(start)) / np.imag(direction)])

    positive = [c for c in candidates if c > 0]
    distance = max(positive) if positive else 4.0
    distance = max(distance, 1.0)
    z = start + t * distance * direction
    dz = np.full_like(t, distance * direction, dtype=np.complex128)
    return z, dz


def evaluate_segment(segment: dict[str, Any], t: np.ndarray, bounds: tuple[float, float, float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    kind = segment["type"]
    if kind == "line":
        return _eval_line(segment, t)
    if kind == "arc":
        return _eval_arc(segment, t)
    if kind == "circle":
        return _eval_circle(segment, t)
    if kind == "quadratic":
        return _eval_quadratic(segment, t)
    if kind == "cubic":
        return _eval_cubic(segment, t)
    if kind == "polyline":
        return _eval_polyline(segment, t)
    if kind == "ray":
        if bounds is None:
            raise ValueError("Ray preview requires bounds")
        return _eval_ray_finite_preview(segment, t, bounds)
    raise ValueError(f"Unknown segment type: {kind}")


def _sample_segment(segment: dict[str, Any], n: int, bounds: tuple[float, float, float, float]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    z, _ = evaluate_segment(segment, t, bounds=bounds)
    return np.asarray(z, dtype=np.complex128)


def sample_path(path: list[dict[str, Any]], bounds: tuple[float, float, float, float], points_per_segment: int = 120) -> np.ndarray:
    if not path:
        return np.asarray([], dtype=np.complex128)
    chunks = []
    for segment in path:
        n = points_per_segment
        if segment["type"] == "polyline":
            n = max(points_per_segment, 18 * (len(segment["points"]) - 1))
        pts = _sample_segment(segment, n, bounds)
        if chunks:
            pts = pts[1:]
        chunks.append(pts)
    return np.concatenate(chunks)


def path_xy_for_plot(path: list[dict[str, Any]], bounds: tuple[float, float, float, float], points_per_segment: int = 120) -> tuple[list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for segment in path:
        pts = _sample_segment(segment, points_per_segment, bounds)
        xs.extend([float(np.real(z)) for z in pts] + [None])
        ys.extend([float(np.imag(z)) for z in pts] + [None])
    return xs, ys


def winding_number(path: list[dict[str, Any]], point: complex, bounds: tuple[float, float, float, float]) -> float:
    if not is_closed_path(path):
        return 0.0
    pts = sample_path(path, bounds, points_per_segment=240)
    if len(pts) < 3:
        return 0.0
    shifted = pts - point
    if np.any(np.abs(shifted) < 1e-12):
        return float("nan")
    angles = np.unwrap(np.angle(shifted))
    delta = angles[-1] - angles[0]
    return float(delta / (2 * math.pi))


def distance_to_path(path: list[dict[str, Any]], point: complex, bounds: tuple[float, float, float, float]) -> float:
    pts = sample_path(path, bounds, points_per_segment=300)
    if len(pts) == 0:
        return float("inf")
    return float(np.min(np.abs(pts - point)))


def path_summary(path: list[dict[str, Any]]) -> str:
    if not path:
        return "Empty path."
    counts: dict[str, int] = {}
    for segment in path:
        counts[segment["type"]] = counts.get(segment["type"], 0) + 1
    bits = [f"{v} {k}" + ("s" if v != 1 else "") for k, v in sorted(counts.items())]
    closed = "closed" if is_closed_path(path) else "open"
    return f"{closed} path with " + ", ".join(bits) + "."


def finite_endpoints(path: list[dict[str, Any]]) -> tuple[complex | None, complex | None]:
    if not path:
        return None, None
    return segment_start(path[0]), segment_end(path[-1])
