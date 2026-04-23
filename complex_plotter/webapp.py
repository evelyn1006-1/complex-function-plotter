from __future__ import annotations

import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .expressions import classify_expression, evaluate_scalar
from .integration import integrate_path
from .number_parsing import parse_real
from .plotting import compute_plot_cached

# ==========================
# Configuration / constants
# ==========================

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

ALLOWED_HOSTS = {
    h.strip().lower()
    for h in (os.getenv("ALLOWED_HOSTS", "complex.princessevelyn.com,localhost,127.0.0.1,::1")).split(",")
    if h.strip()
}

if not FLASK_SECRET_KEY:
    raise RuntimeError("Missing FLASK_SECRET_KEY in environment or .env")

# ==========================
# Security helpers
# ==========================

def get_request_host():
    """Extract host from request, stripping port if present."""
    host = request.host or ""
    if host.startswith("[") and "]" in host:
        return host[1:host.index("]")].lower()
    return host.rsplit(":", 1)[0].lower()


def is_safe_host(host):
    """Check if the host is in our allowed list."""
    return host in ALLOWED_HOSTS


def json_payload() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object request body.")
    return data


def finite_float(data: dict[str, Any], key: str, default: float) -> float:
    raw = data.get(key, default)
    try:
        value = parse_real(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number.") from exc
    return value


def bounded_int(data: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    raw = data.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be an integer.")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{key} must be an integer.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return value


def bounded_float(data: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    value = finite_float(data, key, default)
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return value


def bounds_from_mapping(data: dict[str, Any]) -> tuple[float, float, float, float]:
    xmin = finite_float(data, "xmin", -2)
    xmax = finite_float(data, "xmax", 2)
    ymin = finite_float(data, "ymin", -2)
    ymax = finite_float(data, "ymax", 2)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Bounds must satisfy xmin < xmax and ymin < ymax.")
    return xmin, xmax, ymin, ymax


def path_from_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    path = data.get("path", [])
    if not isinstance(path, list):
        raise ValueError("path must be a list of path segments.")
    if len(path) > 80:
        raise ValueError("Paths are limited to 80 segments.")
    allowed_types = {"line", "arc", "circle", "quadratic", "cubic", "polyline", "ray"}
    for segment in path:
        if not isinstance(segment, dict):
            raise ValueError("Each path segment must be an object.")
        if not isinstance(segment.get("type"), str):
            raise ValueError("Each path segment needs a type.")
        if segment["type"] not in allowed_types:
            raise ValueError(f"Unknown path segment type: {segment['type']}.")
        if segment["type"] == "polyline":
            points = segment.get("points", [])
            if not isinstance(points, list):
                raise ValueError("Polyline points must be a list.")
            if len(points) > 3000:
                raise ValueError("Freeform polylines are limited to 3000 points.")
    return path


# ==========================
# App setup
# ==========================

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# ----------------------------------------------------------
# Secure session configuration
# ----------------------------------------------------------
app.config.update(
    SESSION_COOKIE_SECURE=True,       # Only send cookie over HTTPS
    SESSION_COOKIE_HTTPONLY=True,     # JavaScript cannot access cookie
    SESSION_COOKIE_SAMESITE="Lax",    # CSRF protection
    PERMANENT_SESSION_LIFETIME=timedelta(hours=2),
    MAX_CONTENT_LENGTH=1_000_000,
)

# ----------------------------------------------------------
# Security middleware
# ----------------------------------------------------------

@app.before_request
def enforce_host_check():
    """Block requests with invalid Host headers (DNS rebinding protection)."""
    if not is_safe_host(get_request_host()):
        abort(400)


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/plot")
def plot():
    try:
        data = json_payload()
        xmin, xmax, ymin, ymax = bounds_from_mapping(data)
        result = compute_plot_cached(
            str(data.get("expr", "z")),
            str(data.get("mode", "colors")),
            xmin,
            xmax,
            ymin,
            ymax,
            bounded_int(data, "n", 350, 100, 800),
            bounded_int(data, "stride", 18, 6, 80),
            bounded_int(data, "frames", 28, 10, 120),
            bounded_float(data, "vector_cap", 0.72, 0.15, 2.5),
            bounded_int(data, "grid_lines", 9, 3, 41),
            bounded_int(data, "grid_samples", 120, 20, 500),
            bool(data.get("highlight_zeros", False)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/evaluate")
def evaluate_point():
    try:
        data = json_payload()
        x = finite_float(data, "x", 0)
        y = finite_float(data, "y", 0)
        value = evaluate_scalar(str(data.get("expr", "z")), complex(x, y))
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
        result = {
            "kind": "evaluation",
            "z": [x, y],
            "finite": finite,
            "value": [value.real if finite else None, value.imag if finite else None],
            "abs": abs(value) if finite else None,
            "arg": math.atan2(value.imag, value.real) if finite else None,
        }
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/classify")
def classify():
    try:
        data = json_payload()
        return jsonify(classify_expression(str(data.get("expr", "z"))))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/integrate")
def integrate():
    try:
        data = json_payload()
        bounds_dict = data.get("bounds", {})
        if not isinstance(bounds_dict, dict):
            raise ValueError("bounds must be an object.")
        bounds = bounds_from_mapping(bounds_dict)
        result = integrate_path(
            str(data.get("expr", "z")),
            path_from_payload(data),
            bounds,
            bool(data.get("use_theorem", True)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ----------------------------------------------------------
# Error handlers
# ----------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500
