from __future__ import annotations

import math
from fractions import Fraction
from typing import Any


def _format_fraction(frac: Fraction) -> str:
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def _format_pi_fraction(frac: Fraction) -> str:
    if frac.numerator == 0:
        return "0"
    sign = "-" if frac.numerator < 0 else ""
    numerator = abs(frac.numerator)
    if frac.denominator == 1:
        coeff = "" if numerator == 1 else str(numerator)
        return f"{sign}{coeff}pi"
    coeff = "" if numerator == 1 else str(numerator)
    return f"{sign}{coeff}pi/{frac.denominator}"


def near_exact_label(value: float, *, scale: float = 1.0) -> str | None:
    if not math.isfinite(value):
        return None

    tolerance = max(1e-9, 5e-8 * max(1.0, abs(value), scale))

    rational = Fraction(value).limit_denominator(100)
    rational_value = rational.numerator / rational.denominator
    if abs(value - rational_value) <= tolerance:
        return _format_fraction(rational)

    pi_ratio = value / math.pi
    pi_fraction = Fraction(pi_ratio).limit_denominator(20)
    pi_value = math.pi * pi_fraction.numerator / pi_fraction.denominator
    if abs(value - pi_value) <= tolerance:
        return _format_pi_fraction(pi_fraction)

    return None


def component_label(value: float, *, scale: float = 1.0) -> dict[str, Any]:
    label = near_exact_label(value, scale=scale)
    return {"value": float(value), "label": label}


def complex_component_labels(pair: complex | list[float] | tuple[float, float]) -> dict[str, Any]:
    if isinstance(pair, complex):
        re = float(pair.real)
        im = float(pair.imag)
    else:
        re = float(pair[0])
        im = float(pair[1])
    scale = max(1.0, abs(re), abs(im))
    return {
        "re": component_label(re, scale=scale),
        "im": component_label(im, scale=scale),
    }
