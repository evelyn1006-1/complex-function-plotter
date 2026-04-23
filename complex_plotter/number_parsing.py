from __future__ import annotations

import math
from fractions import Fraction
from typing import Any


def parse_real(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numbers.")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty value is not a number.")
        try:
            if "/" in text:
                parts = text.split("/")
                if len(parts) != 2:
                    raise ValueError
                numerator = float(Fraction(parts[0].strip()))
                denominator = float(Fraction(parts[1].strip()))
                if denominator == 0:
                    raise ZeroDivisionError
                parsed = numerator / denominator
            else:
                parsed = float(Fraction(text))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid numeric value: {value!r}.") from exc
    else:
        raise ValueError(f"Invalid numeric value: {value!r}.")

    if not math.isfinite(parsed):
        raise ValueError(f"Numeric value must be finite: {value!r}.")
    return parsed
