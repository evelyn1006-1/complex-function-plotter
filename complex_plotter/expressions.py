from __future__ import annotations

import ast
import io
import math
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import special as sp


def sec(z: np.ndarray) -> np.ndarray:
    return 1 / np.cos(z)


def csc(z: np.ndarray) -> np.ndarray:
    return 1 / np.sin(z)


def cot(z: np.ndarray) -> np.ndarray:
    return 1 / np.tan(z)


def sech(z: np.ndarray) -> np.ndarray:
    return 1 / np.cosh(z)


def csch(z: np.ndarray) -> np.ndarray:
    return 1 / np.sinh(z)


def coth(z: np.ndarray) -> np.ndarray:
    return 1 / np.tanh(z)


def ln(z: np.ndarray) -> np.ndarray:
    return np.log(z)


def conj(z: np.ndarray) -> np.ndarray:
    return np.conjugate(z)


def real(z: np.ndarray) -> np.ndarray:
    return np.real(z)


def imag(z: np.ndarray) -> np.ndarray:
    return np.imag(z)


def angle(z: np.ndarray) -> np.ndarray:
    return np.angle(z)


def cis(z: np.ndarray) -> np.ndarray:
    return np.exp(1j * z)


def re(z: np.ndarray) -> np.ndarray:
    return np.real(z)


def im(z: np.ndarray) -> np.ndarray:
    return np.imag(z)


def _as_bool_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=bool)


def where_fn(condition: Any, when_true: Any, when_false: Any) -> np.ndarray:
    return np.where(_as_bool_array(condition), when_true, when_false)


def piecewise_fn(*args: Any) -> np.ndarray:
    if len(args) < 3 or len(args) % 2 == 0:
        raise ValueError("piecewise expects cond1, value1, ..., default")
    conditions = [_as_bool_array(args[i]) for i in range(0, len(args) - 1, 2)]
    choices = [np.asarray(args[i], dtype=np.complex128) for i in range(1, len(args) - 1, 2)]
    default = np.asarray(args[-1], dtype=np.complex128)
    return np.select(conditions, choices, default=default).astype(np.complex128)


def and_fn(a: Any, b: Any) -> np.ndarray:
    return np.logical_and(_as_bool_array(a), _as_bool_array(b))


def or_fn(a: Any, b: Any) -> np.ndarray:
    return np.logical_or(_as_bool_array(a), _as_bool_array(b))


def not_fn(a: Any) -> np.ndarray:
    return np.logical_not(_as_bool_array(a))


def fresnels(z: np.ndarray) -> np.ndarray:
    return sp.fresnel(z)[0]


def fresnelc(z: np.ndarray) -> np.ndarray:
    return sp.fresnel(z)[1]


def airy_ai(z: np.ndarray) -> np.ndarray:
    return sp.airy(z)[0]


def airy_aip(z: np.ndarray) -> np.ndarray:
    return sp.airy(z)[1]


def airy_bi(z: np.ndarray) -> np.ndarray:
    return sp.airy(z)[2]


def airy_bip(z: np.ndarray) -> np.ndarray:
    return sp.airy(z)[3]


ALLOWED: dict[str, Any] = {
    "z": None,
    "x": None,
    "y": None,
    "pi": np.pi,
    "e": np.e,
    "tau": math.tau,
    "i": 1j,
    "j": 1j,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "exp": np.exp,
    "log": np.log,
    "ln": ln,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sec": sec,
    "csc": csc,
    "cot": cot,
    "sech": sech,
    "csch": csch,
    "coth": coth,
    "conj": conj,
    "conjugate": np.conjugate,
    "real": real,
    "imag": imag,
    "re": re,
    "im": im,
    "angle": angle,
    "arg": angle,
    "sinc": np.sinc,
    "cis": cis,
    "where": where_fn,
    "piecewise": piecewise_fn,
    "and_": and_fn,
    "or_": or_fn,
    "not_": not_fn,
    "gamma": sp.gamma,
    "loggamma": sp.loggamma,
    "rgamma": sp.rgamma,
    "digamma": sp.digamma,
    "psi": sp.digamma,
    "erf": sp.erf,
    "erfc": sp.erfc,
    "erfi": sp.erfi,
    "erfcx": sp.erfcx,
    "wofz": sp.wofz,
    "dawsn": sp.dawsn,
    "zeta": sp.zeta,
    "zetac": sp.zetac,
    "lambertw": sp.lambertw,
    "expi": sp.expi,
    "exp1": sp.exp1,
    "fresnels": fresnels,
    "fresnelc": fresnelc,
    "airyai": airy_ai,
    "airyaip": airy_aip,
    "airybi": airy_bi,
    "airybip": airy_bip,
    "jv": sp.jv,
    "yv": sp.yv,
    "iv": sp.iv,
    "kv": sp.kv,
}


BRANCHY_NAMES = {
    "log",
    "ln",
    "log10",
    "sqrt",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "lambertw",
    "loggamma",
    "exp1",
    "expi",
    "kv",
    "yv",
}
NONANALYTIC_NAMES = {
    "x",
    "y",
    "conj",
    "conjugate",
    "real",
    "imag",
    "re",
    "im",
    "abs",
    "angle",
    "arg",
    "where",
    "piecewise",
}
KNOWN_POLE_FAMILIES = {
    "gamma",
    "digamma",
    "psi",
    "tan",
    "sec",
    "cot",
    "csc",
    "zeta",
}
ENTIRE_FUNCTIONS = {
    "sin",
    "cos",
    "sinh",
    "cosh",
    "exp",
    "erf",
    "erfc",
    "erfi",
    "erfcx",
    "wofz",
    "dawsn",
    "fresnels",
    "fresnelc",
    "airyai",
    "airyaip",
    "airybi",
    "airybip",
    "iv",
    "jv",
    "rgamma",
}
SAFE_CONSTANTS = {"z", "pi", "e", "tau", "i", "j"}
VALUE_NAMES = SAFE_CONSTANTS | {"x", "y"}
CALLABLE_NAMES = set(ALLOWED) - VALUE_NAMES


class SafeExpressionValidator(ast.NodeVisitor):
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.BitAnd,
        ast.BitOr,
        ast.Invert,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    }

    def __init__(self, allowed_names: set[str]) -> None:
        self.allowed_names = allowed_names

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in self.allowed_nodes:
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise ValueError(f"Unknown name: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        if node.func.id not in self.allowed_names:
            raise ValueError(f"Unknown function: {node.func.id}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        for arg in node.args:
            self.visit(arg)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) != 1:
            raise ValueError("Chained comparisons are not supported; use (a < b) & (b < c).")
        if not isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            raise ValueError(f"Unsupported comparison: {type(node.ops[0]).__name__}")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)


class NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)


class DenominatorCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.denominators: list[str] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div):
            self.denominators.append(ast.unparse(node.right))
        elif isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub):
                self.denominators.append(ast.unparse(node.left))
            elif isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.value < 0:
                self.denominators.append(ast.unparse(node.left))
        self.generic_visit(node)


@dataclass(frozen=True)
class ExpressionFeatures:
    expr: str
    used_names: frozenset[str]
    denominator_exprs: tuple[str, ...]
    has_branchy: bool
    has_nonanalytic: bool
    has_piecewise: bool
    has_denominator: bool
    uses_known_pole_families: bool
    theorem_eligible: bool
    proven_entire: bool


def _cauchy_riemann_check(expr: str) -> dict[str, Any]:
    sample = np.asarray(
        [
            -0.8 - 0.6j,
            -0.8 + 0.4j,
            -0.25 - 0.9j,
            -0.2 + 0.7j,
            0.25 - 0.35j,
            0.35 + 0.85j,
            0.75 - 0.55j,
            0.9 + 0.25j,
        ],
        dtype=np.complex128,
    )
    h = 1e-5
    try:
        fx_plus = evaluate(expr, sample + h)
        fx_minus = evaluate(expr, sample - h)
        fy_plus = evaluate(expr, sample + 1j * h)
        fy_minus = evaluate(expr, sample - 1j * h)
    except Exception as exc:
        return {"passes": False, "reason": f"Cauchy-Riemann check could not run: {exc}"}

    if not all(np.all(np.isfinite(vals)) for vals in (fx_plus, fx_minus, fy_plus, fy_minus)):
        return {"passes": False, "reason": "Cauchy-Riemann check hit non-finite sampled values."}

    df_dx = (fx_plus - fx_minus) / (2 * h)
    df_dy = (fy_plus - fy_minus) / (2 * h)
    residual = np.maximum(np.abs(np.real(df_dx) - np.imag(df_dy)), np.abs(np.imag(df_dx) + np.real(df_dy)))
    scale = np.maximum(1.0, np.maximum(np.abs(df_dx), np.abs(df_dy)))
    relative = residual / scale
    max_relative = float(np.max(relative))
    return {
        "passes": bool(max_relative < 2e-5),
        "max_relative_residual": max_relative,
        "reason": f"Numerical Cauchy-Riemann residual: {max_relative:.2e}.",
    }


def classify_expression(expr: str) -> dict[str, Any]:
    features = analyze_expression(expr)
    names = set(features.used_names)
    function_names = names - SAFE_CONSTANTS - {"x", "y"}
    reasons: list[str] = []
    explicit_nonanalytic = sorted((names & NONANALYTIC_NAMES) - {"x", "y", "where", "piecewise"})
    component_only_nonanalytic = bool({"x", "y"} & names) and not explicit_nonanalytic and not features.has_piecewise
    cr_check = _cauchy_riemann_check(expr) if component_only_nonanalytic else None

    if features.has_piecewise:
        label = "piecewise analytic (not globally holomorphic)"
        reasons.append("Piecewise definitions can introduce seams where complex differentiability fails.")
    elif cr_check and cr_check["passes"]:
        if features.has_branchy:
            label = "holomorphic on a branch domain"
            reasons.append("Uses branch-cut functions, but the component form passed a numerical Cauchy-Riemann check away from sampled cuts.")
        elif features.has_denominator or features.uses_known_pole_families:
            label = "meromorphic"
            reasons.append("Component form passed a numerical Cauchy-Riemann check away from sampled singularities.")
        else:
            label = "holomorphic / analytic"
            reasons.append("Although it uses x and y directly, the component form passed a numerical Cauchy-Riemann check.")
        reasons.append(cr_check["reason"])
    elif features.has_nonanalytic:
        label = "non-holomorphic"
        bad = sorted((names & NONANALYTIC_NAMES) - {"where", "piecewise"})
        if bad:
            reasons.append("Uses non-holomorphic component operations: " + ", ".join(bad) + ".")
        if {"x", "y"} & names:
            if cr_check:
                reasons.append("Depends directly on real and imaginary components, and the numerical Cauchy-Riemann check did not pass.")
                reasons.append(cr_check["reason"])
            else:
                reasons.append("Depends directly on real and imaginary components.")
    elif features.has_branchy:
        label = "holomorphic on a branch domain"
        reasons.append("Uses multi-valued or branch-cut functions: " + ", ".join(sorted(function_names & BRANCHY_NAMES)) + ".")
    elif features.proven_entire:
        label = "entire"
        reasons.append("Built from z, constants, and functions in the app's conservative entire-function class.")
    elif features.has_denominator or features.uses_known_pole_families:
        label = "meromorphic"
        if features.has_denominator:
            reasons.append("Has denominator expressions that may create isolated poles.")
        if features.uses_known_pole_families:
            reasons.append("Uses functions with known pole families: " + ", ".join(sorted(function_names & KNOWN_POLE_FAMILIES)) + ".")
    else:
        label = "holomorphic / analytic"
        reasons.append("No branch cuts, denominators, or non-holomorphic operations were detected.")

    return {
        "kind": "classification",
        "label": label,
        "expression": features.expr,
        "reasons": reasons,
        "flags": {
            "branch_cuts": features.has_branchy,
            "non_holomorphic": (features.has_nonanalytic or features.has_piecewise) and not bool(cr_check and cr_check["passes"]),
            "denominators": features.has_denominator,
            "known_poles": features.uses_known_pole_families,
            "proven_entire": features.proven_entire,
            "cauchy_riemann_verified": bool(cr_check and cr_check["passes"]),
        },
    }


def _needs_implicit_multiply(prev: tokenize.TokenInfo, cur: tokenize.TokenInfo) -> bool:
    if prev.type not in {tokenize.NUMBER, tokenize.NAME, tokenize.OP}:
        return False
    if cur.type not in {tokenize.NUMBER, tokenize.NAME, tokenize.OP}:
        return False

    prev_can_end = prev.type in {tokenize.NUMBER, tokenize.NAME} or prev.string == ")"
    cur_can_start = cur.type in {tokenize.NUMBER, tokenize.NAME} or cur.string == "("
    if not (prev_can_end and cur_can_start):
        return False

    if prev.string in {"(", ",", "+", "-", "*", "/", "**", "%", "&", "|", "~", "<", "<=", ">", ">=", "==", "!="}:
        return False
    if cur.string in {")", ",", "+", "-", "*", "/", "**", "%", "&", "|", "<", "<=", ">", ">=", "==", "!="}:
        return False

    # Keep ordinary function calls like sin(z) intact and avoid turning "sin z" into "sin*z".
    if prev.type == tokenize.NAME and prev.string in CALLABLE_NAMES:
        return False
    return True


def _apply_implicit_multiplication(expr: str) -> str:
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(expr).readline)
            if token.type not in {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}
        ]
    except tokenize.TokenError:
        return expr

    output: list[tuple[int, str]] = []
    prev_significant: tokenize.TokenInfo | None = None
    for token in tokens:
        if prev_significant is not None and _needs_implicit_multiply(prev_significant, token):
            output.append((tokenize.OP, "*"))
        output.append((token.type, token.string))
        if token.type not in {tokenize.INDENT, tokenize.DEDENT}:
            prev_significant = token
    return tokenize.untokenize(output)


def preprocess(expr: str) -> str:
    processed = expr.strip().replace("^", "**")
    return _apply_implicit_multiplication(processed)


@lru_cache(maxsize=256)
def _parsed_expression(expr: str) -> tuple[ast.Expression, Any]:
    processed = preprocess(expr)
    tree = ast.parse(processed, mode="eval")
    SafeExpressionValidator(set(ALLOWED)).visit(tree)
    code = compile(tree, "<expr>", "eval")
    return tree, code


@lru_cache(maxsize=256)
def analyze_expression(expr: str) -> ExpressionFeatures:
    tree, _ = _parsed_expression(expr)
    name_collector = NameCollector()
    name_collector.visit(tree)
    used_names = frozenset(name_collector.names)

    denominator_collector = DenominatorCollector()
    denominator_collector.visit(tree)
    denominator_exprs = tuple(dict.fromkeys(denominator_collector.denominators))

    functional_names = used_names - SAFE_CONSTANTS
    has_branchy = bool(functional_names & BRANCHY_NAMES)
    has_nonanalytic = bool(functional_names & NONANALYTIC_NAMES)
    has_piecewise = "where" in functional_names or "piecewise" in functional_names
    has_denominator = bool(denominator_exprs)
    uses_known_pole_families = bool(functional_names & KNOWN_POLE_FAMILIES)

    theorem_eligible = not has_branchy and not has_nonanalytic and not has_piecewise

    allowed_entire_names = ENTIRE_FUNCTIONS | SAFE_CONSTANTS
    proven_entire = theorem_eligible and not has_denominator and not uses_known_pole_families and functional_names <= allowed_entire_names

    return ExpressionFeatures(
        expr=preprocess(expr),
        used_names=used_names,
        denominator_exprs=denominator_exprs,
        has_branchy=has_branchy,
        has_nonanalytic=has_nonanalytic,
        has_piecewise=has_piecewise,
        has_denominator=has_denominator,
        uses_known_pole_families=uses_known_pole_families,
        theorem_eligible=theorem_eligible,
        proven_entire=proven_entire,
    )


def evaluate(expr: str, z: np.ndarray | complex) -> np.ndarray:
    _, code = _parsed_expression(expr)
    z_arr = np.asarray(z, dtype=np.complex128)
    env = dict(ALLOWED)
    env["z"] = z_arr
    env["x"] = np.real(z_arr)
    env["y"] = np.imag(z_arr)

    with np.errstate(all="ignore"):
        out = eval(code, {"__builtins__": {}}, env)
    out_arr = np.asarray(out, dtype=np.complex128)
    if out_arr.shape == z_arr.shape:
        return out_arr
    try:
        return np.array(np.broadcast_to(out_arr, z_arr.shape), dtype=np.complex128, copy=True)
    except ValueError as exc:
        raise ValueError(
            f"Expression result shape {out_arr.shape} cannot be broadcast to the input grid shape {z_arr.shape}."
        ) from exc


def evaluate_scalar(expr: str, z: complex) -> complex:
    value = evaluate(expr, np.asarray([z], dtype=np.complex128))
    return complex(np.ravel(value)[0])
