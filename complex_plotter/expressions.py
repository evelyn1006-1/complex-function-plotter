from __future__ import annotations

import ast
import io
import math
import re as regex
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import optimize
from scipy import special as sp
import sympy as sy


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


def zetac(z: np.ndarray) -> np.ndarray:
    return sp.zeta(z) - 1


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
    "zetac": zetac,
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
    "tanh",
    "sec",
    "cot",
    "csc",
    "sech",
    "csch",
    "coth",
    "zeta",
    "zetac",
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
    "cis",
    "sinc",
}
VARIABLE_ORDER_SPECIAL_FUNCTIONS = {"jv", "iv"}
TRANSCENDENTAL_SINGULAR_COMPOSITION_NAMES = {
    "sin",
    "cos",
    "sinh",
    "cosh",
    "exp",
    "cis",
    "tan",
    "sec",
    "cot",
    "csc",
    "sech",
    "csch",
    "coth",
}
SAFE_CONSTANTS = {"z", "pi", "e", "tau", "i", "j"}
VALUE_NAMES = SAFE_CONSTANTS | {"x", "y"}
CALLABLE_NAMES = set(ALLOWED) - VALUE_NAMES
SYM_Z = sy.Symbol("z")
SYM_X = sy.Symbol("x", real=True)
SYM_Y = sy.Symbol("y", real=True)


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


def _node_uses_variable(node: ast.AST) -> bool:
    collector = NameCollector()
    collector.visit(node)
    return bool(collector.names & {"z", "x", "y"})


def _numeric_constant(node: ast.AST) -> complex | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
        return complex(node.value)
    if isinstance(node, ast.Name):
        if node.id in {"i", "j"}:
            return 1j
        if node.id == "pi":
            return complex(math.pi)
        if node.id == "e":
            return complex(math.e)
        if node.id == "tau":
            return complex(math.tau)
    if isinstance(node, ast.UnaryOp):
        value = _numeric_constant(node.operand)
        if value is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.BinOp):
        left = _numeric_constant(node.left)
        right = _numeric_constant(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and abs(right) > 0:
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        except Exception:
            return None
    return None


def _integer_exponent_value(node: ast.AST) -> int | None:
    value = _numeric_constant(node)
    if value is None or abs(value.imag) > 0:
        return None
    real = value.real
    if not math.isfinite(real):
        return None
    rounded = int(round(real))
    if abs(real - rounded) <= 1e-12:
        return rounded
    return None


def _constant_nonzero(node: ast.AST) -> bool:
    value = _numeric_constant(node)
    return bool(value is not None and math.isfinite(value.real) and math.isfinite(value.imag) and abs(value) > 0)


SYM_FUNCTIONS: dict[str, Any] = {
    "sin": sy.sin,
    "cos": sy.cos,
    "tan": sy.tan,
    "asin": sy.asin,
    "acos": sy.acos,
    "atan": sy.atan,
    "sinh": sy.sinh,
    "cosh": sy.cosh,
    "tanh": sy.tanh,
    "asinh": sy.asinh,
    "acosh": sy.acosh,
    "atanh": sy.atanh,
    "exp": sy.exp,
    "log": sy.log,
    "ln": sy.log,
    "log10": lambda arg: sy.log(arg, 10),
    "sqrt": sy.sqrt,
    "sec": sy.sec,
    "csc": sy.csc,
    "cot": sy.cot,
    "sech": sy.sech,
    "csch": sy.csch,
    "coth": sy.coth,
    "gamma": sy.gamma,
    "digamma": getattr(sy, "digamma", None),
    "psi": getattr(sy, "digamma", None),
    "erf": sy.erf,
    "erfc": sy.erfc,
    "erfi": sy.erfi,
    "erfcx": lambda arg: sy.exp(arg**2) * sy.erfc(arg),
    "loggamma": sy.loggamma,
    "rgamma": lambda arg: 1 / sy.gamma(arg),
    "cis": lambda arg: sy.exp(sy.I * arg),
    "sinc": lambda arg: sy.sinc(sy.pi * arg),
    "zeta": sy.zeta,
    "zetac": lambda arg: sy.zeta(arg) - 1,
    "lambertw": sy.LambertW,
    "expi": sy.Ei,
    "exp1": lambda arg: sy.expint(1, arg),
    "fresnels": sy.fresnels,
    "fresnelc": sy.fresnelc,
    "airyai": sy.airyai,
    "airyaip": sy.airyaiprime,
    "airybi": sy.airybi,
    "airybip": sy.airybiprime,
    "jv": sy.besselj,
    "yv": sy.bessely,
    "iv": sy.besseli,
    "kv": sy.besselk,
}


def _sympy_number(value: Any) -> sy.Expr | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return sy.Integer(value)
    if isinstance(value, float) and math.isfinite(value):
        return sy.Rational(repr(value))
    if isinstance(value, complex) and math.isfinite(value.real) and math.isfinite(value.imag):
        return sy.Rational(repr(float(value.real))) + sy.I * sy.Rational(repr(float(value.imag)))
    return None


def _to_sympy(node: ast.AST) -> sy.Expr | None:
    if isinstance(node, ast.Constant):
        return _sympy_number(node.value)
    if isinstance(node, ast.Name):
        if node.id == "z":
            return SYM_Z
        if node.id in {"i", "j"}:
            return sy.I
        if node.id == "pi":
            return sy.pi
        if node.id == "e":
            return sy.E
        if node.id == "tau":
            return 2 * sy.pi
        # Component expressions may still be symbolically holomorphic after x + i*y style rewrites.
        if node.id == "x":
            return sy.re(SYM_Z)
        if node.id == "y":
            return sy.im(SYM_Z)
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
        if isinstance(node.op, ast.Pow):
            return left ** right
        return None
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return None
        func = SYM_FUNCTIONS.get(node.func.id)
        if func is None:
            return None
        args = [_to_sympy(arg) for arg in node.args]
        if any(arg is None for arg in args):
            return None
        try:
            return func(*args)
        except Exception:
            return None
    return None


@lru_cache(maxsize=512)
def _sympy_from_text(expr: str) -> sy.Expr | None:
    try:
        tree = ast.parse(preprocess(expr), mode="eval")
    except SyntaxError:
        return None
    return _to_sympy(tree.body)


def _to_sympy_xy(node: ast.AST) -> sy.Expr | None:
    if isinstance(node, ast.Constant):
        return _sympy_number(node.value)
    if isinstance(node, ast.Name):
        if node.id == "z":
            return SYM_X + sy.I * SYM_Y
        if node.id == "x":
            return SYM_X
        if node.id == "y":
            return SYM_Y
        if node.id in {"i", "j"}:
            return sy.I
        if node.id == "pi":
            return sy.pi
        if node.id == "e":
            return sy.E
        if node.id == "tau":
            return 2 * sy.pi
        return None
    if isinstance(node, ast.UnaryOp):
        operand = _to_sympy_xy(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        return None
    if isinstance(node, ast.BinOp):
        left = _to_sympy_xy(node.left)
        right = _to_sympy_xy(node.right)
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
        if isinstance(node.op, ast.Pow):
            return left ** right
        return None
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return None
        func = SYM_FUNCTIONS.get(node.func.id)
        if func is None:
            return None
        args = [_to_sympy_xy(arg) for arg in node.args]
        if any(arg is None for arg in args):
            return None
        try:
            return func(*args)
        except Exception:
            return None
    return None


@lru_cache(maxsize=256)
def _sympy_xy_expr(expr: str) -> sy.Expr | None:
    try:
        tree, _code = _parsed_expression(expr)
    except Exception:
        return None
    return _to_sympy_xy(tree.body)


def _denominator_status(expr: str, *, deep: bool) -> str:
    if not deep:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return "unknown"
        if _constant_nonzero(tree.body):
            return "provably_nonzero"
        if isinstance(tree.body, ast.Call) and isinstance(tree.body.func, ast.Name) and tree.body.func.id in {"exp", "cis", "gamma"}:
            return "provably_nonzero"
        return "unknown"

    sym_expr = _sympy_from_text(expr)
    if sym_expr is None:
        return "unknown"
    try:
        simplified = sy.simplify(sym_expr)
    except Exception:
        simplified = sym_expr
    if simplified == 0 or simplified.is_zero:
        return "identically_zero"
    if simplified.is_number:
        return "provably_nonzero" if _is_finite_sympy_value(simplified) else "unknown"
    if simplified.is_nonzero:
        return "provably_nonzero"
    if simplified.func == sy.exp:
        return "provably_nonzero"
    if simplified.func == sy.gamma:
        return "provably_nonzero"
    if isinstance(simplified, sy.Pow) and simplified.base.func == sy.exp:
        return "provably_nonzero"
    return "unknown"


def _node_has_pole_like_singularity(node: ast.AST) -> bool:
    collector = ExpressionFeatureCollector()
    collector.visit(node)
    return bool(
        collector.effective_denominators
        or collector.identically_zero_denominators
        or _active_known_pole_names_for_collector(collector)
    )


class ExpressionFeatureCollector(ast.NodeVisitor):
    def __init__(self, *, deep: bool = False) -> None:
        self.used_names: set[str] = set()
        self.used_call_names: set[str] = set()
        self.deep = deep
        self.denominators: list[str] = []
        self.effective_denominators: list[str] = []
        self.provably_nonzero_denominators: list[str] = []
        self.identically_zero_denominators: list[str] = []
        self.branchy_power_exprs: list[str] = []
        self.essential_singularity_exprs: list[str] = []
        self.nonanalytic_operator_exprs: list[str] = []
        self.variable_order_special_calls: list[str] = []
        self.used_pole_function_names: set[str] = set()
        self.power_exponent_depth = 0
        self.denominator_depth = 0

    def _add_denominator(self, node: ast.AST) -> None:
        text = ast.unparse(node)
        self.denominators.append(text)
        status = _denominator_status(text, deep=self.deep)
        if status == "identically_zero":
            self.identically_zero_denominators.append(text)
        elif status == "provably_nonzero":
            self.provably_nonzero_denominators.append(text)
        else:
            self.effective_denominators.append(text)

    def visit_Name(self, node: ast.Name) -> None:
        self.used_names.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            self.used_call_names.add(name)
            if name in KNOWN_POLE_FAMILIES and self.denominator_depth == 0 and any(_node_uses_variable(arg) for arg in node.args):
                self.used_pole_function_names.add(name)
            if name in VARIABLE_ORDER_SPECIAL_FUNCTIONS and node.args:
                if _node_uses_variable(node.args[0]):
                    self.variable_order_special_calls.append(ast.unparse(node))
            if name in TRANSCENDENTAL_SINGULAR_COMPOSITION_NAMES and any(_node_has_pole_like_singularity(arg) for arg in node.args):
                self.essential_singularity_exprs.append(ast.unparse(node))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod):
            self.nonanalytic_operator_exprs.append(ast.unparse(node))
            self.generic_visit(node)
            return
        if isinstance(node.op, ast.Div):
            if self.power_exponent_depth == 0:
                self._add_denominator(node.right)
            self.visit(node.left)
            self.denominator_depth += 1
            try:
                self.visit(node.right)
            finally:
                self.denominator_depth -= 1
            return
        elif isinstance(node.op, ast.Pow):
            exponent = _integer_exponent_value(node.right)
            if exponent is None:
                if _constant_nonzero(node.left):
                    if _node_has_pole_like_singularity(node.right):
                        self.essential_singularity_exprs.append(ast.unparse(node))
                else:
                    self.branchy_power_exprs.append(ast.unparse(node))
            elif exponent < 0:
                self._add_denominator(node.left)
            if exponent is not None and exponent < 0:
                self.denominator_depth += 1
                try:
                    self.visit(node.left)
                finally:
                    self.denominator_depth -= 1
            else:
                self.visit(node.left)
            self.power_exponent_depth += 1
            try:
                self.visit(node.right)
            finally:
                self.power_exponent_depth -= 1
            return
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.nonanalytic_operator_exprs.append(ast.unparse(node))
        self.generic_visit(node)


def _active_known_pole_names_for_collector(collector: ExpressionFeatureCollector) -> set[str]:
    return set(collector.used_pole_function_names)


@dataclass(frozen=True)
class ExpressionFeatures:
    expr: str
    used_names: frozenset[str]
    denominator_exprs: tuple[str, ...]
    effective_denominator_exprs: tuple[str, ...]
    provably_nonzero_denominator_exprs: tuple[str, ...]
    identically_zero_denominator_exprs: tuple[str, ...]
    branchy_power_exprs: tuple[str, ...]
    essential_singularity_exprs: tuple[str, ...]
    nonanalytic_operator_exprs: tuple[str, ...]
    variable_order_special_calls: tuple[str, ...]
    has_branchy: bool
    has_nonanalytic: bool
    has_piecewise: bool
    has_denominator: bool
    has_effective_denominator: bool
    has_undefined_denominator: bool
    has_branchy_power: bool
    has_essential_singularity: bool
    has_variable_order_special_function: bool
    uses_known_pole_families: bool
    active_known_pole_names: frozenset[str]
    theorem_eligible: bool
    proven_entire: bool


def _active_known_pole_names(features: ExpressionFeatures) -> set[str]:
    return set(features.active_known_pole_names)


def _cauchy_riemann_check(expr: str, *, deep: bool = False) -> dict[str, Any]:
    if deep:
        symbolic = _symbolic_cauchy_riemann_check(expr)
        if symbolic is not None:
            return symbolic

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


def _symbolic_cauchy_riemann_check(expr: str) -> dict[str, Any] | None:
    sym_expr = _sympy_xy_expr(expr)
    if sym_expr is None:
        return None
    try:
        real_part = sy.simplify(sy.re(sym_expr))
        imag_part = sy.simplify(sy.im(sym_expr))
        cr_real = sy.simplify(sy.diff(real_part, SYM_X) - sy.diff(imag_part, SYM_Y))
        cr_imag = sy.simplify(sy.diff(real_part, SYM_Y) + sy.diff(imag_part, SYM_X))
    except Exception:
        return None

    real_zero = cr_real == 0 or cr_real.is_zero
    imag_zero = cr_imag == 0 or cr_imag.is_zero
    if real_zero and imag_zero:
        return {
            "passes": True,
            "reason": "Symbolic Cauchy-Riemann check passed.",
        }
    if real_zero is False or imag_zero is False:
        return {
            "passes": False,
            "reason": "Symbolic Cauchy-Riemann check did not pass.",
        }
    return None


def classify_expression(expr: str, *, deep: bool = False) -> dict[str, Any]:
    features = analyze_expression(expr, deep=deep)
    names = set(features.used_names)
    function_names = names - SAFE_CONSTANTS - {"x", "y"}
    active_known_poles = _active_known_pole_names(features)
    reasons: list[str] = []
    explicit_nonanalytic = sorted((names & NONANALYTIC_NAMES) - {"x", "y", "where", "piecewise"})
    component_only_nonanalytic = bool({"x", "y"} & names) and not explicit_nonanalytic and not features.has_piecewise
    cr_check = _cauchy_riemann_check(expr, deep=deep) if component_only_nonanalytic else None

    if features.has_undefined_denominator:
        label = "undefined / singular everywhere"
        reasons.append("Has denominator expressions that simplify to 0: " + ", ".join(features.identically_zero_denominator_exprs) + ".")
    elif features.has_piecewise:
        label = "piecewise analytic (not globally holomorphic)"
        reasons.append("Piecewise definitions can introduce seams where complex differentiability fails.")
    elif features.has_essential_singularity:
        label = "holomorphic except at isolated singularities"
        reasons.append("Composes a pole-like singularity into a transcendental function, producing non-meromorphic isolated singular behavior: " + ", ".join(features.essential_singularity_exprs) + ".")
    elif cr_check and cr_check["passes"]:
        if features.has_branchy:
            label = "holomorphic on a branch domain"
            reasons.append("Uses branch-cut or fractional-power behavior, but the component form passed a Cauchy-Riemann check away from sampled cuts.")
        elif features.has_effective_denominator or active_known_poles:
            label = "meromorphic"
            reasons.append("Component form passed a Cauchy-Riemann check away from sampled singularities.")
        else:
            label = "locally holomorphic / analytic"
            reasons.append("Although it uses x and y directly, the component form passed a Cauchy-Riemann check.")
        reasons.append(cr_check["reason"])
    elif features.has_nonanalytic:
        label = "non-holomorphic"
        bad = sorted((names & NONANALYTIC_NAMES) - {"where", "piecewise"})
        if bad:
            reasons.append("Uses non-holomorphic component operations: " + ", ".join(bad) + ".")
        if features.nonanalytic_operator_exprs:
            reasons.append("Uses comparison or modulo syntax, which is not complex-holomorphic: " + ", ".join(features.nonanalytic_operator_exprs) + ".")
        if {"x", "y"} & names:
            if cr_check:
                reasons.append("Depends directly on real and imaginary components, and the Cauchy-Riemann check did not pass.")
                reasons.append(cr_check["reason"])
            else:
                reasons.append("Depends directly on real and imaginary components.")
    elif features.has_branchy:
        label = "holomorphic on a branch domain"
        branch_names = sorted(function_names & BRANCHY_NAMES)
        if branch_names:
            reasons.append("Uses multi-valued or branch-cut functions: " + ", ".join(branch_names) + ".")
        if features.branchy_power_exprs:
            reasons.append("Uses non-integer or variable powers that require a branch choice: " + ", ".join(features.branchy_power_exprs) + ".")
    elif features.has_variable_order_special_function:
        label = "special-function analytic status uncertain"
        reasons.append("Uses Bessel-family calls with z in the order argument: " + ", ".join(features.variable_order_special_calls) + ".")
    elif features.proven_entire:
        label = "entire"
        reasons.append("Built from z, constants, and functions in the app's conservative entire-function class.")
        if features.provably_nonzero_denominator_exprs:
            reasons.append("All denominator expressions present were recognized as nonzero: " + ", ".join(features.provably_nonzero_denominator_exprs) + ".")
    elif features.has_effective_denominator or active_known_poles:
        label = "meromorphic"
        if features.has_effective_denominator:
            reasons.append("Has denominator expressions that may create isolated poles: " + ", ".join(features.effective_denominator_exprs) + ".")
        if active_known_poles:
            reasons.append("Uses functions with known pole families in pole-producing contexts: " + ", ".join(sorted(active_known_poles)) + ".")
    else:
        label = "probably holomorphic (unclassified singularities)"
        reasons.append("No branch cuts, denominators, or non-holomorphic operations were detected, but the expression is outside the app's conservative entire-function class.")

    singularities = classify_singularities_from_features(features)
    if deep and not features.has_undefined_denominator:
        deep_singularities, deep_reasons = _deep_symbolic_singularity_notes(features)
        singularities.extend(deep_singularities)
        reasons.extend(deep_reasons)
        symbolic_kinds = {item["kind"] for item in deep_singularities}
        if (
            label == "meromorphic"
            and "symbolic_removable_singularity" in symbolic_kinds
            and symbolic_kinds <= {"symbolic_removable_singularity"}
            and not active_known_poles
        ):
            label = "meromorphic with removable singularities"
            reasons.append("Deep SymPy pass found only removable symbolic singularities.")
        if label == "holomorphic on a branch domain" and _deep_simplifies_without_branch(features):
            label = "entire after symbolic simplification"
            reasons.append("Deep SymPy pass simplified away the detected branch behavior.")
        if (
            label not in {"entire", "entire after symbolic simplification", "non-holomorphic", "undefined / singular everywhere"}
            and not features.has_nonanalytic
            and not features.has_piecewise
            and not features.has_variable_order_special_function
            and _deep_simplifies_without_branch(features)
        ):
            label = "entire after symbolic simplification"
            reasons.append("Deep SymPy pass simplified the expression to a branch-free form with no singularities.")
        if (
            label == "probably holomorphic (unclassified singularities)"
            and not deep_singularities
            and any("found no singularities in z" in reason for reason in deep_reasons)
            and not features.has_branchy
            and not features.has_nonanalytic
            and not features.has_piecewise
            and not features.has_variable_order_special_function
            and not features.has_effective_denominator
            and not active_known_poles
        ):
            label = "entire after symbolic simplification"
            reasons.append("Deep SymPy pass found no singularities after accounting for known nonzero special functions.")

    return {
        "kind": "classification",
        "analysis_depth": "deep" if deep else "fast",
        "label": label,
        "expression": features.expr,
        "reasons": reasons,
        "flags": {
            "branch_cuts": features.has_branchy,
            "non_holomorphic": (features.has_nonanalytic or features.has_piecewise) and not bool(cr_check and cr_check["passes"]),
            "denominators": features.has_effective_denominator,
            "provably_nonzero_denominators": bool(features.provably_nonzero_denominator_exprs),
            "undefined_denominators": features.has_undefined_denominator,
            "known_poles": bool(active_known_poles),
            "essential_singularities": features.has_essential_singularity,
            "proven_entire": features.proven_entire,
            "cauchy_riemann_verified": bool(cr_check and cr_check["passes"]),
        },
        "singularities": singularities,
    }


def _sympy_text(value: Any) -> str:
    return regex.sub(r"\bI\b", "i", sy.sstr(value))


def _sympy_has_branch_constructs(sym_expr: sy.Expr) -> bool:
    branch_funcs = {
        sy.log,
        sy.asin,
        sy.acos,
        sy.atan,
        sy.asinh,
        sy.acosh,
        sy.atanh,
        sy.LambertW,
        sy.loggamma,
        sy.Ei,
    }
    for node in sy.preorder_traversal(sym_expr):
        if getattr(node, "func", None) in branch_funcs:
            return True
        func_name = getattr(getattr(node, "func", None), "__name__", "")
        if func_name in {"bessely", "besselk"}:
            return True
        if isinstance(node, sy.Pow):
            exponent = node.exp
            if not (exponent.is_integer is True):
                return True
    return False


def _sympy_has_known_pole_constructs(sym_expr: sy.Expr) -> bool:
    for node in sy.preorder_traversal(sym_expr):
        func_name = getattr(getattr(node, "func", None), "__name__", "")
        if func_name in KNOWN_POLE_FAMILIES:
            return True
    return False


def _deep_simplifies_without_branch(features: ExpressionFeatures) -> bool:
    sym_expr = _sympy_from_text(features.expr)
    if sym_expr is None:
        return False
    try:
        simplified = sy.simplify(sym_expr)
        singularities = sy.singularities(simplified, SYM_Z)
    except Exception:
        return False
    return bool(
        singularities in (sy.S.EmptySet, sy.EmptySet)
        and not _sympy_has_branch_constructs(simplified)
        and not _sympy_has_known_pole_constructs(simplified)
    )


def _is_finite_sympy_value(value: sy.Expr) -> bool:
    if isinstance(value, sy.AccumBounds):
        return False
    if value in (sy.oo, -sy.oo, sy.zoo, sy.nan):
        return False
    if value.has(sy.oo, -sy.oo, sy.zoo, sy.nan):
        return False
    finite = value.is_finite
    return bool(finite is True)


def _is_known_nonzero_sympy_expr(value: sy.Expr) -> bool:
    try:
        simplified = sy.simplify(value)
    except Exception:
        simplified = value
    if simplified.is_nonzero:
        return True
    if simplified.func in {sy.exp, sy.gamma}:
        return True
    if isinstance(simplified, sy.Pow):
        return _is_known_nonzero_sympy_expr(simplified.base)
    if isinstance(simplified, sy.Mul):
        return all(_is_known_nonzero_sympy_expr(arg) for arg in simplified.args)
    return False


def _singularity_set_is_known_empty(singularities: sy.Set) -> bool:
    if not isinstance(singularities, sy.ConditionSet):
        return False
    condition = singularities.condition
    if not isinstance(condition, sy.Equality):
        return False
    left, right = condition.lhs, condition.rhs
    return bool(
        (right == 0 or right.is_zero) and _is_known_nonzero_sympy_expr(left)
        or (left == 0 or left.is_zero) and _is_known_nonzero_sympy_expr(right)
    )


def _symbolic_pole_order(sym_expr: sy.Expr, point: sy.Expr, max_order: int = 12) -> int | None:
    for order in range(1, max_order + 1):
        try:
            scaled = sy.simplify((SYM_Z - point) ** order * sym_expr)
            value = sy.limit(scaled, SYM_Z, point)
        except Exception:
            continue
        if _is_finite_sympy_value(value) and not (value == 0 or value.is_zero):
            return order
    return None


def _numeric_local_singularity_classification(expr: str, point: complex) -> tuple[str | None, int | None, str | None]:
    radii = np.asarray([1e-2, 3e-3, 1e-3, 3e-4, 1e-4], dtype=float)
    angles = np.asarray([0.0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3], dtype=float)
    magnitudes: list[float] = []
    for radius in radii:
        values: list[float] = []
        for angle in angles:
            try:
                val = evaluate_scalar(expr, point + radius * complex(math.cos(angle), math.sin(angle)))
            except Exception:
                continue
            if math.isfinite(val.real) and math.isfinite(val.imag):
                values.append(abs(val))
        if not values:
            magnitudes.append(float("nan"))
        else:
            magnitudes.append(float(np.median(values)))

    finite = np.asarray([m for m in magnitudes if math.isfinite(m) and m > 0], dtype=float)
    finite_radii = np.asarray([r for r, m in zip(radii, magnitudes) if math.isfinite(m) and m > 0], dtype=float)
    if finite.size < 3:
        return None, None, None

    if float(np.max(finite)) < 1e6 and float(np.max(finite) / max(np.min(finite), 1e-300)) < 20:
        return "symbolic_removable_singularity", None, "Numerical local sampling stayed bounded near this point."

    slope, _intercept = np.polyfit(np.log(finite_radii), np.log(finite), 1)
    estimated_order = int(round(-float(slope)))
    if 1 <= estimated_order <= 12 and abs((-float(slope)) - estimated_order) < 0.35:
        return (
            "symbolic_pole",
            estimated_order,
            f"Numerical local sampling estimated a pole of order {estimated_order}.",
        )

    if float(np.max(finite)) > 1e6:
        return "symbolic_essential_candidate", None, "Numerical local sampling grew rapidly but did not match a stable pole order."
    return None, None, None


def _simple_point_for_local_analysis(point: sy.Expr) -> bool:
    text = sy.sstr(point)
    return "RootOf" not in text and len(text) <= 48


def _numeric_classification_item(point: sy.Expr, features: ExpressionFeatures, suffix: str = "") -> dict[str, Any] | None:
    numeric_point = _complex_from_sympy(point)
    if numeric_point is None:
        return None
    numeric_kind, numeric_order, numeric_reason = _numeric_local_singularity_classification(features.expr, numeric_point)
    if numeric_kind is None or numeric_reason is None:
        return None
    item = {
        "kind": numeric_kind,
        "source": _sympy_text(point),
        "reason": numeric_reason + suffix,
    }
    if numeric_order is not None:
        item["pole_order"] = numeric_order
    return item


def _symbolic_is_zero(value: sy.Expr) -> bool:
    try:
        simplified = sy.simplify(value)
    except Exception:
        simplified = value
    if simplified == 0 or simplified.is_zero:
        return True
    numeric = _complex_from_sympy(simplified)
    return bool(numeric is not None and abs(numeric) < 1e-8)


def _point_satisfies_branch_equation(expr: str, point: sy.Expr) -> bool:
    for equation in _branch_point_equations_from_ast(expr):
        try:
            value = equation.subs(SYM_Z, point)
        except Exception:
            continue
        if _symbolic_is_zero(value):
            return True
    return False


def _classify_symbolic_point(sym_expr: sy.Expr, point: sy.Expr, features: ExpressionFeatures) -> dict[str, Any]:
    source = _sympy_text(point)
    if features.has_branchy and _point_satisfies_branch_equation(features.expr, point):
        return {
            "kind": "symbolic_branch_point_candidate",
            "source": source,
            "reason": "Deep SymPy pass found this point as a branch candidate.",
        }

    if features.has_essential_singularity:
        numeric_item = _numeric_classification_item(point, features, " Candidate came from a pole-family expression inside a transcendental composition.")
        if numeric_item is not None and numeric_item["kind"] != "symbolic_removable_singularity":
            return numeric_item
        return {
            "kind": "symbolic_essential_candidate",
            "source": source,
            "reason": "This point is a candidate from a pole-like expression inside a transcendental composition.",
        }

    if _active_known_pole_names(features):
        numeric_item = _numeric_classification_item(point, features, " Known pole-family local proof used numerical sampling.")
        if numeric_item is not None:
            return numeric_item

    if not _simple_point_for_local_analysis(point):
        numeric_item = _numeric_classification_item(point, features, " Symbolic local proof was skipped at a complicated algebraic point.")
        if numeric_item is not None:
            return numeric_item
        if features.has_effective_denominator or _active_known_pole_names(features):
            return {
                "kind": "symbolic_pole_candidate",
                "source": source,
                "reason": "Deep SymPy pass found this exact singularity candidate, but skipped expensive local proof at a complicated algebraic point.",
            }
        return {
            "kind": "symbolic_isolated_singularity",
            "source": source,
            "reason": "Deep SymPy pass found this isolated singularity candidate, but skipped expensive local proof at a complicated algebraic point.",
        }

    try:
        limit_value = sy.limit(sym_expr, SYM_Z, point)
    except Exception:
        limit_value = None

    if limit_value is not None and _is_finite_sympy_value(limit_value):
        return {
            "kind": "symbolic_removable_singularity",
            "source": source,
            "reason": f"Deep SymPy pass found a finite limit {_sympy_text(limit_value)} here.",
        }

    pole_order = _symbolic_pole_order(sym_expr, point)
    if pole_order is not None:
        return {
            "kind": "symbolic_pole",
            "source": source,
            "reason": f"Deep SymPy pass found a pole of order {pole_order}.",
            "pole_order": pole_order,
        }

    return {
        "kind": "symbolic_isolated_singularity",
        "source": source,
        "reason": "Deep SymPy pass found this isolated singularity candidate.",
    }


def _candidate_denominator_zeroes(features: ExpressionFeatures, limit: int = 24) -> list[sy.Expr]:
    points: list[sy.Expr] = []
    for denominator in features.effective_denominator_exprs:
        den_expr = _sympy_from_text(denominator)
        if den_expr is None:
            continue
        try:
            zeroes = sy.solveset(den_expr, SYM_Z, domain=sy.S.Complexes)
        except Exception:
            continue
        if not isinstance(zeroes, sy.FiniteSet):
            continue
        for point in zeroes:
            if not any(sy.simplify(point - existing) == 0 for existing in points):
                points.append(point)
                if len(points) >= limit:
                    return points
    return points


@lru_cache(maxsize=2048)
def _complex_from_sympy(value: sy.Expr) -> complex | None:
    try:
        numeric = complex(sy.N(value, 18))
    except Exception:
        return None
    if not math.isfinite(numeric.real) or not math.isfinite(numeric.imag):
        return None
    return numeric


def _inside_bounds(point: complex, bounds: tuple[float, float, float, float], pad: float = 1e-9) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return xmin - pad <= point.real <= xmax + pad and ymin - pad <= point.imag <= ymax + pad


def _add_symbolic_point(points: list[sy.Expr], point: sy.Expr, bounds: tuple[float, float, float, float], limit: int) -> bool:
    numeric = _complex_from_sympy(point)
    if numeric is None or not _inside_bounds(numeric, bounds):
        return True
    if _has_nearby_symbolic_point(points, numeric):
        return True
    points.append(point)
    return len(points) < limit


def _has_nearby_symbolic_point(points: list[sy.Expr], numeric: complex, tol: float = 1e-7) -> bool:
    return any(
        abs(numeric - existing_numeric) < tol
        for existing in points
        if (existing_numeric := _complex_from_sympy(existing)) is not None
    )


def _integer_range_for_linear_imageset(expr: sy.Expr, var: sy.Symbol, bounds: tuple[float, float, float, float]) -> range | None:
    coeff = sy.diff(expr, var)
    offset = sy.simplify(expr.subs(var, 0))
    if coeff.has(var) or offset.has(var):
        return None
    coeff_numeric = _complex_from_sympy(coeff)
    offset_numeric = _complex_from_sympy(offset)
    if coeff_numeric is None or offset_numeric is None or abs(coeff_numeric) < 1e-12:
        return None

    xmin, xmax, ymin, ymax = bounds
    intervals: list[tuple[float, float]] = []
    if abs(coeff_numeric.real) > 1e-12:
        a = (xmin - offset_numeric.real) / coeff_numeric.real
        b = (xmax - offset_numeric.real) / coeff_numeric.real
        intervals.append((min(a, b), max(a, b)))
    elif not xmin <= offset_numeric.real <= xmax:
        return range(0)

    if abs(coeff_numeric.imag) > 1e-12:
        a = (ymin - offset_numeric.imag) / coeff_numeric.imag
        b = (ymax - offset_numeric.imag) / coeff_numeric.imag
        intervals.append((min(a, b), max(a, b)))
    elif not ymin <= offset_numeric.imag <= ymax:
        return range(0)

    if not intervals:
        return None
    low = max(item[0] for item in intervals)
    high = min(item[1] for item in intervals)
    if low > high:
        return range(0)
    return range(math.floor(low) - 1, math.ceil(high) + 2)


def _collect_symbolic_singularity_points(
    singularities: sy.Set,
    bounds: tuple[float, float, float, float],
    points: list[sy.Expr],
    limit: int,
) -> bool:
    if singularities in (sy.S.EmptySet, sy.EmptySet):
        return True
    if isinstance(singularities, sy.FiniteSet):
        for point in singularities:
            if not _add_symbolic_point(points, point, bounds, limit):
                return False
        return True
    if isinstance(singularities, sy.Union):
        for subset in singularities.args:
            if not _collect_symbolic_singularity_points(subset, bounds, points, limit):
                return False
        return True
    if isinstance(singularities, sy.ImageSet) and isinstance(singularities.lamda, sy.Lambda) and singularities.base_set == sy.S.Integers:
        variables = singularities.lamda.variables
        if len(variables) != 1:
            return False
        n_range = _integer_range_for_linear_imageset(singularities.lamda.expr, variables[0], bounds)
        if n_range is None:
            return False
        for n in n_range:
            point = sy.simplify(singularities.lamda.expr.subs(variables[0], n))
            if not _add_symbolic_point(points, point, bounds, limit):
                return False
        return True
    return False


def _solve_symbolic_points(expr: sy.Expr, bounds: tuple[float, float, float, float], limit: int) -> list[sy.Expr]:
    try:
        solutions = sy.solveset(expr, SYM_Z, domain=sy.S.Complexes)
    except Exception:
        return []
    points: list[sy.Expr] = []
    _collect_symbolic_singularity_points(solutions, bounds, points, limit)
    return points


@lru_cache(maxsize=256)
def _branch_point_equations_from_ast(expr: str) -> tuple[sy.Expr, ...]:
    try:
        tree, _code = _parsed_expression(expr)
    except Exception:
        return ()
    equations: list[sy.Expr] = []

    def add_equation(sym_expr: sy.Expr) -> None:
        if not any(_symbolic_is_zero(sym_expr - existing) for existing in equations):
            equations.append(sym_expr)

    class BranchPointCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.args:
                arg = _to_sympy(node.args[0])
                if arg is not None:
                    name = node.func.id
                    if name in {"log", "ln", "log10", "sqrt"}:
                        add_equation(arg)
                    elif name in {"asin", "acos", "atanh", "acosh"}:
                        add_equation(arg - 1)
                        add_equation(arg + 1)
                    elif name in {"atan", "asinh"}:
                        add_equation(arg - sy.I)
                        add_equation(arg + sy.I)
                    elif name == "lambertw":
                        add_equation(arg + sy.E**-1)
                    elif name in {"exp1", "expi"}:
                        add_equation(arg)
            self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp) -> None:
            if isinstance(node.op, ast.Pow) and _integer_exponent_value(node.right) is None:
                base = _to_sympy(node.left)
                if base is not None:
                    add_equation(base)
            self.generic_visit(node)

    BranchPointCollector().visit(tree)
    return tuple(equations)


def _branch_points_from_ast(expr: str, bounds: tuple[float, float, float, float], limit: int) -> list[sy.Expr]:
    points: list[sy.Expr] = []
    for equation in _branch_point_equations_from_ast(expr):
        for point in _solve_symbolic_points(equation, bounds, limit):
            if len(points) >= limit:
                return points
            _add_symbolic_point(points, point, bounds, limit)
    return points


def _solve_family_preimage_points(arg: sy.Expr, target: sy.Expr, bounds: tuple[float, float, float, float], limit: int) -> list[sy.Expr]:
    try:
        equation = sy.simplify(arg - target)
    except Exception:
        return []
    if equation == 0 or equation.is_zero:
        return []
    try:
        if not equation.is_polynomial(SYM_Z):
            return []
        degree = sy.Poly(equation, SYM_Z).degree()
    except Exception:
        return []
    if degree < 1 or degree > 5:
        return []
    return _solve_symbolic_points(equation, bounds, limit)


def _known_family_points(features: ExpressionFeatures, bounds: tuple[float, float, float, float], limit: int) -> list[sy.Expr]:
    xmin, xmax, ymin, ymax = bounds
    names = _active_known_pole_names(features)
    points: list[sy.Expr] = []
    if not names:
        return points

    def add(point: sy.Expr) -> None:
        if len(points) >= limit:
            return
        numeric = _complex_from_sympy(point)
        if numeric is None or not _inside_bounds(numeric, bounds):
            return
        if _has_nearby_symbolic_point(points, numeric):
            return
        points.append(point)

    span = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax), 1.0)
    integer_radius = min(max(limit, math.ceil(span) + 8), 120)
    real_period_radius = min(max(limit, math.ceil(span / math.pi) + 8), 120)

    try:
        tree = ast.parse(features.expr, mode="eval")
    except SyntaxError:
        return points

    class KnownFamilyPointCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if not isinstance(node.func, ast.Name) or not node.args:
                self.generic_visit(node)
                return
            name = node.func.id
            if name not in names:
                self.generic_visit(node)
                return
            arg = _to_sympy(node.args[0])
            if arg is None:
                self.generic_visit(node)
                return

            targets: list[sy.Expr] = []
            if name in {"gamma", "digamma", "psi"}:
                targets = [sy.Integer(k) for k in range(-integer_radius, 1)]
            elif name in {"zeta", "zetac"}:
                targets = [sy.Integer(1)]
            elif name in {"tan", "sec"}:
                targets = [sy.pi / 2 + sy.Integer(n) * sy.pi for n in range(-real_period_radius, real_period_radius + 1)]
            elif name in {"cot", "csc"}:
                targets = [sy.Integer(n) * sy.pi for n in range(-real_period_radius, real_period_radius + 1)]
            elif name in {"tanh", "sech"}:
                targets = [sy.I * (sy.pi / 2 + sy.Integer(n) * sy.pi) for n in range(-real_period_radius, real_period_radius + 1)]
            elif name in {"coth", "csch"}:
                targets = [sy.I * sy.Integer(n) * sy.pi for n in range(-real_period_radius, real_period_radius + 1)]

            for target in targets:
                if len(points) >= limit:
                    break
                for point in _solve_family_preimage_points(arg, target, bounds, limit - len(points)):
                    add(point)
                    if len(points) >= limit:
                        break
            self.generic_visit(node)

    KnownFamilyPointCollector().visit(tree)
    return points


def _numeric_denominator_roots(
    den_expr: str,
    bounds: tuple[float, float, float, float],
    *,
    limit: int,
) -> list[complex]:
    xmin, xmax, ymin, ymax = bounds
    sample_n = 25
    xs = np.linspace(xmin, xmax, sample_n)
    ys = np.linspace(ymin, ymax, sample_n)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    try:
        vals = evaluate(den_expr, Z)
    except Exception:
        return []
    score = np.abs(vals)
    score[~np.isfinite(score)] = np.inf
    if not np.any(np.isfinite(score)):
        return []

    flat_order = np.argsort(score, axis=None)
    seed_spacing = 0.35 * max((xmax - xmin) / max(sample_n - 1, 1), (ymax - ymin) / max(sample_n - 1, 1))
    seeds: list[complex] = []
    for idx in flat_order[: min(60, flat_order.size)]:
        iy, ix = np.unravel_index(int(idx), score.shape)
        seed = complex(float(X[iy, ix]), float(Y[iy, ix]))
        if not any(abs(seed - existing) < seed_spacing for existing in seeds):
            seeds.append(seed)
        if len(seeds) >= max(12, 3 * limit):
            break

    roots: list[complex] = []
    for seed in seeds:
        def f_xy(v):
            z = complex(float(v[0]), float(v[1]))
            val = evaluate_scalar(den_expr, z)
            return [float(np.real(val)), float(np.imag(val))]

        try:
            sol = optimize.root(f_xy, [seed.real, seed.imag], method="hybr")
        except Exception:
            continue
        if not sol.success:
            continue
        root = complex(float(sol.x[0]), float(sol.x[1]))
        if not _inside_bounds(root, bounds, pad=1e-5):
            continue
        try:
            residual = evaluate_scalar(den_expr, root)
        except Exception:
            continue
        if not (math.isfinite(residual.real) and math.isfinite(residual.imag)) or abs(residual) > 1e-5:
            continue
        if any(abs(root - existing) < 1e-5 for existing in roots):
            continue
        roots.append(root)
        if len(roots) >= limit:
            break
    return roots


def singularity_points_in_bounds(
    expr: str,
    bounds: tuple[float, float, float, float],
    *,
    max_points: int = 80,
) -> list[dict[str, Any]]:
    features = analyze_expression(expr, deep=True)
    if features.has_undefined_denominator:
        return []
    sym_expr = _sympy_from_text(features.expr)
    if sym_expr is None:
        return []

    points: list[sy.Expr] = []
    try:
        singularities = sy.singularities(sym_expr, SYM_Z)
    except Exception:
        singularities = sy.S.EmptySet
    if _singularity_set_is_known_empty(singularities):
        singularities = sy.S.EmptySet
    _collect_symbolic_singularity_points(singularities, bounds, points, max_points)
    for point in _known_family_points(features, bounds, max_points):
        if len(points) >= max_points:
            break
        _add_symbolic_point(points, point, bounds, max_points)
    if not points:
        for point in _candidate_denominator_zeroes(features, limit=max_points):
            if len(points) >= max_points:
                break
            _add_symbolic_point(points, point, bounds, max_points)
    for den_expr in features.effective_denominator_exprs:
        if len(points) >= max_points:
            break
        for root in _numeric_denominator_roots(den_expr, bounds, limit=max_points - len(points)):
            if not _has_nearby_symbolic_point(points, root, tol=1e-5):
                points.append(sy.N(root.real, 15) + sy.I * sy.N(root.imag, 15))
            if len(points) >= max_points:
                break
    for point in _branch_points_from_ast(features.expr, bounds, max_points):
        if len(points) >= max_points:
            break
        _add_symbolic_point(points, point, bounds, max_points)

    items: list[dict[str, Any]] = []
    for point in points[:max_points]:
        numeric = _complex_from_sympy(point)
        if numeric is None:
            continue
        classified = _classify_symbolic_point(sym_expr, point, features)
        items.append({
            "point": [float(numeric.real), float(numeric.imag)],
            "exact_point": _sympy_text(point),
            "kind": classified["kind"],
            "source": classified["source"],
            "reason": classified["reason"],
            **({"pole_order": classified["pole_order"]} if "pole_order" in classified else {}),
        })
    return items


def _deep_symbolic_singularity_notes(features: ExpressionFeatures) -> tuple[list[dict[str, Any]], list[str]]:
    if features.proven_entire:
        return [], ["Deep pass skipped SymPy singularity search because the fast classifier proved the expression entire."]

    sym_expr = _sympy_from_text(features.expr)
    if sym_expr is None:
        return [], ["Deep SymPy pass could not translate the expression."]
    try:
        singularities = sy.singularities(sym_expr, SYM_Z)
    except Exception as exc:
        return [], [f"Deep SymPy singularity search could not run: {exc}"]
    if _singularity_set_is_known_empty(singularities):
        singularities = sy.S.EmptySet

    items: list[dict[str, Any]] = []
    if singularities in (sy.S.EmptySet, sy.EmptySet):
        for point in _candidate_denominator_zeroes(features):
            classified = _classify_symbolic_point(sym_expr, point, features)
            if classified["kind"] == "symbolic_removable_singularity":
                items.append(classified)
        if items:
            return items, ["Deep SymPy pass found removable denominator zeroes after simplification."]
        if features.has_branchy:
            return [], ["Deep SymPy pass found no additional isolated singularities; branch behavior may still remain."]
        return [], ["Deep SymPy pass found no singularities in z."]

    if isinstance(singularities, sy.FiniteSet):
        for point in singularities:
            items.append(_classify_symbolic_point(sym_expr, point, features))
    else:
        items.append({
            "kind": "symbolic_singularity_set",
            "source": _sympy_text(singularities),
            "reason": "Deep SymPy pass found this singularity set.",
        })
    return items, ["Deep SymPy pass found singularities in z."]


def classify_singularities_from_features(features: ExpressionFeatures) -> list[dict[str, Any]]:
    singularities: list[dict[str, Any]] = []
    for expr in features.identically_zero_denominator_exprs:
        singularities.append({
            "kind": "undefined_everywhere",
            "source": expr,
            "reason": "This denominator simplifies to zero identically.",
        })
    for expr in features.essential_singularity_exprs:
        singularities.append({
            "kind": "essential_or_nonisolated",
            "source": expr,
            "reason": "A pole-like singularity appears inside a transcendental function.",
        })
    for expr in features.branchy_power_exprs:
        singularities.append({
            "kind": "branch_point",
            "source": expr,
            "reason": "Non-integer or variable powers need a branch choice.",
        })
    branch_functions = sorted((set(features.used_names) - SAFE_CONSTANTS) & BRANCHY_NAMES)
    for name in branch_functions:
        singularities.append({
            "kind": "branch_cut",
            "source": name,
            "reason": "This function is multi-valued or uses a principal branch.",
        })
    for expr in features.effective_denominator_exprs:
        singularities.append({
            "kind": "possible_pole",
            "source": expr,
            "reason": "Zeros of this denominator may be poles unless cancelled.",
        })
    for name in sorted(_active_known_pole_names(features)):
        singularities.append({
            "kind": "known_pole_family",
            "source": name,
            "reason": "This function has a known family of poles in a pole-producing context.",
        })
    return singularities


def classify_singularities(expr: str, *, deep: bool = False) -> list[dict[str, Any]]:
    return classify_singularities_from_features(analyze_expression(expr, deep=deep))


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


@lru_cache(maxsize=512)
def analyze_expression(expr: str, deep: bool = False) -> ExpressionFeatures:
    tree, _ = _parsed_expression(expr)
    feature_collector = ExpressionFeatureCollector(deep=deep)
    feature_collector.visit(tree)
    used_names = frozenset(feature_collector.used_names)

    denominator_exprs = tuple(dict.fromkeys(feature_collector.denominators))
    effective_denominator_exprs = tuple(dict.fromkeys(feature_collector.effective_denominators))
    provably_nonzero_denominator_exprs = tuple(dict.fromkeys(feature_collector.provably_nonzero_denominators))
    identically_zero_denominator_exprs = tuple(dict.fromkeys(feature_collector.identically_zero_denominators))
    branchy_power_exprs = tuple(dict.fromkeys(feature_collector.branchy_power_exprs))
    essential_singularity_exprs = tuple(dict.fromkeys(feature_collector.essential_singularity_exprs))
    nonanalytic_operator_exprs = tuple(dict.fromkeys(feature_collector.nonanalytic_operator_exprs))
    variable_order_special_calls = tuple(dict.fromkeys(feature_collector.variable_order_special_calls))

    functional_names = used_names - SAFE_CONSTANTS
    call_names = feature_collector.used_call_names
    has_branchy_power = bool(branchy_power_exprs)
    has_branchy = bool(functional_names & BRANCHY_NAMES) or has_branchy_power
    has_nonanalytic = bool(functional_names & NONANALYTIC_NAMES) or bool(nonanalytic_operator_exprs)
    has_piecewise = "where" in functional_names or "piecewise" in functional_names
    has_denominator = bool(denominator_exprs)
    has_effective_denominator = bool(effective_denominator_exprs)
    has_undefined_denominator = bool(identically_zero_denominator_exprs)
    has_essential_singularity = bool(essential_singularity_exprs)
    has_variable_order_special_function = bool(variable_order_special_calls)
    active_known_pole_names = frozenset(feature_collector.used_pole_function_names)
    uses_known_pole_families = bool(active_known_pole_names)

    theorem_eligible = (
        not has_branchy
        and not has_nonanalytic
        and not has_piecewise
        and not has_essential_singularity
        and not has_undefined_denominator
        and not has_variable_order_special_function
    )

    allowed_entire_names = ENTIRE_FUNCTIONS | SAFE_CONSTANTS
    whole_function_call_names = call_names - VARIABLE_ORDER_SPECIAL_FUNCTIONS
    variable_order_names = call_names & VARIABLE_ORDER_SPECIAL_FUNCTIONS
    constant_order_special_ok = not (variable_order_names and has_variable_order_special_function)
    proven_entire = (
        theorem_eligible
        and not has_effective_denominator
        and not uses_known_pole_families
        and constant_order_special_ok
        and functional_names <= allowed_entire_names
        and whole_function_call_names <= allowed_entire_names
    )

    return ExpressionFeatures(
        expr=preprocess(expr),
        used_names=used_names,
        denominator_exprs=denominator_exprs,
        effective_denominator_exprs=effective_denominator_exprs,
        provably_nonzero_denominator_exprs=provably_nonzero_denominator_exprs,
        identically_zero_denominator_exprs=identically_zero_denominator_exprs,
        branchy_power_exprs=branchy_power_exprs,
        essential_singularity_exprs=essential_singularity_exprs,
        nonanalytic_operator_exprs=nonanalytic_operator_exprs,
        variable_order_special_calls=variable_order_special_calls,
        has_branchy=has_branchy,
        has_nonanalytic=has_nonanalytic,
        has_piecewise=has_piecewise,
        has_denominator=has_denominator,
        has_effective_denominator=has_effective_denominator,
        has_undefined_denominator=has_undefined_denominator,
        has_branchy_power=has_branchy_power,
        has_essential_singularity=has_essential_singularity,
        has_variable_order_special_function=has_variable_order_special_function,
        uses_known_pole_families=uses_known_pole_families,
        active_known_pole_names=active_known_pole_names,
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
