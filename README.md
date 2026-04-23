# Complex Function Plotter

A small Flask app for visualizing and integrating complex functions.

## Features

### Plotting modes
- **Colors**: domain coloring
- **Vectors**: sampled points `z` connected to `f(z)`
- **Transformation**: an input grid smoothly interpolating to its image under `f`

### Integration mode
- click-built paths using:
  - line segments
  - full circles
  - circular arcs
  - quadratic Bézier curves
  - cubic Bézier curves
  - freeform polylines
  - rays to infinity
- reverse orientation and close loops easily
- server-side contour integration
- conservative theorem shortcutting:
  - Cauchy shortcut for a conservative entire-function class
  - residue theorem for closed contours when the expression is in the app's conservative theorem-safe class and enclosed isolated singularities are found
  - residues may be estimated numerically from small circles around singularities
- if a detected singularity lies on the path, the app reports the integral as undefined for ordinary contour integration mode

## Run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local Flask address that prints in the terminal.

## Expression syntax

Use Python-style expressions in `z`.

Examples:
- `z^2`
- `1/z`
- `exp(z)`
- `sin(z)`
- `where(x > 0, gamma(z), conj(z))`
- `piecewise(x < 0, z, x >= 0, conj(z), 0)`
- `gamma(z)`
- `erf(z)`
- `zeta(z)`

`^` is accepted as shorthand for `**`.

## Supported names

### Variable and component aliases
- `z`
- `x = Re(z)`
- `y = Im(z)`

### Constants
- `pi`, `e`, `tau`, `i`, `j`

### Elementary functions
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`
- `sinh`, `cosh`, `tanh`
- `asinh`, `acosh`, `atanh`
- `exp`, `log`, `ln`, `log10`, `sqrt`
- `sec`, `csc`, `cot`, `sech`, `csch`, `coth`

### Component / helper functions
- `abs`
- `conj`, `conjugate`
- `real`, `imag`, `re`, `im`
- `angle`, `arg`
- `cis`
- `sinc`
- `where(cond, a, b)`
- `piecewise(cond1, val1, cond2, val2, default)`

### Special functions
- `gamma`, `loggamma`, `rgamma`
- `digamma`, `psi`
- `erf`, `erfc`, `erfi`, `erfcx`
- `wofz`, `dawsn`
- `zeta`, `zetac`
- `lambertw`
- `expi`, `exp1`
- `fresnels`, `fresnelc`
- `airyai`, `airyaip`, `airybi`, `airybip`
- `jv`, `yv`, `iv`, `kv`

## Notes

- Integration mode is intentionally conservative about theorem shortcuts.
- Branch cuts and non-analytic expressions fall back to direct numerical path integration.
- Principal value integrals are **not** implemented automatically.
- Rays to infinity are supported, but divergent or conditionally convergent cases may still produce warnings or non-finite results.
