# Complex Function Plotter

A Flask app for visualizing and integrating complex functions.

**Live instance:** [complex.princessevelyn.com](https://complex.princessevelyn.com)

## Features

### Plotting modes

- **Colors** — domain coloring: hue = argument, brightness varies with magnitude. Non-finite values are shown in pale blue.
- **Vectors** — sampled points `z` connected to `f(z)` with capped arrows; color encodes displacement length.
- **Transformation** — an input grid smoothly interpolating to its image under `f`, animated with Play/Pause.

All plot modes support scroll/pan zoom, pinch-to-zoom on touch devices, and a hover/click probe that evaluates `f(z)` at the cursor position.

### Integration mode

Click inside the plot to build a path from:
- line segments
- full circles
- circular arcs
- quadratic Bézier curves
- cubic Bézier curves
- freeform polylines
- rays to infinity

After drawing, the app computes the contour integral numerically. For closed contours, it can automatically apply theorem shortcuts when the expression is in the app's conservative theorem-safe class:
- **Cauchy shortcut** for proven entire functions
- **Residue theorem** for closed contours enclosing isolated singularities, with numerically estimated residues from small circles

If a detected singularity lies on the path, the integral is reported as undefined for ordinary contour integration mode.

### Expression classification

As you type, the app classifies the expression in real time:
- **entire** — built from `z`, constants, and functions in the conservative entire-function class
- **holomorphic / analytic** — no branch cuts, denominators, or non-holomorphic operations detected
- **holomorphic on a branch domain** — uses branch-cut functions
- **meromorphic** — has denominators or known pole families
- **non-holomorphic** — uses `conj`, `real`, `imag`, `abs`, etc.
- **piecewise analytic** — uses `where` or `piecewise`

Click the **Why** button next to the class badge to see the reasoning, including whether a numerical Cauchy-Riemann check passed.

### Zero highlighting

Toggle **Highlight zeros** to numerically detect roots of `f(z) = 0` and mark them on Colors and Vectors plots.

### Auto re-render

Toggle **Auto re-render** to automatically resample the plot using the current viewport bounds after panning or zooming.

## Run

### Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local Flask address printed in the terminal.

You will need a `.env` file with at least:

```
FLASK_SECRET_KEY=your-secret-key-here
```

### Production

The repo includes deployment configs for nginx + gunicorn + systemd:

- [`gunicorn.conf.py`](gunicorn.conf.py) — gunicorn settings
- [`deploy/complex.service`](deploy/complex.service) — systemd service
- [`deploy/complex.princessevelyn.com`](deploy/complex.princessevelyn.com) — nginx site config
- [`deploy/deploy.sh`](deploy/deploy.sh) — deploy script

## Expression syntax

Use Python-style expressions in `z`.

- `^` is accepted as shorthand for `**`
- Implicit multiplication is supported: `2z`, `z sin(z)`
- Logical comparisons work inside `where` and `piecewise`: `(x >= -1) & (x <= 1)`

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
- Bound inputs accept decimal numbers and fractions like `1/2` or `-3/4`.

Love you always and forever!~
	- Princess Evelyn 🧡
