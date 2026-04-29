# Complex Function Plotter

A Flask app for visualizing and integrating complex functions.

**Live instance:** [complex.princessevelyn.com](https://complex.princessevelyn.com)

## Features

### Plotting modes

- **Colors** — domain coloring: hue = argument, brightness varies with magnitude. Non-finite values are shown in pale blue.
- **Vectors** — sampled points `z` connected to `f(z)` with capped arrows; color encodes displacement length.
- **Transformation** — an input grid smoothly interpolating to its image under `f`, animated with Play/Pause.
  Transformation mode can also highlight a chosen source curve, including axes, vertical lines,
  horizontal lines, diagonal lines, or off-center circles, so its image can be tracked during the
  homotopy.

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
- full lines from `-∞` to `∞`

After drawing, the app computes the contour integral with **Exact first** mode by default. When the expression is simple enough, the app reports both an exact symbolic value and a decimal approximation. Exact formulas are clickable and render as LaTeX. You can also switch to theorem/numeric fallback mode or numerical-only mode.

Exact mode currently handles:
- **Closed contours** where SymPy can enumerate isolated meromorphic singularities in the plot bounds, including polynomial denominators and common Laurent-series cases like `1/sin(z)` or `1/(exp(z)-1)`. Individual enclosed residues are shown exactly too.
- **Open contours** of conservatively recognized entire functions when SymPy finds an elementary antiderivative
- **Rays to infinity and full lines** when SymPy finds an antiderivative and finite symbolic limits at infinity. Meromorphic antiderivative support is conservative and currently limited to single improper line/ray paths. Full lines cannot be combined with other path segments.
- **Residue derivations** for classic real-axis integrals, including rational full-line integrals, even rational half-line integrals, Fourier/Jordan-lemma integrals of `exp(i*a*z)`, `cos(a*z)`, or `sin(a*z)` times a rational factor, and keyhole-contour rational half-line integrals.

For exact residues, the result panel includes local observability such as pole order, Laurent expansion snippets, and denominator series when available.

For closed contours, the app can also apply theorem shortcuts when the expression is in the app's conservative theorem-safe class:
- **Cauchy shortcut** for proven entire functions
- **Residue theorem** for closed contours enclosing isolated singularities, with numerically estimated residues from small circles

If a detected singularity lies on the path, the integral is reported as undefined for ordinary contour integration mode.

### Expression classification

As you type, the app classifies the expression in real time:
- **entire** — built from `z`, constants, and functions in the conservative entire-function class
- **entire after symbolic simplification** — deep mode simplified away detected branch or singular behavior
- **locally holomorphic / analytic** — component expressions that pass a Cauchy-Riemann check
- **probably holomorphic (unclassified singularities)** — no obvious issues detected, but not in the conservative entire class
- **holomorphic on a branch domain** — uses branch-cut functions
- **meromorphic** — has denominators or known pole families
- **meromorphic with removable singularities** — deep mode found only removable symbolic singularities
- **holomorphic except at isolated singularities** — has detected isolated non-meromorphic singular behavior, such as a pole inside `exp`
- **special-function analytic status uncertain** — uses special functions in a way the fast classifier does not try to prove
- **undefined / singular everywhere** — deep mode found a denominator that simplifies to zero identically
- **non-holomorphic** — uses `conj`, `real`, `imag`, `abs`, etc.
- **piecewise analytic** — uses `where` or `piecewise`

Click the **Why** button next to the class badge to see the reasoning and singularity notes. Click **Deep** to request a slower SymPy-backed pass.

When the expression simplifies to a nonconstant linear fractional transformation `(az+b)/(cz+d)`,
the class badge identifies it as a Möbius/LFT directly. The reasoning panel also reports
matrix coefficients, determinant, `tr²/det`, zero, pole, fixed points, and a conservative
type label such as elliptic, parabolic, hyperbolic, loxodromic, or identity.

### Zero highlighting

Toggle **Highlight zeros** to numerically detect roots of `f(z) = 0` and mark them on Colors and Vectors plots.

### Singularity highlighting

Toggle **Show singularities** to overlay classified singularity markers when the app can locate them in the current viewport. Pole, removable, essential-candidate, and branch-point labels use distinct marker styles.

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
- Rays to infinity and full lines can be exact when a symbolic antiderivative has finite limits at infinity; divergent, conditionally convergent, or symbolically ambiguous cases fall back to numerical quadrature and may still produce warnings or non-finite results.
- Bound inputs accept decimal numbers and fractions like `1/2` or `-3/4`.

Love you always and forever!~
	- Princess Evelyn 🧡
