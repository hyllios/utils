"""
Command line front end:

    python -m waw.utils.eliashberg a2f.npz --mu 0.11 --omega-c 500meV
    python -m waw.utils.eliashberg a2f.dat --mu 0.11 --unit cm-1
    python -m waw.utils.eliashberg a2f.npz --mu 0.11 --scan 0.08 0.16 5

Input is either

  * an ``.npz`` holding ``omega`` (Hartree unless ``--unit`` says otherwise) and
    ``a2f`` of shape ``(n_omega,)`` or ``(nb, nb, n_omega)`` -- i.e. what
    ``np.savez`` on `waw.analysis.elph.alpha2f_matrix` output produces; or
  * a whitespace-delimited text file whose first column is omega and whose
    remaining columns are the alpha^2F blocks in row-major (i, j) order: one
    column for a single band, nb*nb columns for nb bands.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from waw.analysis.eliashberg import lambda_matrix
from waw.units import CM1_TO_HARTREE, EV_TO_HARTREE, HARTREE_TO_EV

from .kernels import as_band_matrix, isotropic_average
from .linearized import tc_linearized

#: multiply a frequency in the named unit to get Hartree
_TO_HARTREE = {
    "ha": 1.0, "hartree": 1.0, "ry": 0.5,
    "ev": EV_TO_HARTREE, "mev": 1e-3 * EV_TO_HARTREE,
    "cm-1": CM1_TO_HARTREE, "cm": CM1_TO_HARTREE,
    "thz": 4.13566769692e-3 * EV_TO_HARTREE,
}


def _unit_factor(unit: str) -> float:
    key = unit.strip().lower()
    if key not in _TO_HARTREE:
        raise ValueError(f"unknown unit {unit!r}; known: {sorted(_TO_HARTREE)}")
    return _TO_HARTREE[key]


def _energy(text: str) -> float:
    """Parse '500meV' / '0.5eV' / '0.018Ha' into Hartree. Bare number = meV."""
    t = text.strip().lower()
    for suffix in sorted(_TO_HARTREE, key=len, reverse=True):
        if t.endswith(suffix) and t[: -len(suffix)].strip():
            return float(t[: -len(suffix)]) * _TO_HARTREE[suffix]
    return float(t) * _TO_HARTREE["mev"]


def load_a2f(path, unit: str = "Ha"):
    """
    Read ``(omega, a2f)`` from an .npz or a columnar text file.

    omega comes back in Hartree with any non-positive point dropped, since
    alpha^2F/omega diverges at zero; a2f comes back as ``(nb, nb, n_omega)``.
    """
    factor = _unit_factor(unit)
    if str(path).endswith(".npz"):
        z = np.load(path)
        for key in ("omega", "a2f"):
            if key not in z:
                raise ValueError(f"{path}: .npz must contain '{key}'")
        omega = np.asarray(z["omega"], dtype=np.float64) * factor
        a2f = as_band_matrix(np.asarray(z["a2f"], dtype=np.float64), len(omega))
    else:
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError(f"{path}: need at least two columns (omega, alpha^2F)")
        omega = raw[:, 0] * factor
        cols = raw[:, 1:]
        nb = int(round(np.sqrt(cols.shape[1])))
        if nb * nb != cols.shape[1]:
            raise ValueError(
                f"{path}: {cols.shape[1]} alpha^2F columns is not a perfect "
                f"square, so the band matrix shape is ambiguous"
            )
        a2f = cols.T.reshape(nb, nb, -1)
    keep = omega > 0.0
    return omega[keep], a2f[..., keep]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m waw.utils.eliashberg",
        description="Tc from the linearized band-resolved Eliashberg equations.",
    )
    p.add_argument("a2f", help=".npz with omega/a2f, or a columnar text file")
    p.add_argument("--unit", default="Ha",
                   help="unit of omega in the input file (default: Ha)")
    p.add_argument("--mu", type=float, required=True,
                   help="Coulomb pseudopotential mu*, quoted at --omega-c")
    p.add_argument("--omega-c", type=_energy,
                   help="Coulomb cutoff, e.g. 500meV (default: 5x max phonon)")
    p.add_argument("--omega-max", type=_energy,
                   help="Matsubara extent (default: 40x max phonon)")
    p.add_argument("--n-matsubara", type=int, help="pin the Matsubara count")
    p.add_argument("--scan", nargs=3, metavar=("MU_LO", "MU_HI", "N"),
                   help="scan mu* over a range instead of a single value")
    p.add_argument("--dense", action="store_true",
                   help="diagonalise fully instead of power iteration")
    args = p.parse_args(argv)

    omega, a2f = load_a2f(args.a2f, args.unit)
    mev = HARTREE_TO_EV * 1e3                 # Hartree -> meV, for reporting
    nb = a2f.shape[0]
    lam = lambda_matrix(a2f, omega)

    print(f"{args.a2f}: {nb} band(s), {len(omega)} frequencies")
    print(f"  phonon range   : {omega.min()*mev:.3f} .. {omega.max()*mev:.3f} meV")
    if nb == 1:
        print(f"  lambda         : {lam[0, 0]:.4f}")
    else:
        rows = np.array2string(lam, precision=4).splitlines()
        print("  lambda_ij      : " + ("\n" + " " * 19).join(rows))
        iso = isotropic_average(a2f)
        print(f"  isotropic sum  : {np.trapezoid(2 * iso / omega, omega):.4f}"
              f"   (equal DOS weights)")

    kw = dict(omega_c=args.omega_c, omega_max_matsubara=args.omega_max,
              method="dense" if args.dense else "power")
    if args.n_matsubara:
        kw["n_matsubara"] = args.n_matsubara

    if args.scan:
        lo, hi, n = float(args.scan[0]), float(args.scan[1]), int(args.scan[2])
        print(f"\n  {'mu*':>8s}  {'Tc (K)':>10s}  {'n_mats':>7s}")
        for m in np.linspace(lo, hi, n):
            r = tc_linearized(a2f, omega, m, **kw)
            print(f"  {m:8.4f}  {r.tc:10.3f}  {r.n_matsubara:7d}")
        return 0

    r = tc_linearized(a2f, omega, args.mu, **kw)
    print(f"  omega_c        : {r.omega_c*mev:.1f} meV")
    print(f"  mu*            : {args.mu}")
    print(f"  Matsubara pts  : {r.n_matsubara}  ({r.n_evaluations} evaluations)")
    print(f"  Tc             : {r.tc:.4f} K   (rho = {r.rho_at_tc:.6f})")
    if r.tc == 0.0:
        print("  -> does not superconduct: mu* exceeds the coupling")
    elif r.gap_symmetry is not None and nb > 1:
        signs = "".join("+" if s > 0 else "-" for s in r.gap_symmetry)
        print(f"  gap signs      : {signs}   "
              f"({'s++' if len(set(signs)) == 1 else 's+-'} solution)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
