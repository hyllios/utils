"""
Post-processing of the isotropic Eliashberg spectral function alpha2F(omega):
frequency moments (the running lambda(omega) and omega_log(omega)), the
second moment omega_2, and the McMillan/Allen-Dynes critical temperature.

Everything here consumes a plain ``(a2f, omega_grid)`` pair (Hartree, from
`analysis.elph.alpha2f` or any other source, e.g. a parsed EPW ``.a2f``
file) and knows nothing about the electron-phonon vertex itself -- that
machinery lives in `analysis.elph`. Integration follows EPW's own
convention: rectangle sums over the full grid, every omega > 0 point
contributing its bin (verified to machine precision against EPW's
``l_a2f``/``logavg`` sums).

Atomic units throughout, like the rest of `analysis`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lambda_matrix",
    "lambda_effective",
    "eliashberg_moments",
    "eliashberg_omega_2",
    "allen_dynes_tc",
    "allen_dynes_tc_from_a2f",
    "lambda_from_a2f",
]


def lambda_matrix(a2f_ij: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    """The el-ph coupling matrix ``lambda_ij = 2 integral alpha2F_ij(w)/w dw``
    from `alpha2f_matrix`'s ``alpha2F_ij`` -- per-element `lambda_from_a2f`.
    Total ``lambda = sum_ij (N_i / sum N) lambda_ij``."""
    n = a2f_ij.shape[0]
    return np.array([[lambda_from_a2f(a2f_ij[i, j], omega_grid)
                      for j in range(n)] for i in range(n)])


def lambda_effective(lam_ij: np.ndarray) -> float:
    """
    Effective coupling of a multiband system: the largest eigenvalue of the
    interband coupling matrix ``lambda_ij`` (from `lambda_matrix`).

    This is the standard multiband-McMillan reduction (e.g. the two-band
    treatment of MgB2): the linearized multiband gap equations
    ``Delta_i = sum_j lambda_ij Delta_j * L(T)`` reach Tc when the largest
    eigenvalue of lambda_ij satisfies the single-band condition, so
    ``lambda_eff = max eig(lambda_ij)`` slots directly into
    `allen_dynes_tc` in place of the isotropic lambda. Note lambda_ij is
    NOT symmetric (``N_i lambda_ij = N_j lambda_ji``); the eigenvalues are
    real nonetheless (similar to a symmetric matrix via the N_i weights).

    For MgB2 this is the difference between the isotropic lambda ~0.7 (Tc
    underestimated) and lambda_eff ~1.0 dominated by the sigma-sigma block.
    """
    lam_ij = np.asarray(lam_ij, dtype=np.float64)
    ev = np.linalg.eigvals(lam_ij)
    return float(np.max(ev.real))


def eliashberg_moments(a2f: np.ndarray, omega_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Running (cumulative-in-frequency) Eliashberg moments:

      lambda(w)     = 2 * integral_0^w  alpha2F(w')/w'          dw'
      omega_log(w)  = exp[ (2/lambda(w)) * integral_0^w
                           alpha2F(w') log(w')/w'  dw' ]

    ``lambda(w->infinity)`` is the usual el-ph mass enhancement (the
    single number `alpha2f`'s own docstring/callers compute directly);
    ``omega_log(w->infinity)`` is the Allen-Dynes logarithmic-average
    phonon frequency, the other ingredient (with lambda) the McMillan-
    Allen-Dynes Tc formula needs. Plotting these as a function of the
    upper cutoff w is a standard diagnostic for which part of the phonon
    spectrum dominates lambda.

    Parameters
    ----------
    a2f : (nE,) float64
        alpha2F(omega) on ``omega_grid`` (Hartree), e.g. `alpha2f`'s
        output. A grid point at exactly omega = 0 contributes nothing
        (1/omega divergence); every point with omega > 0 contributes its
        rectangle-rule bin, INCLUDING the first one -- EPW's own grid is
        ``w_i = i * dw, i = 1..nqstep`` and its `l_a2f`/`logavg` sums run
        over every point of it, so skipping index 0 on such a grid would
        silently lose the first bin and underestimate lambda.
    omega_grid : (nE,) float64, Hartree

    Returns
    -------
    lambda_cum, omega_log_cum : (nE,) float64 each. Where too little has
        been integrated for the ratio to mean anything the omega_log
        entries are NaN.
    """
    domega = omega_grid[1] - omega_grid[0]
    pos = omega_grid > 0.0
    inv_w = np.where(pos, a2f / np.where(pos, omega_grid, 1.0), 0.0)
    lambda_cum = 2.0 * np.cumsum(inv_w) * domega
    log_integrand = np.where(
        pos, inv_w * np.log(np.where(pos, omega_grid, 1.0)), 0.0)
    log_cum = np.cumsum(log_integrand) * domega

    # Near w=0, almost nothing has been integrated yet -- lambda(w) is
    # essentially zero and omega_log(w) is mathematically ill-defined
    # there (not just numerically small); mark those points NaN (a clean
    # gap when plotted) rather than an overflowing exp(huge number).
    valid = lambda_cum > 1e-6
    safe_lambda = np.where(valid, lambda_cum, 1.0)
    with np.errstate(over="ignore"):
        omega_log_valid = np.exp((2.0 / safe_lambda) * log_cum)
    omega_log_cum = np.where(valid, omega_log_valid, np.nan)
    return lambda_cum, omega_log_cum


def eliashberg_omega_2(a2f: np.ndarray, omega_grid: np.ndarray) -> float:
    """
    Second moment of the Eliashberg function,

      omega_2 = sqrt( (2/lambda) integral alpha2F(w) w dw ),

    the extra ingredient Allen-Dynes' strong-coupling factor ``f2`` needs on
    top of lambda and omega_log (`allen_dynes_tc`). Hartree.
    """
    a2f = np.asarray(a2f, dtype=np.float64)
    omega_grid = np.asarray(omega_grid, dtype=np.float64)
    domega = omega_grid[1] - omega_grid[0]
    lam = lambda_from_a2f(a2f, omega_grid)
    if not lam > 0.0:
        return float("nan")
    return float(np.sqrt(2.0 / lam * float((a2f * omega_grid).sum()) * domega))


def allen_dynes_tc(
    lambda_ep: float, omega_log: float, omega_2: float,
    mu_star: float = 0.1,
) -> float:
    """
    Allen-Dynes superconducting Tc, returned as the ENERGY ``k_B * Tc`` in
    Hartree (atomic units throughout `analysis`; divide by
    ``units.K_B_HARTREE`` for Kelvin).

      k_B Tc = (f1 f2 omega_log / 1.2)
               exp[ -1.04 (1 + lambda) / (lambda - mu* (1 + 0.62 lambda)) ]

    with the strong-coupling and shape corrections

      f1 = [1 + (lambda / L1)^(3/2)]^(1/3),   L1 = 2.46 (1 + 3.8 mu*)
      f2 = 1 + (omega_2/omega_log - 1) lambda^2 / (lambda^2 + L2^2),
                                              L2 = 1.82 (1 + 6.3 mu*) omega_2/omega_log

    ``omega_2`` is REQUIRED, deliberately: pass `eliashberg_omega_2`'s output,
    or use `allen_dynes_tc_from_a2f` which derives all three inputs from
    alpha2F so none can be forgotten. Dropping f2 is not a harmless
    simplification -- it is the difference between the plain McMillan form
    and Allen-Dynes proper, it always LOWERS Tc (f2 >= 1 whenever omega_2 >
    omega_log, which holds for any spread-out spectrum), and it grows with
    coupling: negligible for Al at lambda ~ 0.5, several per cent for MgB2
    near lambda ~ 1. To recover the plain form deliberately, pass
    ``omega_2 = omega_log``, which makes f2 exactly 1.

    Asymptotics, as a check on the implementation: at mu* = 0 and large
    lambda, f1 -> sqrt(lambda/2.46), f2 -> omega_2/omega_log and the
    exponential saturates at exp(-1.04), giving Allen-Dynes' strong-coupling
    limit ``k_B Tc -> 0.188 sqrt(lambda) omega_2`` against their quoted
    0.183 (theirs being a fit to numerical solutions, not the closed form).

    Returns NaN when ``lambda <= mu*(1 + 0.62 lambda)``: the exponent's
    denominator is then non-positive and the formula has no meaning. That is
    not an edge case to paper over -- it happens routinely on coarse meshes
    for weakly coupled metals, i.e. at the start of any convergence sequence,
    and a NaN is the honest signal that Tc is not yet defined there.

    Note Tc depends EXPONENTIALLY on lambda: at Al's values a 0.03 shift in
    lambda moves Tc by ~28%, while 3% on omega_log moves it by 3%. Any error
    budget aimed at Tc is therefore a budget on lambda.
    """
    lambda_ep, omega_log, mu_star = float(lambda_ep), float(omega_log), float(mu_star)
    denom = lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)
    if not denom > 0.0 or not omega_log > 0.0:
        return float("nan")
    lam1 = 2.46 * (1.0 + 3.8 * mu_star)
    f1 = (1.0 + (lambda_ep / lam1) ** 1.5) ** (1.0 / 3.0)
    omega_2 = float(omega_2)
    if not (np.isfinite(omega_2) and omega_2 > 0.0):
        return float("nan")
    r = omega_2 / omega_log
    lam2 = 1.82 * (1.0 + 6.3 * mu_star) * r
    f2 = 1.0 + (r - 1.0) * lambda_ep ** 2 / (lambda_ep ** 2 + lam2 ** 2)
    return float(f1 * f2 * omega_log / 1.2
                 * np.exp(-1.04 * (1.0 + lambda_ep) / denom))


def allen_dynes_tc_from_a2f(
    a2f: np.ndarray, omega_grid: np.ndarray, mu_star: float = 0.1,
) -> tuple[float, float, float, float]:
    """
    Allen-Dynes Tc straight from alpha2F, deriving every ingredient so none can
    be omitted -- the recommended entry point.

    Returns ``(k_B*Tc, lambda, omega_log, omega_2)``, all energies in Hartree
    (divide Tc by ``units.K_B_HARTREE`` for Kelvin). Equivalent to calling
    `lambda_from_a2f`, `eliashberg_moments` and `eliashberg_omega_2` and feeding
    all three to `allen_dynes_tc`, which is exactly why it exists: the f2 shape
    factor needs the second moment, and a signature that lets the caller supply
    only lambda and omega_log invites the plain-McMillan answer by accident.
    """
    lam = lambda_from_a2f(a2f, omega_grid)
    omega_log = float(eliashberg_moments(a2f, omega_grid)[1][-1])
    omega_2 = eliashberg_omega_2(a2f, omega_grid)
    return (allen_dynes_tc(lam, omega_log, omega_2, mu_star),
            lam, omega_log, omega_2)


def lambda_from_a2f(a2f: np.ndarray, omega_grid: np.ndarray) -> float:
    """lambda = 2 integral alpha2F(w)/w dw (the omega->infinity limit of
    `eliashberg_moments`'s running lambda), as a plain scalar."""
    return float(eliashberg_moments(a2f, omega_grid)[0][-1])
