"""
The STANDALONE analytic SCDFT functional: explicit xi-space kernels, no
Eliashberg kernel anywhere.

WHY THIS EXISTS. The `unexpanded` module proved that the Sham-Schlueter
condition inverts Migdal-Eliashberg essentially exactly (its Galerkin closure
lands at 1e-4 of the reference Tc), but its profile basis is built by applying
the Eliashberg kernel -- so, as a functional, it can never contain less or
more than Eliashberg and costs about as much. This module is the opposite
trade: kernels that are EXPLICIT functions of (xi, xi'), evaluated with no
Matsubara kernel matrix, no profile iteration, and nothing fitted -- the same
class of object as the Luders-Marques (LM2005) and Sanna-Pellegrini-Gross
(SPG) functionals -- at the price of a measured few-percent accuracy loss.
Its two structural advantages over LM2005/SPG:

  * the propagator lines are Z-DRESSED (phi = Z Delta and w^2 Z^2
    denominators), which is exactly what removes the 2-lambda disease at the
    root instead of repairing it (LM2005) or fitting it away (SPG);
  * the Coulomb channel takes an arbitrary STATICALLY SCREENED W(xi, xi'),
    not a scalar mu, and it is EXACT for any static W: a static interaction
    couples only to the frequency-summed anomalous density, which is
    precisely the quantity the SS condition fixes. No mu*, no cutoff: the
    Morel-Anderson reduction emerges from the solve.

THE CONSTRUCTION (per band; flat DOS folded into a2f as usual).

Two-channel decomposition of the anomalous self-energy: for static W,

    phi_n(xi) = psi_n(xi) + c(xi),          c exactly frequency-flat,
    c(xi)  = - Int dxi' w(xi,xi') g(xi') Delta_s(xi'),
    g(a)   = tanh(beta a/2)/(2a)   [= T P_s],

and only the phonon part psi carries a shape: psi_m ~ a(xi') s_m with the
Lorentzian s_m = W_s^2/(W_s^2 + w_m^2), W_s = w_ln. The SS condition then
inverts POINTWISE with no singularity -- unlike the single-profile closure of
the derivation notes, whose normalising sum has a zero for any mu > 0 --
because sigma_s = T sum_m s_m/D_m > 0 termwise:

    a(xi') = [g(xi') Delta(xi') - c(xi') h_Z(xi')] / sigma_s(xi').

The gap equation, a linear eigenproblem on the xi grid alone:

    Delta(xi) g(xi) = Int dxi' [Y_s(xi,xi') a(xi') + Y_1(xi,xi') c(xi')]
                      + c(xi) h_Z(xi).

DRESSINGS -- the part that was measured, not guessed. Phonon-channel lines
carry a constant Zbar = 1 + lambda (the exact Z(0); the pairing sums are
dominated by the lowest Matsubara frequencies), which makes every double sum
close through ONE master function

    S(x,y,W) = T^2 sum_{nm} 2W/(W^2+(w_n-w_m)^2) / [(w_n^2+x^2)(w_m^2+y^2)]
             = (1/4xy) sum_{s,s'} s s' [Q(sx,s'y,W) + Q(s'y,sx,W)],
    Q(a,b,W) = [1+n_B(W)-f(b)] [f(a)-f(b+W)] / (a-b-W),

validated against brute-force double sums to 1e-11. The FLAT channel line
h_Z(xi) = T sum_n 1/(w_n^2 Z_n^2 + xi^2) uses the TRUE finite-T normal-state
Z_n (a cheap one-dimensional sum): h_Z appears in chi_c, in the a-inversion
and (implicitly) in Y_1's m-line, and the dressing mismatch between them is
amplified by 1/sigma_s ~ xi'^2. Measured on the Einstein benchmark at
lambda=0.5, mu=0.2: consistent-but-constant Zbar everywhere gives 0.78 of the
reference Tc (it honestly solves the wrong model -- constant-Z Eliashberg); a
two-step Z in h_Z alone gives 1.07 (inconsistent with Y_1 at mid
frequencies); the true-Z h_Z gives 1.03, because lambda(n-m) confines Y_1's
m-line to low frequencies where Z ~ Zbar, so the residual mismatch lives only
where the phonon kernel has no weight.

VALIDATION (scripts/analytic_*.py of the derivation notes; tail-completed
band-limited Migdal-Eliashberg references, structured-W case against a
(n,xi)-resolved ME solve):

    Einstein, lambda = 0.5-1.5 x mu = 0-0.3 : Tc ratio 0.97-1.03
    structured W (separable/entangled),
      E_scr = Wb/3, mu(0,0) = 0.2-0.3      : 1.01-1.03  (1.06 at mu = 0.4)

Zero fitted parameters. For comparison, SPG reach a similar accuracy class
with three constants fitted against Eliashberg; LM2005 is qualitatively wrong
at the Fermi surface (2 lambda).

LIMITS. Linearised at Tc (the profile shape and the linearised SS condition;
below-Tc use is untested exactly as for `unexpanded`); flat DOS per band;
isotropic within a band; multiband W currently a (nb, nb) matrix of scalars
or a per-band-pair callable in xi, xi'. Since no Matsubara kernel matrix is
ever formed, there is NO low-temperature floor: cost per temperature is a
single 1-D sum for h_Z plus O(n_xi^2 n_Omega) kernel evaluations, so
realistic multi-eV band widths are affordable -- unlike `unexpanded`, whose
dense Matsubara grid must span the band.

Atomic units in, Kelvin out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waw.units import K_B_HARTREE

from ..eliashberg.kernels import as_band_matrix
from .unexpanded import _mu_matrix, _omega_log_rows

__all__ = ["AnalyticKernels", "build_analytic", "linearized_eigenvalue_analytic",
           "tc_analytic"]

#: |w|max / band_edge the 1-D h_Z sum must reach (same physics as
#: unexpanded._MATSUBARA_REACH: the flat channel lives on the whole axis).
_HZ_REACH = 4.0


@dataclass
class AnalyticKernels:
    """The assembled operator and its ingredients, kept for inspection."""
    xi:      np.ndarray      # (n_xi, nb) per-band energy grid, Hartree
    weights: np.ndarray      # (n_xi, nb) quadrature weights (xi' in (-Wb, Wb))
    g:       np.ndarray      # (n_xi, nb) bare pair weight tanh(beta xi/2)/(2 xi)
    h_z:     np.ndarray      # (n_xi, nb) true-Z flat-channel weight
    sigma_s: np.ndarray      # (n_xi, nb) s-channel weight, positive definite
    operator: np.ndarray     # (nb*n_xi, nb*n_xi) gap operator, Delta -> Delta


def _gfun(a, beta: float):
    a = np.asarray(a, dtype=np.float64)
    safe = np.where(np.abs(a) < 1e-300, 1.0, a)
    return np.where(np.abs(a) < 1e-300, 0.25 * beta,
                    np.tanh(0.5 * beta * safe) / (2.0 * safe))


def _fermi(e, beta: float):
    return 0.5 * (1.0 - np.tanh(0.5 * beta * e))


def _q_master(a, b, w, beta: float):
    """Q(a,b,W) with the removable pole a-b-W = 0 taken by its limit."""
    d = a - b - w
    with np.errstate(over="ignore"):
        nb = 1.0 / np.expm1(beta * w)
    pref = 1.0 + nb - _fermi(b, beta)
    small = np.abs(d) * beta < 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pref * (_fermi(a, beta) - _fermi(b + w, beta)) / d
    arg = np.clip(0.5 * beta * (b + w), -350.0, 350.0)
    fp = -0.25 * beta / np.cosh(arg) ** 2
    return np.where(small, pref * fp, out)


def _s_master(x, y, w, beta: float):
    """S(x,y,W); x (rows) and y (cols) broadcast, w scalar."""
    x = np.asarray(x, dtype=np.float64)[:, None]
    y = np.asarray(y, dtype=np.float64)[None, :]
    tot = 0.0
    for s in (1.0, -1.0):
        for sp in (1.0, -1.0):
            tot = tot + s * sp * (_q_master(s * x, sp * y, w, beta)
                                  + _q_master(sp * y, s * x, w, beta))
    return tot / (4.0 * x * y)


def _s_spectrum(x, y, omega, a2f_ij, beta: float):
    """Int dW a2F(W) S(x,y,W) over the phonon spectrum of one band pair."""
    omega = np.atleast_1d(omega)
    if len(omega) == 1:
        return float(a2f_ij[0]) * _s_master(x, y, float(omega[0]), beta)
    acc = np.zeros((len(x), len(y)))
    wts = np.gradient(omega)
    for k, (wk, ak, dk) in enumerate(zip(omega, a2f_ij, wts)):
        if ak == 0.0:
            continue
        acc += (ak * dk) * _s_master(x, y, float(wk), beta)
    return acc


def _lam_row_d(omega, a_row, kT: float, n_d: int, chunk: int = 65536):
    """Row-summed lambda_i(d) = Int dW [sum_j a2F_ij] 2W/(W^2+nu_d^2) for
    d = 0..n_d-1, chunked in d so no (n_d, n_omega) monolith is formed (at
    low T n_d can reach 1e5)."""
    omega = np.atleast_1d(omega)
    out = np.empty(n_d)
    if len(omega) == 1:
        nu = 2.0 * np.pi * kT * np.arange(n_d, dtype=np.float64)
        return 2.0 * float(a_row[0]) * omega[0] / (omega[0] ** 2 + nu ** 2)
    for k0 in range(0, n_d, chunk):
        d = np.arange(k0, min(k0 + chunk, n_d), dtype=np.float64)
        nu2 = (2.0 * np.pi * kT * d) ** 2
        wgt = 2.0 * omega[None, :] / (omega[None, :] ** 2 + nu2[:, None])
        out[k0:k0 + len(d)] = np.trapezoid(wgt * a_row[None, :], omega, axis=1)
    return out


def _h_z_true(xi, beta: float, omega, a, reach: float, band: int,
              chunk: int = 4096):
    """
    h_Z(xi) = T sum_n 1/(w_n^2 Z_n^2 + xi^2), true finite-T normal-state Z_n
    of the given band (row-summed lambda), plus the integral tail beyond
    `reach` where Z = 1. Z_n is built with cumulative sums, O(N), and the sum
    is chunked so memory stays bounded at any temperature.
    """
    kT = 1.0 / beta
    n_half = int(np.ceil(reach / (2.0 * np.pi * kT)))
    w = (2.0 * np.arange(n_half) + 1.0) * np.pi * kT
    lam_d = _lam_row_d(omega, a[band].sum(axis=0), kT, 2 * n_half)
    cs = np.concatenate([[0.0], np.cumsum(lam_d)])
    n = np.arange(n_half)
    pos = cs[n + 1] + cs[n_half - n] - lam_d[0]
    neg = cs[n + n_half + 1] - cs[n + 1]
    s2 = (w * (1.0 + (np.pi * kT / w) * (pos - neg))) ** 2
    out = np.zeros_like(xi)
    for k0 in range(0, n_half, chunk):
        blk = s2[k0:k0 + chunk]
        out += (1.0 / (blk[None, :] + xi[:, None] ** 2)).sum(axis=1)
    lam_edge = w[-1] + np.pi * kT
    return 2.0 * kT * out + (0.5 * np.pi - np.arctan(lam_edge / xi)) / (
        np.pi * xi)


def _coulomb_matrix(coulomb, xi, nb: int):
    """w_ij(xi, xi') on the per-band grids: scalar, (nb, nb), or callable."""
    n_xi = xi.shape[0]
    wmat = np.empty((nb, n_xi, nb, n_xi))
    if callable(coulomb):
        for i in range(nb):
            for j in range(nb):
                wmat[i, :, j, :] = coulomb(i, j, xi[:, i][:, None],
                                           xi[:, j][None, :])
    else:
        mu_m = _mu_matrix(coulomb, nb)
        wmat[:] = mu_m[:, None, :, None]
    return wmat


def build_analytic(omega, a2f, coulomb, kT: float, *, band_edge,
                   n_xi: int = 300) -> AnalyticKernels:
    """
    Assemble the analytic gap operator.

    Args:
      omega, a2f : phonon grid and alpha^2F, (n_omega,) or (nb, nb, n_omega),
                   `eliashberg.linearized` conventions (partial-DOS weights
                   folded in). A single frequency is an Einstein mode whose
                   a2f is the integrated weight lambda*omega/2.
      coulomb    : the STATICALLY SCREENED Coulomb, dimensionless N(0)W.
                   A scalar or (nb, nb) matrix acts flat over the band(s); a
                   callable ``coulomb(i, j, xi, xi_p)`` (broadcasting arrays)
                   gives full energy structure. It must be even in xi and
                   xi_p (s-wave).
      band_edge  : Wb in Hartree, scalar or per-band (nb,).
    """
    beta = 1.0 / kT
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    a = as_band_matrix(a2f, len(omega))
    nb = a.shape[0]
    wb = np.asarray(band_edge, dtype=np.float64)
    if wb.ndim == 0:
        wb = np.full(nb, float(wb))
    elif wb.shape != (nb,):
        raise ValueError(f"band_edge must be a scalar or ({nb},) array; "
                         f"got shape {wb.shape}")

    # per-band grids, dense near E_F
    x = np.empty((n_xi, nb))
    gw = np.empty((n_xi, nb))
    for j in range(nb):
        xj = np.geomspace(0.02 * kT, wb[j], n_xi)
        gj = np.zeros_like(xj)
        gj[1:-1] = 0.5 * (xj[2:] - xj[:-2])
        gj[0] = 0.5 * (xj[1] - xj[0]) + xj[0]
        gj[-1] = 0.5 * (xj[-1] - xj[-2])
        x[:, j], gw[:, j] = xj, 2.0 * gj

    lam_row = np.trapezoid(2.0 * a / omega[None, None, :], omega,
                           axis=-1).sum(axis=1) if len(omega) > 1 else \
        (2.0 * a[:, :, 0] / omega[0]).sum(axis=1)
    zbar = 1.0 + lam_row                                     # (nb,)
    w_s = _omega_log_rows(omega, a)                          # (nb,)

    g = _gfun(x, beta)                                       # (n_xi, nb)
    h_z = np.empty((n_xi, nb))
    sigma_s = np.empty((n_xi, nb))
    for j in range(nb):
        h_z[:, j] = _h_z_true(x[:, j], beta, omega, a,
                              _HZ_REACH * float(wb.max()), j)
        y = x[:, j] / zbar[j]
        d = y ** 2 - w_s[j] ** 2
        d = np.where(np.abs(d) < 1e-12 * w_s[j] ** 2,
                     1e-12 * w_s[j] ** 2, d)
        sigma_s[:, j] = (w_s[j] ** 2 / zbar[j] ** 2) * (
            _gfun(w_s[j], beta) - _gfun(y, beta)) / d

    wmat = _coulomb_matrix(coulomb, x, nb)

    # block assembly: Delta stacked as (band, xi)
    n = nb * n_xi
    C = np.zeros((n, n))
    Y1W = np.zeros((n, n))                                   # Y_1 with weights
    YsW = np.zeros((n, n))                                   # Y_s with weights
    for i in range(nb):
        xi_i = x[:, i] / zbar[i]
        sl_i = slice(i * n_xi, (i + 1) * n_xi)
        for j in range(nb):
            sl_j = slice(j * n_xi, (j + 1) * n_xi)
            yj = x[:, j] / zbar[j]
            pref = 1.0 / (zbar[i] ** 2 * zbar[j] ** 2)
            s_xy = _s_spectrum(xi_i, yj, omega, a[i, j], beta)
            s_xw = _s_spectrum(xi_i, np.array([w_s[j]]), omega,
                               a[i, j], beta)[:, 0]
            d = yj ** 2 - w_s[j] ** 2
            d = np.where(np.abs(d) < 1e-12 * w_s[j] ** 2,
                         1e-12 * w_s[j] ** 2, d)
            y1 = pref * s_xy
            ys = pref * (w_s[j] ** 2 / d)[None, :] * (s_xw[:, None] - s_xy)
            Y1W[sl_i, sl_j] = y1 * gw[:, j][None, :]
            YsW[sl_i, sl_j] = ys * gw[:, j][None, :]
            C[sl_i, sl_j] = -(wmat[i, :, j, :]
                              * (g[:, j] * gw[:, j])[None, :])

    g_flat = g.T.ravel()
    hz_flat = h_z.T.ravel()
    ss_flat = sigma_s.T.ravel()
    Da = np.diag(g_flat / ss_flat) - (hz_flat / ss_flat)[:, None] * C
    M = YsW @ Da + (Y1W + np.diag(hz_flat)) @ C
    M /= g_flat[:, None]
    return AnalyticKernels(xi=x, weights=gw, g=g, h_z=h_z,
                           sigma_s=sigma_s, operator=M)


def linearized_eigenvalue_analytic(omega, a2f, coulomb, kT: float,
                                   **kw) -> float:
    """Leading REAL eigenvalue (largest real part, not largest modulus: a
    repulsive Coulomb gives the operator a large negative eigenvalue)."""
    k = build_analytic(omega, a2f, coulomb, kT, **kw)
    return float(np.linalg.eigvals(k.operator).real.max())


def tc_analytic(omega, a2f, coulomb, *, band_edge, t_min: float = 0.1,
                t_max: float = 600.0, tol: float = 1e-3, **kw) -> float:
    """
    Tc in Kelvin by bisecting the leading eigenvalue through 1.

    ``t_min`` may be far lower than for `tc_unexpanded`: no Matsubara kernel
    matrix is formed, so cooling only lengthens the 1-D h_Z sum.
    """
    def rho(t):
        return linearized_eigenvalue_analytic(omega, a2f, coulomb,
                                              t * K_B_HARTREE,
                                              band_edge=band_edge, **kw)
    if rho(t_min) < 1.0:
        return 0.0
    if rho(t_max) > 1.0:
        raise RuntimeError(f"rho > 1 at t_max = {t_max} K; raise t_max")
    lo, hi = t_min, t_max
    while (hi - lo) > tol * max(hi, 1.0):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rho(mid) > 1.0 else (lo, mid)
    return 0.5 * (lo + hi)
