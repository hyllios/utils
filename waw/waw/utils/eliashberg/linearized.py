"""
Linearized isotropic Eliashberg equations -> Tc.

As T -> Tc the gap vanishes, so the coupled nonlinear equations reduce to a
linear eigenvalue problem in Delta. With R_m = sqrt((omega_m^2 + Delta_m^2)) Z_m
-> |omega_m| Z_m, the factor Delta_m Z_m / R_m becomes Delta_m / |omega_m| and
Z drops out of the gap equation entirely:

    Delta_n(i) = (pi kT / Z_n(i)) sum_j sum_m
                     [ lambda^+(n,m;i,j) - 2 mu*_ij c_m ] Delta_m(j) / omega_m

with Z_n(i) the normal-state mass renormalization and c_m the Coulomb cutoff
weights. Writing the bracket-and-prefactor as a matrix M, this is

    Delta = M(T) Delta,

so Tc is the temperature at which the leading eigenvalue rho(T) of M crosses 1.
rho decreases monotonically with T, which makes the root safe to bracket.

The factor 2 on mu* comes from folding: the Coulomb kernel is
frequency-independent, so the m < 0 half of the sum contributes identically
rather than with the lambda(n+m+1) combination.

This is a strictly better way to get Tc than watching the gap collapse in the
full equations: rho(T) = 1 is a sharp, well-conditioned condition, whereas
Delta(T) -> 0 has to be extrapolated through the region where the nonlinear
iteration converges most slowly. (The reference implementation fits a model
curve to Delta(T) for exactly this reason.)

Units: kT and omega in Hartree internally; ``tc_*`` helpers return Kelvin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waw.units import K_B_HARTREE

from .kernels import (
    as_band_matrix,
    isotropic_average,
    coulomb_weights,
    gap_kernel,
    lambda_kernel,
    lambda_plus_minus,
    mass_renormalization_normal_state,
    matsubara_frequencies,
)

__all__ = ["LinearizedResult", "TcResult", "linearized_kernel",
           "leading_eigenvalue", "tc_linearized"]


@dataclass
class LinearizedResult:
    """Leading eigenvalue of the linearized gap kernel at one temperature."""
    rho:          float        # leading eigenvalue; == 1 exactly at Tc
    kT:           float        # Hartree
    temperature:  float        # Kelvin
    delta_shape:  np.ndarray   # (n_matsubara, nb) leading eigenvector, normalised
    z:            np.ndarray   # (n_matsubara, nb) normal-state Z_n
    n_matsubara:  int


@dataclass
class TcResult:
    """Critical temperature from the linearized equations."""
    tc:            float             # Kelvin
    rho_at_tc:     float             # residual check, should be ~1
    n_matsubara:   int
    omega_c:       float             # Hartree, the Coulomb cutoff actually used
    lambda_matrix: np.ndarray        # (nb, nb) static lambda_ij
    gap_symmetry:  np.ndarray | None # sign of the leading eigenvector per band
    n_evaluations: int


def _mu_matrix(mu_star, nb: int) -> np.ndarray:
    """Broadcast a scalar or (nb, nb) mu* to a full matrix."""
    m = np.asarray(mu_star, dtype=np.float64)
    if m.ndim == 0:
        return np.full((nb, nb), float(m))
    if m.shape != (nb, nb):
        raise ValueError(
            f"mu_star must be a scalar or ({nb}, {nb}) matrix; got {m.shape}"
        )
    return m


def _n_matsubara_for(kT: float, omega_max_matsubara: float,
                     n_min: int, n_max: int) -> int:
    """
    How many Matsubara points reach ``omega_max_matsubara`` at this kT.

    The count grows as 1/T, and the kernel is a dense (n*nb)^2 matrix, so this
    must be capped: probing a bracket down at 0.05 K once asked for 671k points
    (3.3 TiB). Truncating raises Tc slightly -- the neglected high-frequency
    tail of lambda(d) is pair-breaking -- so the cap is deliberately generous
    and `tc_linearized` brackets near Tc instead of at absurdly low T.
    """
    n = int(round(0.5 * omega_max_matsubara / (np.pi * kT) - 0.5))
    return int(np.clip(n, n_min, n_max))


def _allen_dynes_guess(a2f_iso: np.ndarray, omega: np.ndarray,
                       mu_star_scalar: float) -> float:
    """Allen-Dynes Tc in Kelvin, used only to bracket the root search."""
    from waw.analysis.eliashberg import allen_dynes_tc_from_a2f
    try:
        tc_ha, _, _, _ = allen_dynes_tc_from_a2f(a2f_iso, omega, mu_star_scalar)
        tc = float(tc_ha) / K_B_HARTREE
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return 0.0
    return tc if np.isfinite(tc) and tc > 0.0 else 0.0


def linearized_kernel(
    a2f: np.ndarray,
    omega: np.ndarray,
    mu_star,
    kT: float,
    *,
    n_matsubara: int,
    omega_c: float,
):
    """
    Build the linearized gap kernel M and the normal-state Z at temperature kT.

    Returns ``(M, z, omega_n)`` with M of shape
    ``(n_matsubara*nb, n_matsubara*nb)`` in row-major (n, band) order.
    """
    a = as_band_matrix(a2f, len(np.asarray(omega)))
    nb = a.shape[0]
    mu = _mu_matrix(mu_star, nb)

    omega_n = matsubara_frequencies(kT, n_matsubara)
    lam = lambda_kernel(a, omega, kT, 2 * n_matsubara - 1)
    lam_p, lam_m = lambda_plus_minus(lam, n_matsubara)
    z = mass_renormalization_normal_state(lam_m, omega_n, kT)
    c = coulomb_weights(kT, n_matsubara, omega_c)

    # M[(n,i),(m,j)] = pi kT [lambda^+(n,m;i,j) - 2 mu_ij c_m] / (Z_n(i) omega_m)
    kern = gap_kernel(lam_p, mu, c, kT)
    pref = 1.0 / (z[:, None, :, None] * omega_n[None, :, None, None])
    M = (pref * kern).transpose(0, 2, 1, 3).reshape(n_matsubara * nb,
                                                    n_matsubara * nb)
    return M, z, omega_n


def leading_eigenvalue(
    a2f: np.ndarray,
    omega: np.ndarray,
    mu_star,
    kT: float,
    *,
    n_matsubara: int,
    omega_c: float,
    method: str = "power",
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> LinearizedResult:
    """
    Leading eigenvalue rho of the linearized gap kernel at temperature kT.

    ``method='power'`` (default) runs power iteration, which is what one wants
    here: the kernel is large (n_matsubara*nb square) but only its dominant
    eigenpair matters, and the dominant eigenvector is the gap symmetry.
    ``method='dense'`` diagonalises fully and is used to verify the former --
    the kernel is NOT symmetric (the 1/(Z_n omega_m) prefactor breaks it), so
    the dominant eigenvalue is not variational and power iteration deserves a
    cross-check.
    """
    M, z, omega_n = linearized_kernel(a2f, omega, mu_star, kT,
                                      n_matsubara=n_matsubara, omega_c=omega_c)
    nb = z.shape[1]

    if method == "dense":
        vals, vecs = np.linalg.eig(M)
        k = int(np.argmax(vals.real))
        rho = float(vals.real[k])
        vec = vecs[:, k].real
    elif method == "power":
        v = np.ones(M.shape[0]) / np.sqrt(M.shape[0])
        rho = 0.0
        for _ in range(max_iter):
            w = M @ v
            nrm = np.linalg.norm(w)
            if nrm == 0.0:
                rho, v = 0.0, w
                break
            w /= nrm
            new_rho = float(v @ (M @ v))          # Rayleigh quotient
            if abs(new_rho - rho) < tol * max(1.0, abs(new_rho)):
                rho, v = new_rho, w
                break
            rho, v = new_rho, w
        vec = v
    else:
        raise ValueError(f"method must be 'power' or 'dense', got {method!r}")

    vec = vec.reshape(n_matsubara, nb)
    if vec.size and vec.flat[np.argmax(np.abs(vec))] < 0:
        vec = -vec                                # fix the global sign
    nrm = np.linalg.norm(vec)
    return LinearizedResult(
        rho=rho, kT=kT, temperature=kT / K_B_HARTREE,
        delta_shape=vec / nrm if nrm else vec, z=z, n_matsubara=n_matsubara,
    )


def tc_linearized(
    a2f: np.ndarray,
    omega: np.ndarray,
    mu_star,
    *,
    omega_c: float | None = None,
    omega_max_matsubara: float | None = None,
    n_matsubara: int | None = None,
    n_matsubara_min: int = 512,
    n_matsubara_max: int = 2048,
    t_min: float = 0.02,
    t_max: float = 1000.0,
    tol: float = 1e-4,
    method: str = "power",
) -> TcResult:
    """
    Tc (Kelvin) from the linearized Eliashberg equations.

    Args:
      a2f       : (n_omega,) or (nb, nb, n_omega) Eliashberg spectral function.
      omega     : (n_omega,) phonon frequencies in HARTREE, strictly positive.
      mu_star   : scalar or (nb, nb) Coulomb pseudopotential. Must be quoted at
                  the same ``omega_c`` used here -- mu* is cutoff-dependent and
                  the pair are meaningless apart.
      omega_c   : Coulomb cutoff, Hartree. Default 10x the maximum phonon
                  frequency (EPW's stated common choice). mu* is defined AT
                  this cutoff -- use `rescale_mu_star` to move a mu* quoted at
                  a different one.
      omega_max_matsubara : Matsubara sum extent, Hartree. Default 40x the
                  maximum phonon frequency (and never below 4x omega_c);
                  lambda(d) falls off as 1/d^2 so this converges quickly, but
                  truncating it too early RAISES Tc. Deliberately independent
                  of omega_c, unlike EPW's single wscut.
      n_matsubara : fix the number of Matsubara points instead of deriving it
                  from ``omega_max_matsubara`` (which makes the count vary with
                  T). Useful for reproducing another code exactly.
      n_matsubara_min : floor on the derived count, so high temperatures keep a
                  usable grid.
      n_matsubara_max : cap on the derived count. The kernel is a dense
                  (n*nb)^2 matrix and the count grows as 1/T, so this bounds
                  both time and memory.
      t_min, t_max : outer safety bounds for the root search, Kelvin. The
                  search itself starts from the Allen-Dynes estimate and
                  expands geometrically, so these are rarely reached.
      tol       : relative convergence on Tc.

    Returns a `TcResult`; ``tc`` is 0.0 when the material does not superconduct
    within the bracket (rho < 1 even at t_min), which is a physical answer, not
    a failure -- mu* can exceed the coupling.
    """
    omega = np.asarray(omega, dtype=np.float64)
    a = as_band_matrix(a2f, len(omega))
    omega_ph_max = float(omega.max())
    if omega_c is None:
        # 10x the maximum phonon frequency, the convention EPW's documentation
        # calls common and its own Pb example uses exactly (phonons to ~10 meV,
        # wscut = 100 meV); its MgB2 example is ~5x (80 meV, 500 meV) and EPW's
        # hard default is 1 eV. Values from 3x to 10x appear in the literature.
        # This matters: on MgB2 at fixed mu* = 0.11, Tc runs from 30.4 K at 3x
        # to 34.1 K at 10x, so mu* MUST be quoted at the cutoff it belongs to
        # (see `rescale_mu_star`).
        omega_c = 10.0 * omega_ph_max
    if omega_max_matsubara is None:
        # 40x the phonon bandwidth, with the n_matsubara_min floor, reproduces
        # the n->infinity Tc to 0.01% on both reference systems (CaC6 14.0937 vs
        # 14.0918 converged; MgB2 31.9068 vs 31.9038) in about two seconds.
        # Dropping to 20x costs 0.06%/0.15%, which is still far below the
        # uncertainty in mu* -- raise it if that ever matters.
        #
        # Kept SEPARATE from omega_c, and never smaller than it. EPW uses one
        # variable (wscut) for both, so raising its cutoff to converge the
        # phonon sum also redefines mu*. Converging the numerics here leaves
        # the meaning of mu* alone. For strict EPW parity pass
        # omega_max_matsubara=omega_c.
        omega_max_matsubara = max(40.0 * omega_ph_max, 4.0 * omega_c)

    calls = 0

    def rho_of_t(t_kelvin: float) -> float:
        nonlocal calls
        calls += 1
        kT = t_kelvin * K_B_HARTREE
        n = (n_matsubara if n_matsubara is not None
             else _n_matsubara_for(kT, omega_max_matsubara,
                                   n_matsubara_min, n_matsubara_max))
        return leading_eigenvalue(a, omega, mu_star, kT, n_matsubara=n,
                                  omega_c=omega_c, method=method).rho

    lam_matrix = lambda_kernel(a, omega, 1.0, 0)[0]     # kT irrelevant at d=0

    # Bracket around the Allen-Dynes estimate rather than over [t_min, t_max]:
    # the Matsubara count goes as 1/T, so evaluating rho at a very low T is
    # both enormous and pointless when Tc is tens of kelvin.
    mu_scalar = float(np.mean(_mu_matrix(mu_star, a.shape[0])))
    guess = _allen_dynes_guess(isotropic_average(a), omega, mu_scalar)
    seed = float(np.clip(guess if guess > 0.0 else 10.0, 10.0 * t_min, 0.1 * t_max))

    lo = hi = seed
    if rho_of_t(seed) > 1.0:                            # walk up to find rho<1
        while True:
            lo, hi = hi, hi * 2.0
            if hi > t_max:
                raise RuntimeError(
                    f"rho > 1 up to t_max = {t_max} K: Tc is above the bracket. "
                    f"Raise t_max."
                )
            if rho_of_t(hi) <= 1.0:
                break
    else:                                               # walk down to find rho>1
        while True:
            lo, hi = lo / 2.0, lo
            if lo < t_min:
                return TcResult(tc=0.0, rho_at_tc=rho_of_t(t_min),
                                n_matsubara=0, omega_c=omega_c,
                                lambda_matrix=lam_matrix, gap_symmetry=None,
                                n_evaluations=calls)
            if rho_of_t(lo) > 1.0:
                break

    while (hi - lo) > tol * max(hi, 1.0):               # bisection on rho - 1
        mid = 0.5 * (lo + hi)
        if rho_of_t(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    tc = 0.5 * (lo + hi)

    kT = tc * K_B_HARTREE
    n = (n_matsubara if n_matsubara is not None
         else _n_matsubara_for(kT, omega_max_matsubara,
                               n_matsubara_min, n_matsubara_max))
    final = leading_eigenvalue(a, omega, mu_star, kT, n_matsubara=n,
                               omega_c=omega_c, method=method)
    # sign of each band's gap at the lowest Matsubara frequency: s+- vs s++
    sym = np.sign(final.delta_shape[0]) if final.delta_shape.size else None
    return TcResult(tc=tc, rho_at_tc=final.rho, n_matsubara=n, omega_c=omega_c,
                    lambda_matrix=lam_matrix, gap_symmetry=sym,
                    n_evaluations=calls)
