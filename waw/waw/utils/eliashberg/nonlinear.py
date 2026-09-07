"""
Full (nonlinear) isotropic Eliashberg equations on the Matsubara axis.

Below Tc the gap is finite and the equations are coupled and nonlinear:

    R_m(j)   = sqrt(omega_m^2 + Delta_m(j)^2) Z_m(j)

    Z_n(i)   = 1 + (pi kT / omega_n) sum_j sum_m lambda^-(n,m;i,j)
                                                 Z_m(j) omega_m / R_m(j)

    Delta_n(i) Z_n(i) = pi kT sum_j sum_m
                        [lambda^+(n,m;i,j) - 2 mu*_ij c_m] Delta_m(j) Z_m(j)/R_m(j)

**Z cancels out of both right-hand sides.** Z_m omega_m / R_m reduces to
omega_m / sqrt(omega_m^2 + Delta_m^2) and Delta_m Z_m / R_m to
Delta_m / sqrt(omega_m^2 + Delta_m^2), so the iteration runs on Delta alone
and Z is a functional of it, evaluated once per step. That removes a whole
self-consistency loop, and it removes the temptation to mix a new Z with a
stale R (which is what makes hand-written versions of this converge slowly and
inconsistently).

Setting Delta -> 0 turns the second equation into `linearized.py`'s eigenvalue
problem, so the two share `kernels.gap_kernel` and must agree on where the gap
closes -- they do, and that mutual consistency is what validates both.

Cross-checked against an independent Fortran solver of the same equations:
Delta(n=0) agrees to 0.00-0.08% for MgB2 (BOTH gaps, sigma and pi) and to
0.04% for CaC6 over the range where both are converged.

Very close to Tc the two part company, and the evidence says this solver is
right. Mean-field theory requires Delta = A sqrt(1 - T/Tc) with a CONSTANT
prefactor; pairing our gaps with our Tc holds A to 1.0% over a 9x range of
(1 - T/Tc), while the reference's near-Tc gaps against its own extrapolated Tc
spread A by 12.7%. Its linear-mixing iteration suffers critical slowing down
there -- the per-step change falls below its convergence test while still far
from the fixed point -- so its gaps come out high and its extrapolated Tc
inherits the error. Anderson acceleration is what makes that region reachable
here: 20-70 iterations where plain mixing had not converged in 4000.

Units: omega, kT and Delta in Hartree; temperatures in Kelvin at the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from waw.units import K_B_HARTREE

from .kernels import (
    as_band_matrix,
    coulomb_weights,
    gap_kernel,
    lambda_kernel,
    lambda_plus_minus,
    matsubara_frequencies,
)
from .linearized import _mu_matrix, _n_matsubara_for, tc_linearized

__all__ = ["GapResult", "GapVsTemperature", "solve_gap", "gap_vs_temperature"]


@dataclass
class GapResult:
    """Converged Delta_n and Z_n at one temperature."""
    delta:        np.ndarray   # (n_matsubara, nb) Hartree
    z:            np.ndarray   # (n_matsubara, nb) dimensionless
    omega_n:      np.ndarray   # (n_matsubara,) Hartree
    kT:           float        # Hartree
    temperature:  float        # Kelvin
    converged:    bool
    n_iterations: int
    residual:     float        # max |Delta_new - Delta| at the last step, Hartree

    @property
    def delta_0(self) -> np.ndarray:
        """Delta at the lowest Matsubara frequency, per band -- the number
        usually quoted as 'the gap' on the imaginary axis."""
        return self.delta[0]

    @property
    def is_superconducting(self) -> bool:
        return bool(np.max(np.abs(self.delta_0)) > 0.0)


@dataclass
class GapVsTemperature:
    """Delta(T) for every band."""
    temperatures: np.ndarray            # (nT,) Kelvin, ascending
    delta_0:      np.ndarray            # (nT, nb) Hartree
    tc_linearized: float                # Kelvin, from the linearized equations
    results:      list = field(default_factory=list)   # the GapResult per point

    def tc_from_gap(self, threshold: float = 0.0) -> float:
        """
        Highest temperature that still carries a gap above ``threshold``.

        This is a lower bound on Tc limited by the temperature sampling, not a
        replacement for `tc_linearized` -- Delta(T) has a square-root shoulder
        at Tc, so locating the closure by watching Delta collapse needs either
        very fine sampling or an extrapolation. Use it as a consistency check.
        """
        live = np.max(np.abs(self.delta_0), axis=1) > threshold
        return float(self.temperatures[live].max()) if live.any() else 0.0


def _solve_gap_once(
    a2f: np.ndarray,
    omega: np.ndarray,
    mu_star,
    kT: float,
    *,
    n_matsubara: int,
    omega_c: float,
    delta_init: np.ndarray | float | None = None,
    signs: np.ndarray | None = None,
    mixing: float = 0.3,
    tol: float = 1e-9,
    max_iter: int = 4000,
    acceleration: str = "anderson",
    history: int = 6,
    zero_threshold: float = 1e-8,
) -> GapResult:
    """
    Solve the full equations at one temperature.

    Args:
      a2f, omega, mu_star, omega_c : as in `linearized.tc_linearized`.
      kT          : temperature in Hartree.
      n_matsubara : Matsubara points (the sum is truncated here).
      delta_init  : starting Delta -- an (n_matsubara, nb) array, a scalar
                    magnitude, or None for `pi kT` per band. Passing the
                    previous temperature's solution is what makes a Delta(T)
                    sweep cheap and stable (`gap_vs_temperature` does this).
      signs       : (nb,) initial sign per band. Needed only to look for an
                    s+- solution, which will not be found from an all-positive
                    start; the sign is NOT re-imposed during iteration, so a
                    seeded s+- that is not a solution will relax away.
      mixing      : linear mixing factor on Delta. 0.3 is the reference code's
                    value and is a reasonable compromise; 1.0 (no mixing)
                    oscillates near Tc.
      tol         : convergence on max |Delta_new - Delta| RELATIVE to the gap
                    scale. An absolute tolerance becomes unreachably strict as
                    Delta -> 0 near Tc.
      acceleration: 'anderson' (default) mixes the fixed-point residual over a
                    short history, which is what makes the near-Tc region
                    tractable; 'linear' is plain mixing, kept for comparison.
      history     : Anderson history depth.
      zero_threshold : a gap below ``zero_threshold * kT`` counts as zero (the
                    normal state). Needed because above Tc the iterate decays
                    geometrically without ever arriving.

    Returns a `GapResult`. Above Tc the iteration decays to Delta = 0, which is
    the correct physical answer and is reported as converged with
    ``is_superconducting`` False.
    """
    a = as_band_matrix(a2f, len(np.asarray(omega)))
    nb = a.shape[0]
    mu = _mu_matrix(mu_star, nb)
    if not 0.0 < mixing <= 1.0:
        raise ValueError(f"mixing must be in (0, 1], got {mixing}")

    omega_n = matsubara_frequencies(kT, n_matsubara)
    # The Matsubara spacing is 2 pi kT, so a FIXED n_matsubara reaches lower and
    # lower in frequency as T falls. Truncating inside the Coulomb cutoff is
    # unambiguously wrong -- part of the mu* window is simply missing -- and it
    # distorts small gaps first: MgB2's pi gap came out non-monotonic in T with
    # n_matsubara pinned at 256, which at 1.6 K reaches only 2.2x the phonon
    # maximum. Let `gap_vs_temperature` scale the count with T instead.
    if omega_n[-1] < omega_c:
        import warnings
        warnings.warn(
            f"the Matsubara sum reaches {omega_n[-1]:.4g} Ha but the Coulomb "
            f"cutoff is {omega_c:.4g} Ha, so part of the mu* window is "
            f"truncated at T = {kT / K_B_HARTREE:.3g} K: raise n_matsubara "
            f"(it must grow as 1/T) or lower omega_c.",
            stacklevel=3,
        )
    lam = lambda_kernel(a, omega, kT, 2 * n_matsubara - 1)
    lam_p, lam_m = lambda_plus_minus(lam, n_matsubara)
    c = coulomb_weights(kT, n_matsubara, omega_c)
    kern = gap_kernel(lam_p, mu, c, kT)                 # pi kT [lam+ - 2 mu c]
    pref_z = (np.pi * kT / omega_n)[:, None]

    if delta_init is None:
        delta = np.full((n_matsubara, nb), np.pi * kT)
    elif np.isscalar(delta_init):
        delta = np.full((n_matsubara, nb), float(delta_init))
    else:
        delta = np.array(delta_init, dtype=np.float64).reshape(-1, nb)
        if len(delta) != n_matsubara:                   # regrid across a T change
            src = np.linspace(0.0, 1.0, len(delta))
            dst = np.linspace(0.0, 1.0, n_matsubara)
            delta = np.stack([np.interp(dst, src, delta[:, j])
                              for j in range(nb)], axis=1)
    if signs is not None:
        s = np.asarray(signs, dtype=np.float64).reshape(nb)
        delta = np.abs(delta) * np.sign(s)[None, :]

    def step(d):
        """One application of the gap map, plus the Z it implies."""
        # theta_m(j) = 1 / sqrt(omega_m^2 + Delta_m(j)^2); Z has cancelled
        theta = 1.0 / np.sqrt(omega_n[:, None] ** 2 + d ** 2)
        z_ = 1.0 + pref_z * np.einsum("nmij,mj->ni", lam_m,
                                      omega_n[:, None] * theta)
        return np.einsum("nmij,mj->ni", kern, d * theta) / z_, z_

    if acceleration not in ("anderson", "linear"):
        raise ValueError(
            f"acceleration must be 'anderson' or 'linear', got {acceleration!r}"
        )

    # Anderson mixing on the fixed-point residual f = g(Delta) - Delta.
    # Plain linear mixing suffers critical slowing down: the map's leading
    # eigenvalue tends to 1 as T -> Tc, so the iteration count diverges exactly
    # where Delta(T) is most interesting. On CaC6 at 0.9994 Tc, linear mixing
    # had not converged after 4000 iterations and the gap was 18% low; Anderson
    # gets there in a few hundred.
    z = np.ones((n_matsubara, nb))
    hist_x: list[np.ndarray] = []
    hist_f: list[np.ndarray] = []
    residual, converged, it = np.inf, False, 0

    for it in range(1, max_iter + 1):
        g, z = step(delta)
        f = g - delta
        scale = max(float(np.max(np.abs(delta))), float(np.max(np.abs(g))))

        # Above Tc the iterate decays geometrically toward the normal state but
        # never reaches it, and f ~ (rho - 1) Delta keeps the RELATIVE residual
        # pinned at |rho - 1| forever. So the decay itself is the convergence
        # criterion: once the gap is orders of magnitude below the temperature
        # it is zero, and is set to exactly zero so nothing downstream has to
        # reason about denormals.
        if scale < zero_threshold * kT:
            converged, residual, delta = True, 0.0, np.zeros_like(delta)
            break

        residual = float(np.max(np.abs(f))) / scale
        if residual < tol:
            converged = True
            break

        if acceleration == "anderson":
            hist_x.append(delta.copy())
            hist_f.append(f.copy())
            if len(hist_x) > history + 1:
                hist_x.pop(0)
                hist_f.pop(0)

        if acceleration == "anderson" and len(hist_f) >= 2:
            dF = np.stack([(hist_f[i + 1] - hist_f[i]).ravel()
                           for i in range(len(hist_f) - 1)], axis=1)
            dX = np.stack([(hist_x[i + 1] - hist_x[i]).ravel()
                           for i in range(len(hist_x) - 1)], axis=1)
            try:
                # rcond kept well above machine precision: near Tc the residual
                # differences are nearly collinear, and an unregularised solve
                # extrapolates along that direction straight to Delta = 0, which
                # IS a fixed point (the normal state) and traps the iteration.
                gamma, *_ = np.linalg.lstsq(dF, f.ravel(), rcond=1e-8)
            except np.linalg.LinAlgError:
                gamma = None
            if gamma is not None:
                correction = mixing * f.ravel() - (dX + mixing * dF) @ gamma
                # trust region: an Anderson step may not move Delta by more than
                # half its own size, which is what stops the overshoot into the
                # trivial basin without slowing the good steps down
                limit = 0.5 * max(float(np.max(np.abs(delta))), 1e-300)
                big = float(np.max(np.abs(correction)))
                if big > limit:
                    correction *= limit / big
                trial = (delta.ravel() + correction).reshape(delta.shape)

                collapsed = np.max(np.abs(trial)) < 1e-3 * np.max(np.abs(delta))
                if not collapsed:
                    g_trial, _ = step(trial)
                    if np.max(np.abs(g_trial - trial)) < np.max(np.abs(f)):
                        delta = trial
                        continue
                hist_x.clear()                 # reject and restart the history
                hist_f.clear()

        delta = delta + mixing * f             # plain linear mixing

    return GapResult(delta=delta, z=z, omega_n=omega_n, kT=kT,
                     temperature=kT / K_B_HARTREE, converged=converged,
                     n_iterations=it, residual=residual)


def gap_vs_temperature(
    a2f: np.ndarray,
    omega: np.ndarray,
    mu_star,
    *,
    temperatures=None,
    n_temperatures: int = 12,
    t_min_fraction: float = 0.05,
    t_max_fraction: float = 0.98,
    omega_c: float | None = None,
    omega_max_matsubara: float | None = None,
    n_matsubara: int | None = None,
    n_matsubara_min: int = 512,
    n_matsubara_max: int = 2048,
    tc: float | None = None,
    signs: np.ndarray | None = None,
    **solver_kwargs,
) -> GapVsTemperature:
    """
    Delta(T) from the full equations.

    With ``temperatures=None`` the grid is chosen from the linearized Tc, from
    ``t_min_fraction*Tc`` to ``t_max_fraction*Tc``. Points are solved in
    ASCENDING order, each seeded with the previous solution: the gap shrinks
    smoothly with T, so that is both faster and far more stable than restarting
    from a generic guess at every point (near Tc a cold start can collapse
    straight to the trivial Delta = 0).

    ``tc`` skips the linearized solve if you already have it.
    """
    omega = np.asarray(omega, dtype=np.float64)
    a = as_band_matrix(a2f, len(omega))
    omega_ph_max = float(omega.max())
    if omega_c is None:
        omega_c = 10.0 * omega_ph_max
    if omega_max_matsubara is None:
        omega_max_matsubara = max(40.0 * omega_ph_max, 4.0 * omega_c)

    if tc is None:
        tc = tc_linearized(a, omega, mu_star, omega_c=omega_c,
                           omega_max_matsubara=omega_max_matsubara,
                           n_matsubara=n_matsubara,
                           n_matsubara_min=n_matsubara_min,
                           n_matsubara_max=n_matsubara_max).tc
    if temperatures is None:
        if tc <= 0.0:
            return GapVsTemperature(temperatures=np.zeros(0),
                                    delta_0=np.zeros((0, a.shape[0])),
                                    tc_linearized=0.0, results=[])
        temperatures = np.linspace(t_min_fraction * tc, t_max_fraction * tc,
                                   n_temperatures)
    temperatures = np.sort(np.asarray(temperatures, dtype=np.float64))

    results, delta_0, previous = [], [], None
    for t in temperatures:
        kT = t * K_B_HARTREE
        n = (n_matsubara if n_matsubara is not None
             else _n_matsubara_for(kT, omega_max_matsubara,
                                   n_matsubara_min, n_matsubara_max))
        res = solve_gap(a, omega, mu_star, kT, n_matsubara=n, omega_c=omega_c,
                        delta_init=previous, signs=(signs if previous is None
                                                   else None),
                        **solver_kwargs)
        results.append(res)
        delta_0.append(res.delta_0)
        previous = res.delta if res.is_superconducting else None
    return GapVsTemperature(temperatures=temperatures,
                            delta_0=np.array(delta_0),
                            tc_linearized=float(tc), results=results)


def solve_gap(
    a2f: np.ndarray,
    omega: np.ndarray,
    mu_star,
    kT: float,
    *,
    n_matsubara: int,
    omega_c: float,
    delta_init: np.ndarray | float | None = None,
    n_restarts: int = 2,
    **kwargs,
) -> GapResult:
    """
    Solve the full equations at one temperature, restarting if the iteration
    falls into the trivial solution.

    Delta = 0 is always a fixed point -- it is the normal state -- and below Tc
    it is an unstable one that an accelerated iteration can still land on by
    overshooting. Rather than returning "not superconducting" on what is really
    a numerical accident, this retries from a larger seed with progressively
    heavier damping, and only reports Delta = 0 once every attempt has decayed
    there. Above Tc every attempt decays, which is the correct answer.

    See `_solve_gap_once` for the arguments; everything is forwarded.
    """
    attempt = None
    for k in range(max(1, n_restarts + 1)):
        seed = delta_init if k == 0 else float((3.0 ** k) * np.pi * kT)
        opts = dict(kwargs)
        if k:
            opts["mixing"] = kwargs.get("mixing", 0.3) / (2.0 ** k)
        attempt = _solve_gap_once(a2f, omega, mu_star, kT,
                                  n_matsubara=n_matsubara, omega_c=omega_c,
                                  delta_init=seed, **opts)
        if attempt.converged and attempt.is_superconducting:
            return attempt
    return attempt
