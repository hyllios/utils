"""
An SCDFT functional built from the Sham-Schlueter condition WITHOUT expanding it.

WHY THIS EXISTS. LM2005 and SPG both start from the Nambu Sham-Schlueter
equation and then replace the interacting propagators by Kohn-Sham ones. SPG
themselves note that this "is not justified by Migdal's theorem, resulting in
an uncontrolled approximation", and Luders et al. that the resulting Z(0) is
"approximately 2 lambda ... twice the value expected". The two published
functionals repair that differently -- LM2005 by patching Z (its Eqs. 79-81),
SPG by fitting three constants into the anomalous self-energy -- but both
replace the propagators first and correct afterwards.

The construction here does not expand. Linearised in the gap, the SS condition
chi(xi) = chi_s(xi) is, with Migdal-Eliashberg propagators,

    Delta_s(xi) = [ sum_n phi_n / (w_n^2 Z_n^2 + xi^2) ]
                / [ sum_n 1     / (w_n^2         + xi^2) ]                  (*)

which keeps both renormalisation effects that (*) contains and that the
expansion separates: the Z^2 from the two dressed propagators, and the fact
that the anomalous self-energy is phi_n = Z_n Delta_n, not Delta_n. Their
product leaves one power of 1/Z instead of two, which is the difference between
Z(0) = lambda and Z(0) = 2 lambda.

CLOSING THE EQUATION. (*) fixes only the FREQUENCY-SUMMED anomalous density, so
the one thing still needed is the frequency profile of phi. The ansatz is a
small per-band PROFILE BASIS whose coefficients the SS condition itself
determines by least squares (see the projection comment in `build_kernels`).
Four basis choices are provided, none fitted:

  profile="galerkin", n_basis=P : THE DEFAULT, with P = 4. The Krylov basis
      {seed, M seed, ..., M^(P-1) seed} of the Eliashberg kernel M, applied to
      the closed-form seed below. This is a Galerkin (Arnoldi-like) projection
      of M in the metric the SS condition supplies, so it is convergent in P
      and branch-safe (the large negative Coulomb eigenvalue of M becomes a
      separated Ritz value instead of a runaway direction). It costs P
      matrix-vector products per band pair -- no eigensolve, nothing iterated
      to self-consistency, nothing fitted. P is a truncation order like a
      basis-set size, not a tuning parameter.
  profile="lorentzian" : the closed-form seed alone (galerkin with P = 1),

          s_m = 1/(1 + (w_m/w_ln)^2),      phihat_m = s_m - r,
          r   = mu S/(1 + mu Q),   S = T sum_m A_m s_m,   Q = T sum_m A_m,
          A_m = (2/|w_m| Z_m) arctan(Wb/|w_m| Z_m)

      s_m is the weak-coupling-exact frequency dependence of the Eliashberg gap
      function for an Einstein spectrum -- SPG say so themselves, and it is
      their gamma2 factor at gamma2 = 1. The amplitude r is NOT guessed: for
      |w_n| -> infinity the phonon kernel vanishes and the Eliashberg equation
      gives phi_inf = -mu T sum_m A_m phi_m exactly, which closes on itself.
      mu Q is the Morel-Anderson denominator, to which r reduces for a step s.
      Multiband, the closure is the linear system (1 + mu Q) r = mu S.
  profile="iterate", n_iter=k : the single vector M^k seed -- the hierarchy of
      the derivation notes, kept for the record. NOT convergent in k once
      mu > 0: the kernel's negative Coulomb eigenvalue can exceed the physical
      one in modulus, so large k amplifies the wrong branch (measured: k = 2
      on the two-band model at mu = 0.2 gives Tc ratio 0.73). Small k works
      because the seed is close; prefer "galerkin", which uses the same
      matvecs and does not diverge. seed="flat" reproduces the notes' bare
      hierarchy.
  profile="exact"      : the leading Eliashberg eigenvector itself. The
      P -> infinity limit and the REFERENCE case -- it makes the construction
      reproduce Migdal-Eliashberg by definition, so it validates the algebra and
      the quadrature but is not a functional (it costs an eigensolve of the
      Matsubara kernel, i.e. it solves Eliashberg).

THE AMPLITUDE IS EXTRACTED BY PROJECTION (fixed 2026-08-03). Inverting the SS
condition with a single-profile ansatz pointwise divides by
sigma(xi) = sum_m phihat_m D(xi,m), which has a ZERO CROSSING whenever phihat
changes sign -- i.e. for any mu > 0, since that sign change is the
Morel-Anderson mechanism this construction relies on. The pointwise form is
therefore inconsistent, not merely delicate, and it was measurably so: on the
Einstein model at lambda = 0.5, mu = 0.2 the leading eigenvalue drifted 20% from
n_xi = 150 to 800 with no plateau, rho(T) lost monotonicity, and the Tc
bisection returned 75.7 K for Al. It affected all three profiles identically
(the division happens after the profile is chosen) and was present in the
prototype, so it was a defect of the construction as derived. See
`tests/test_utils_scdft_unexpanded_amplitude.py` for the details and
`build_kernels` for the replacement, a projection with a positive-definite
denominator that makes the operator rank one.

MIND WHICH ELIASHBERG YOU COMPARE TO. This construction is BAND-LIMITED: every
xi' integral is cut at band_edge, which at mu = 0 truncates the phonon channel
too, so the window is a parameter and not only a convergence knob.

VALIDATION (scripts/unexpanded_{bench,multiband,spectra}.py of the derivation
notes; band-limited Migdal-Eliashberg reference with the same bare static mu
and the same Matsubara reach on both sides). Tc/Tc_El:

  Einstein, lambda = 0.5-1.5 x mu = 0-0.3 (11 combinations):
    closed form (P = 1)   : 0.956-1.020, mean 0.986
    galerkin  P = 3       : 1.0000-1.0004
    galerkin  P = 4       : 1.0000-1.0001   <- default
  two-band MgB2-like lambda matrix, mu = 0-0.2, incl. unequal band widths:
    galerkin  P = 4       : 1.0000-1.0001
    (at uniform mu = 0.2 the leading state is s+- -- the pi gap changes sign
    -- and the construction follows it)
  Pb-like two-peak and Debye-like omega^2 spectra, mu = 0-0.3:
    galerkin  P = 3       : 1.0000-1.0004,  P = 4: within 1e-4.

Two corrections to earlier claims recorded here. The closed form is NOT always
below unity (1.020 at lambda = 1.5, mu = 0): "a fixed trial profile can only
lower the eigenvalue" would be a variational theorem for a Hermitian operator,
but this operator is not Hermitian, so the direction is a tendency and not a
bound. For the same reason the Galerkin Ritz value may overshoot at
intermediate P (P = 2 reached 1.06 on the two-band model) -- convergence in P
is fast but not monotone. Real Al (lambda 0.478, closed-form profile): at
mu = 0.2 over Wb = 0.3-0.8 eV, Tc = 1.02-1.42 K against the measured 1.18 K,
and at mu = 0, 8.06-8.39 K against McMillan's 8.6 K at the same mu*. The
band-edge sensitivity is mild (4% at mu = 0 across a 2.7x change in Wb).

MULTIBAND. a2f may be a band-resolved (nb, nb, n_omega) matrix with the
partial-DOS weights folded in (the `eliashberg.linearized` convention), mu a
(nb, nb) matrix, and band_edge per band. The Sham-Schlueter condition holds
per band, the ansatz becomes phi_m(i) = sum_p a_ip phihat^(p)_m(i), and the
projection turns the gap operator into a sum of nb*P rank-one terms whose
nonzero spectrum is the (nb*P) x (nb*P) amplitude map
(`UnexpandedKernels.reduced`).

NUMBERS PREDATING THE FIX ARE VOID and have been removed rather than kept for
comparison: the earlier quoted ratios (0.928 / 1.043 / 0.966, and the "iterate"
sequence 0.830, 0.936, 0.960, 0.966) came from the pointwise form at a single
unconverged n_xi, and sat ABOVE Eliashberg because the pole inflated the
operator norm. The derivation notes' own Table of 1.00-1.08 at mu = 0.1-0.3 is
void for the same reason.

STATUS -- READ THIS. Flat DOS per band, isotropic coupling within a band, a
static structureless mu, linearised at T_c. Validated against band-limited
Migdal-Eliashberg (to 1e-4 at the default settings) rather than against
experiment, and only on models plus one real material (Al); the finite-gap
regime remains untested: Z_n below Tc acquires gap dependence this module does
not carry. The reachable temperature is bounded from below because the
Matsubara grid must span the band: T >~ 4 Wb/(2 pi N_max k_B), which is 0.62 K
at Wb = 0.5 eV -- realistic multi-eV band widths are therefore out of reach,
and Wb must be treated as an effective Coulomb window. `spg` remains the
production default until this construction has seen real multiband inputs.

Atomic units in, Kelvin out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waw.units import K_B_HARTREE

from ..eliashberg.kernels import as_band_matrix, lambda_kernel

__all__ = ["UnexpandedKernels", "build_kernels", "linearized_eigenvalue_unexpanded",
           "tail_axis", "tc_unexpanded"]


@dataclass
class UnexpandedKernels:
    """Everything the gap operator is built from, kept for inspection.

    Every band-resolved array carries a trailing band axis, also for nb = 1.
    `operator` is the full gap operator in row-major (band, xi) order;
    `reduced` is the nb x nb amplitude map with the same nonzero spectrum,
    which is what the eigenvalue is actually read from (the operator is a sum
    of nb rank-one terms, one amplitude per band, so its nonzero spectrum is
    exactly that of the small matrix)."""
    omega_n: np.ndarray      # (n_tot,) full axis: dense Matsubara + tail nodes
    q_n:     np.ndarray      # (n_tot,) Matsubara sum weights: 1 dense, dw/2pikT tail
    z_n:     np.ndarray      # (n_tot, nb) mass renormalisation, gap-free at Tc
    phihat:  np.ndarray      # (n_tot, nb, P) per-band profile basis of phi
    xi:      np.ndarray      # (n_xi, nb) per-band energy grid, Hartree
    weights: np.ndarray      # (n_xi, nb) quadrature weights
    operator: np.ndarray     # (nb*n_xi, nb*n_xi) the gap operator M
    reduced: np.ndarray      # (nb*P, nb*P) amplitude map, same nonzero spectrum


def _full_axis(kT: float, n_half: int):
    n = np.arange(-n_half, n_half)
    return (2.0 * n + 1.0) * np.pi * kT


def _abs_index(n2: int) -> np.ndarray:
    """|n - m| lookup table, int32 to halve the memory of the (2N, 2N) block."""
    i = np.arange(n2, dtype=np.int32)
    return np.abs(i[:, None] - i[None, :])


def _lambda_d(omega, a: np.ndarray, kT: float, n2: int) -> np.ndarray:
    """
    lambda_ij(d) for index differences d = 0 .. n2-1, shape (n2, nb, nb).

    A SINGLE frequency has to be special-cased exactly as `solver._omega_nodes`
    does: `lambda_kernel` integrates with the trapezoid rule, which returns 0 on
    a one-point grid, so an Einstein mode would silently give lambda = 0 and
    Tc = 0. There a2f is the already integrated weight lambda*omega/2, so

        lambda_ij(d) = 2 a_ij w / (w^2 + nu_d^2),   nu_d = 2 pi kT d.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if len(omega) == 1:
        nu = 2.0 * np.pi * kT * np.arange(n2, dtype=np.float64)
        return (2.0 * omega[0] * a[None, :, :, 0]
                / (omega[0] ** 2 + nu[:, None, None] ** 2))
    return lambda_kernel(a, omega, kT, n2 - 1)


def _lambda_matrix(omega, a2f, kT: float, omega_n: np.ndarray):
    """Single-band lambda(n-m) on the full axis, kept for tests/benchmarks."""
    n2 = len(omega_n)
    a = as_band_matrix(a2f, len(np.atleast_1d(np.asarray(omega))))
    return _lambda_d(omega, a, kT, n2)[:, 0, 0][_abs_index(n2)]


def _omega_log_rows(omega, a: np.ndarray) -> np.ndarray:
    """w_ln per band, from the row-summed spectrum sum_j a2F_ij: the shape seen
    by band i is set by everything it couples to. The Einstein mode itself for
    a single frequency."""
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if len(omega) == 1:
        return np.full(a.shape[0], float(omega[0]))
    rows = a.sum(axis=1)                                     # (nb, n_omega)
    lam = 2.0 * np.trapezoid(rows / omega[None, :], omega, axis=1)
    num = 2.0 * np.trapezoid(rows * (np.log(omega) / omega)[None, :],
                             omega, axis=1)
    return np.exp(num / lam)


#: |w_n|max / band_edge that the DENSE Matsubara grid must reach. The
#: Morel-Anderson reduction IS the high-frequency sign change of phi, so a grid
#: that stops at the band edge cannot produce it and Tc collapses to zero -- a
#: failure that looks like physics. Enforced, not merely warned about.
_MATSUBARA_REACH = 4.0
#: cap on n_half. Beyond this the full-band Matsubara construction is simply not
#: the right tool; use `spg` or `lm2005`, which work with mu and a cutoff.
_N_HALF_MAX = 6000
#: log-spaced tail quadrature nodes per side beyond the dense grid, and the
#: factor by which they extend it. WHY THEY EXIST: every Matsubara sum here
#: converges only like 1/|w|, so cutting at 4x the band edge truncates the
#: Morel-Anderson logarithm and was MEASURED to cost 5.7% of Tc at
#: lambda = 0.5, mu = 0.2 (Tc rises 1.321 -> 1.394 K from reach 4 to 48, still
#: not converged). Beyond the dense grid the physics is exactly known --
#: lambda(n-m) has decayed (reach >> phonon frequencies), Z = 1, only the bare
#: mu acts -- so the remainder of each sum is smooth in w and a handful of
#: log-spaced midpoint nodes carrying real phi unknowns integrates it:
#: sum_n f(w_n) -> integral dw f(w)/(2 pi kT). With 12 nodes per side over
#: (reach, 400*reach] the residual quadrature error is ~1e-4 of Tc.
_N_TAIL = 12
_TAIL_EXTENT = 400.0


def n_half_for(kT: float, band_edge: float) -> int:
    """Matsubara half-count that reaches _MATSUBARA_REACH * band_edge."""
    return int(np.ceil(_MATSUBARA_REACH * band_edge / (2.0 * np.pi * kT)))


def tail_axis(w_edge: float, kT: float, n_tail: int = _N_TAIL,
              extent: float = _TAIL_EXTENT):
    """
    One-sided log-spaced midpoint nodes over (w_edge, extent * w_edge] and
    their weights q such that   sum_tail-n f(w_n)  ~  sum_t q_t f(w_t)
    (i.e. q_t = dw_t / (2 pi kT), the Matsubara-sum-to-integral measure).
    """
    edges = w_edge * np.exp(np.linspace(0.0, np.log(extent), n_tail + 1))
    mids = np.sqrt(edges[1:] * edges[:-1])
    return mids, np.diff(edges) / (2.0 * np.pi * kT)


def _mu_matrix(mu, nb: int) -> np.ndarray:
    """Broadcast a scalar or (nb, nb) Coulomb parameter to a full matrix."""
    m = np.asarray(mu, dtype=np.float64)
    if m.ndim == 0:
        return np.full((nb, nb), float(m))
    if m.shape != (nb, nb):
        raise ValueError(f"mu must be a scalar or ({nb}, {nb}) matrix; "
                         f"got shape {m.shape}")
    return m


def _lorentzian_profile(w, q, a_m, kT: float, mu_m, w_ln) -> np.ndarray:
    """
    The closed form per band: phihat_i = s_i - r_i.

    The asymptotic closure generalises to a small linear system. For
    |omega_n| -> infinity the phonon kernel vanishes and the Eliashberg
    equation gives phi_inf(i) = -sum_j mu_ij T sum_m A_m(j) phi_m(j) exactly;
    inserting the ansatz with EQUAL band amplitudes,

        r_i = sum_j mu_ij (S_j - Q_j r_j)   =>   (1 + mu Q) r = mu S ,

    which reduces to r = mu S/(1 + mu Q) for one band. Equal amplitudes are an
    approximation of the SEED only: the Galerkin basis rebuilds the inter-band
    structure. q are the Matsubara sum weights (1 on the dense grid, the
    integral measure on the tail nodes).
    """
    nb = a_m.shape[1]
    s = 1.0 / (1.0 + (np.abs(w)[:, None] / w_ln[None, :]) ** 2)   # (n_tot, nb)
    big_s = kT * (q[:, None] * a_m * s).sum(axis=0)               # (nb,)
    big_q = kT * (q[:, None] * a_m).sum(axis=0)                   # (nb,)
    r = np.linalg.solve(np.eye(nb) + mu_m * big_q[None, :], mu_m @ big_s)
    return s - r[None, :]


def build_kernels(omega, a2f, mu, kT: float, *, band_edge,
                  n_half: int | None = None, n_xi: int = 400,
                  profile: str = "galerkin", n_iter: int = 1,
                  n_basis: int = 4, seed: str = "lorentzian"
                  ) -> UnexpandedKernels:
    """
    Assemble the gap operator of the unexpanded construction.

    Args:
      omega, a2f : phonon grid and alpha^2F, either a single spectrum
                   (n_omega,) or a band-resolved (nb, nb, n_omega) matrix with
                   the partial-DOS weights folded in, exactly as
                   `eliashberg.linearized` takes it. A single frequency is an
                   Einstein mode whose a2f is the integrated weight
                   lambda*omega/2, matching `scdft.solver`.
      mu         : dimensionless static Coulomb parameter, scalar or (nb, nb),
                   acting over [-band_edge, band_edge]. No mu* and no cutoff
                   parameter: the Morel-Anderson reduction comes out of the
                   full-band integral.
      band_edge  : Wb in Hartree, scalar or per-band (nb,).
      n_half     : half the number of Matsubara frequencies. ``None`` picks
                   enough to reach _MATSUBARA_REACH * max(band_edge), which is
                   REQUIRED: the high-frequency sign change of phi is the
                   Morel-Anderson mechanism, and a grid stopping at the band
                   edge silently gives Tc = 0. Cost grows as 1/T, so this
                   construction is limited to Tc that are not tiny compared
                   with the band width.
      profile    : which per-band profile basis the ansatz spans; see the
                   module docstring. "galerkin" (default) uses the n_basis
                   Krylov vectors {seed, M seed, ..., M^(n_basis-1) seed};
                   "lorentzian" is the closed-form seed alone (= galerkin with
                   n_basis = 1); "iterate" is the single vector M^n_iter seed;
                   "exact" is the Eliashberg eigenvector (reference only).
    """
    if profile not in ("lorentzian", "iterate", "exact", "galerkin"):
        raise ValueError(f"profile must be 'lorentzian', 'iterate', 'galerkin' "
                         f"or 'exact', got {profile!r}")
    if n_basis < 1:
        raise ValueError(f"n_basis must be >= 1, got {n_basis}")
    if seed not in ("lorentzian", "flat"):
        raise ValueError(f"seed must be 'lorentzian' or 'flat', got {seed!r}")
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    a = as_band_matrix(a2f, len(omega))
    nb = a.shape[0]
    mu_m = _mu_matrix(mu, nb)
    wb = np.asarray(band_edge, dtype=np.float64)
    if wb.ndim == 0:
        wb = np.full(nb, float(wb))
    elif wb.shape != (nb,):
        raise ValueError(f"band_edge must be a scalar or ({nb},) array; "
                         f"got shape {wb.shape}")
    wb_max = float(wb.max())

    if n_half is None:
        n_half = n_half_for(kT, wb_max)
        if n_half > _N_HALF_MAX:
            raise ValueError(
                f"reaching {_MATSUBARA_REACH}x the band edge at kT = {kT:.3g} Ha "
                f"needs n_half = {n_half} > {_N_HALF_MAX}. The temperature is too "
                f"low relative to band_edge for a full-band Matsubara treatment; "
                f"raise t_min, shrink band_edge, or use the 'spg'/'lm2005' "
                f"functionals, which work with mu and a cutoff instead.")
    w = _full_axis(kT, n_half)
    n2 = len(w)
    if np.max(np.abs(w)) < 3.0 * wb_max:
        raise ValueError(
            f"Matsubara cutoff {np.max(np.abs(w)):.3g} Ha does not reach beyond "
            f"the band edge {wb_max:.3g} Ha: the Morel-Anderson sign change "
            f"of phi cannot develop. Raise n_half.")

    # Extend the dense grid by log-spaced tail nodes on each side: they carry
    # real phi unknowns and complete every Matsubara sum to (effective)
    # infinity. On the tail lambda(n-m) has decayed and Z = 1, so only the
    # bare Coulomb acts there -- which is exactly the Morel-Anderson tail the
    # dense cutoff was measured to truncate (see _N_TAIL above).
    t_mid, t_q = tail_axis(float(np.max(np.abs(w))) + np.pi * kT, kT)
    w = np.concatenate([-t_mid[::-1], w, t_mid])
    q = np.concatenate([t_q[::-1], np.ones(n2), t_q])
    n_tot = len(w)
    dense = slice(_N_TAIL, _N_TAIL + n2)

    lam_d = _lambda_d(omega, a, kT, n2)                      # (n2, nb, nb)
    idx = _abs_index(n2)
    sgn = np.sign(w[dense])
    z_n = np.ones((n_tot, nb))
    for i in range(nb):
        acc = np.zeros(n2)
        for j in range(nb):
            acc += lam_d[:, i, j][idx] @ sgn
        z_n[dense, i] += (np.pi * kT / w[dense]) * acc

    s_abs = np.abs(w)[:, None] * z_n
    a_m = (2.0 / s_abs) * np.arctan(wb[None, :] / s_abs)     # band-limited xi'

    def apply_columns(phi):
        """One Eliashberg update, kept split by SOURCE band j:
        out[:, i, j] = kT sum_m q_m (lambda_ij(n-m) - mu_ij) A_m(j) phi_m(j).
        The phonon part runs over the dense block only (lambda ~ 0 on the
        tail); the Coulomb part runs over everything."""
        out = np.empty((n_tot, nb, nb))
        for i in range(nb):
            for j in range(nb):
                aqphi = a_m[:, j] * q * phi[:, j]
                col = np.full(n_tot, -mu_m[i, j] * float(aqphi.sum()))
                col[dense] += lam_d[:, i, j][idx] @ aqphi[dense]
                out[:, i, j] = kT * col
        return out

    w_ln = _omega_log_rows(omega, a)
    seed_vec = (_lorentzian_profile(w, q, a_m, kT, mu_m, w_ln)
                if seed == "lorentzian" else np.ones((n_tot, nb)))
    if profile == "lorentzian":
        basis = [seed_vec]
    elif profile == "iterate":
        v = seed_vec
        for _ in range(int(n_iter)):
            v = apply_columns(v).sum(axis=2)
        basis = [v]
    elif profile == "galerkin":
        basis = [seed_vec]
        for _ in range(int(n_basis) - 1):
            nxt = apply_columns(basis[-1]).sum(axis=2)
            basis.append(nxt / np.abs(nxt).max())
    else:  # exact
        # Largest REAL eigenvalue, not largest modulus: with a repulsive mu
        # the Eliashberg kernel carries a large negative eigenvalue whose
        # modulus can exceed the physical one, and a power iteration
        # converges to that branch instead.
        m_el = np.empty((n_tot, nb, n_tot, nb))
        for i in range(nb):
            for j in range(nb):
                lam_full = np.zeros((n_tot, n_tot))
                lam_full[dense, dense] = lam_d[:, i, j][idx]
                m_el[:, i, :, j] = (kT * (lam_full - mu_m[i, j])
                                    * (a_m[:, j] * q)[None, :])
        vals, vecs = np.linalg.eig(m_el.reshape(n_tot * nb, n_tot * nb))
        v = vecs[:, int(np.argmax(vals.real))].real.reshape(n_tot, nb)
        # Fix the sign at the LOWEST |w_n|, not by the sum: once mu > 0 the
        # Morel-Anderson tail is negative over most of the axis, so the sum
        # is negative even for the physical solution and would flip it.
        n0 = int(np.argmin(np.abs(w)))
        basis = [v if v[n0, int(np.argmax(np.abs(v[n0])))] > 0.0 else -v]

    phihat = np.stack(basis, axis=2)                         # (n_tot, nb, P)
    n_p = phihat.shape[2]
    if n_p > 1:
        # Orthonormalise each band's basis: the raw Krylov vectors become
        # nearly parallel as they converge, and the sigma-space Gram solve
        # below inherits that conditioning. Same span, same spectrum.
        for i in range(nb):
            phihat[:, i, :], _ = np.linalg.qr(phihat[:, i, :])

    # per-band energy grid, dense near E_F, out to that band's edge
    x = np.empty((n_xi, nb))
    g = np.empty((n_xi, nb))
    for j in range(nb):
        xj = np.geomspace(0.02 * kT, wb[j], n_xi)
        gj = np.zeros_like(xj)
        gj[1:-1] = 0.5 * (xj[2:] - xj[:-2])
        gj[0] = 0.5 * (xj[1] - xj[0]) + xj[0]
        gj[-1] = 0.5 * (xj[-1] - xj[-2])
        x[:, j], g[:, j] = xj, 2.0 * gj                      # xi' in (-Wb, Wb)

    # THE AMPLITUDES ARE EXTRACTED BY PROJECTION, NOT POINTWISE. With the
    # ansatz phi_m(i) = sum_p a_ip phihat^(p)_m(i) the only unknowns are the
    # nb*P scalars a_ip, and inverting the SS condition at each xi separately,
    #
    #     a_i(xi) = Delta_s(xi) P_s(xi) / sum_m phihat_m(i) D_i(xi, m) ,
    #
    # divides by a SIGNED sum. sigma_i(xi) = sum_m phihat_m(i) D_i(xi,m) has a
    # zero crossing whenever phihat changes sign -- i.e. for ANY mu > 0, since
    # that sign change IS the Morel-Anderson mechanism this construction depends
    # on. At that xi the ansatz demands an infinite phi to reproduce a finite
    # anomalous density, so the pointwise inversion is not merely unstable, it
    # is inconsistent. Measured consequences of doing it anyway (Einstein,
    # lambda = 0.5, mu = 0.2): the leading eigenvalue drifts 20% from n_xi = 150
    # to 800 with NO plateau while min|sigma| shrinks monotonically -- refining
    # the grid samples the pole harder -- and rho(T) loses monotonicity, which
    # makes the Tc bisection return confident nonsense (75.7 K for Al).
    #
    # The well-posed version is a least-squares fit over xi with a
    # positive-definite Gram matrix,
    #
    #     a_i = G_i^{-1} <sigma^(p)_i, P_s Delta_s,i>_g ,
    #     G_i,pq = <sigma^(p)_i, sigma^(q)_i>_g ,
    #
    # under which the gap operator is a sum of nb*P rank-one terms whose
    # nonzero spectrum is that of the small matrix
    #
    #     R_(ip),(jq) = [G_i^{-1}]_pp' <sigma^(p')_i, dressed_i K_ij phihat^(q)_j>_g.
    #
    # For P = 1 this is the projection of the derivation notes; for the Krylov
    # basis it is a Galerkin (Arnoldi-like) projection of the Eliashberg kernel
    # in the metric the SS condition itself supplies. Validated (single band,
    # P = 1): exactly n_xi-independent (6 digits, 150 to 1200) and rho(T)
    # monotonic. mu enters through the Eliashberg update (lambda - mu), the
    # same place the profile construction already uses it.
    phi_cols = np.empty((n_tot, nb, nb, n_p))
    for p in range(n_p):
        phi_cols[:, :, :, p] = apply_columns(phihat[:, :, p])
    u = np.empty((n_xi, nb, nb, n_p))
    sigma = np.empty((n_xi, nb, n_p))
    vfun = np.empty((n_xi, nb, n_p))
    red = np.empty((nb, n_p, nb, n_p))
    for i in range(nb):
        dressed = q[None, :] / (w[None, :] ** 2 * z_n[:, i][None, :] ** 2
                                + x[:, i][:, None] ** 2)
        ps = (q[None, :] / (w[None, :] ** 2 + x[:, i][:, None] ** 2)).sum(axis=1)
        sig = dressed @ phihat[:, i, :]                      # (n_xi, P)
        sigma[:, i, :] = sig
        gram_inv = np.linalg.inv(sig.T @ (g[:, i][:, None] * sig))
        vfun[:, i, :] = (g[:, i] * ps)[:, None] * (sig @ gram_inv)
        for j in range(nb):
            img = dressed @ phi_cols[:, i, j, :]             # SS condition
            u[:, i, j, :] = img / ps[:, None]
            red[i, :, j, :] = gram_inv @ (sig.T @ (g[:, i][:, None] * img))
    reduced = red.reshape(nb * n_p, nb * n_p)
    M = np.einsum('xijq,yjq->ixjy', u, vfun).reshape(nb * n_xi, nb * n_xi)
    return UnexpandedKernels(omega_n=w, q_n=q, z_n=z_n, phihat=phihat, xi=x,
                             weights=g, operator=M, reduced=reduced)


def linearized_eigenvalue_unexpanded(omega, a2f, mu, kT: float, **kw) -> float:
    """
    Leading REAL eigenvalue of the gap operator, read from the (nb*P) x (nb*P)
    amplitude map `reduced`, which carries the operator's nonzero spectrum.

    Largest REAL, not largest modulus: with a repulsive mu the operator
    acquires a large negative eigenvalue whose modulus can exceed the physical
    one, and a power iteration converges to that branch instead.
    """
    k = build_kernels(omega, a2f, mu, kT, **kw)
    return float(np.linalg.eigvals(k.reduced).real.max())


def tc_unexpanded(omega, a2f, mu, *, band_edge, t_min: float = 2.0,
                  t_max: float = 600.0, tol: float = 1e-3, **kw) -> float:
    """
    Tc in Kelvin, by bisecting the leading eigenvalue through 1.

    ``t_min`` defaults to 2 K rather than the 0.2 K of `tc_scdft`: the Matsubara
    grid has to span the band, so its cost grows as 1/T and very low
    temperatures are out of reach (see `n_half_for`).
    """
    def rho(t):
        return linearized_eigenvalue_unexpanded(omega, a2f, mu, t * K_B_HARTREE,
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
