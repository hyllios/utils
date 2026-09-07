"""
Matsubara-axis kernels for the isotropic, band-resolved Eliashberg equations.

Everything here is in ATOMIC UNITS (omega and kT in Hartree), the convention
`waw/core` and `waw/analysis` use; `cli.py` converts at the file boundary and
`linearized.py` reports temperatures in Kelvin.

Conventions follow Carbotte (Rev. Mod. Phys. 62, 1027 (1990)):

Fermionic Matsubara frequencies

    omega_n = pi kT (2n + 1),      n = 0, 1, 2, ...

The electron-phonon kernel is the frequency-difference integral

    lambda_ij(n - m) = 2 integral domega  omega a2F_ij(omega)
                                        / (omega^2 + [2 pi kT (n-m)]^2)

which is even in (n - m) and reduces at n = m to the usual
lambda_ij = 2 integral domega a2F_ij(omega)/omega.

Because Z and Delta obey Z(-n-1) = Z(n) and Delta(-n-1) = Delta(n), the sums
over all Matsubara frequencies fold onto n >= 0 with the combinations

    lambda^+(n, m) = lambda(n - m) + lambda(n + m + 1)     (gap channel)
    lambda^-(n, m) = lambda(n - m) - lambda(n + m + 1)     (mass channel)

Band indices: ``a2f[i, j]`` couples band i (the one being solved for) to band j
(the one summed over), so a2F_ij carries the density of states of band j. That
is the reference code's convention, and it is why the isotropic average is
DOS-weighted on the ROW index (see `isotropic_average`).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "matsubara_frequencies",
    "lambda_kernel",
    "lambda_plus_minus",
    "mass_renormalization_normal_state",
    "coulomb_weights",
    "gap_kernel",
    "isotropic_average",
    "rescale_mu_star",
    "as_band_matrix",
]


def matsubara_frequencies(kT: float, n_matsubara: int) -> np.ndarray:
    """omega_n = pi kT (2n+1) for n = 0 .. n_matsubara-1, Hartree."""
    if kT <= 0.0:
        raise ValueError(f"kT must be positive, got {kT}")
    if n_matsubara < 1:
        raise ValueError(f"n_matsubara must be >= 1, got {n_matsubara}")
    return np.pi * kT * (2.0 * np.arange(n_matsubara) + 1.0)


def as_band_matrix(a2f: np.ndarray, n_omega: int | None = None) -> np.ndarray:
    """
    Normalise an alpha^2F input to shape ``(nb, nb, n_omega)``.

    Accepts a single-band spectrum ``(n_omega,)`` -> ``(1, 1, n_omega)`` or an
    already-resolved ``(nb, nb, n_omega)``. Raises on anything else rather than
    guessing, since a silently transposed band matrix is not detectable
    downstream.
    """
    a = np.asarray(a2f, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, None, :]
    elif a.ndim != 3 or a.shape[0] != a.shape[1]:
        raise ValueError(
            f"a2f must have shape (n_omega,) or (nb, nb, n_omega); got {a.shape}"
        )
    if n_omega is not None and a.shape[-1] != n_omega:
        raise ValueError(
            f"a2f has {a.shape[-1]} frequency points but omega has {n_omega}"
        )
    return a


def lambda_kernel(
    a2f: np.ndarray,
    omega: np.ndarray,
    kT: float,
    n_max: int,
) -> np.ndarray:
    """
    lambda_ij(d) for index differences d = 0 .. n_max.

    Args:
      a2f   : (n_omega,) or (nb, nb, n_omega), dimensionless.
      omega : (n_omega,) phonon frequency grid, Hartree, strictly positive.
      kT    : temperature in Hartree.
      n_max : largest index difference needed. The folded kernels need
              lambda up to d = 2*n_matsubara - 1.

    Returns:
      (n_max+1, nb, nb) array. ``[0]`` is the static lambda_ij.
    """
    omega = np.asarray(omega, dtype=np.float64)
    if np.any(omega <= 0.0):
        raise ValueError(
            "omega must be strictly positive: alpha^2F/omega diverges at omega=0. "
            "Drop the zero point (or any acoustic frequencies below the "
            "eps_acoustic cutoff) before calling."
        )
    a = as_band_matrix(a2f, len(omega))

    d = np.arange(n_max + 1, dtype=np.float64)
    # bosonic difference frequency: omega_n - omega_m = 2 pi kT (n - m)
    nu = 2.0 * np.pi * kT * d                                    # (n_max+1,)
    denom = omega[None, :] ** 2 + nu[:, None] ** 2               # (n_max+1, n_omega)
    weight = 2.0 * omega[None, :] / denom                        # (n_max+1, n_omega)
    # integrate over omega for every (d, i, j)
    return np.trapezoid(weight[:, None, None, :] * a[None, :, :, :], omega, axis=-1)


def lambda_plus_minus(lam: np.ndarray, n_matsubara: int):
    """
    Fold `lambda_kernel` output into the half-axis kernels.

        lambda^+(n, m) = lambda(|n-m|) + lambda(n+m+1)
        lambda^-(n, m) = lambda(|n-m|) - lambda(n+m+1)

    Returns two ``(n_matsubara, n_matsubara, nb, nb)`` arrays.
    """
    n = n_matsubara
    if lam.shape[0] < 2 * n:
        raise ValueError(
            f"lambda_kernel was built up to d={lam.shape[0] - 1} but folding "
            f"{n} Matsubara points needs d up to {2 * n - 1}"
        )
    idx = np.arange(n)
    diff = np.abs(idx[:, None] - idx[None, :])        # |n - m|
    summ = idx[:, None] + idx[None, :] + 1           # n + m + 1
    lam_diff = lam[diff]                              # (n, n, nb, nb)
    lam_summ = lam[summ]
    return lam_diff + lam_summ, lam_diff - lam_summ


def mass_renormalization_normal_state(
    lam_minus: np.ndarray,
    omega_n: np.ndarray,
    kT: float,
) -> np.ndarray:
    """
    Z_n(i) in the NORMAL state, which is what the linearized (Delta -> 0)
    equations require:

        Z_n(i) = 1 + (pi kT / omega_n) sum_j sum_m lambda^-(n, m; i, j)

    In the full equations the factor Z_m omega_m / R_m accompanies
    lambda^-; as Delta -> 0, R_m -> |omega_m| Z_m and that factor becomes 1,
    so Z decouples from the gap and can be evaluated once per temperature.

    Returns (n_matsubara, nb).
    """
    # sum over m (axis 1) and over the summed band j (axis 3)
    s = lam_minus.sum(axis=(1, 3))                    # (n_matsubara, nb)
    return 1.0 + (np.pi * kT / omega_n)[:, None] * s


def coulomb_weights(kT: float, n_matsubara: int, omega_c: float) -> np.ndarray:
    """
    Matsubara weights implementing a sharp Coulomb cutoff at ``omega_c``.

    mu* is defined with respect to a cutoff, and the Matsubara sum can only
    stop at a grid point, so the cutoff would otherwise jump discontinuously
    between omega_n as T varies -- visible as a sawtooth in Tc(omega_c). The
    fix kept here: include every omega_n < omega_c with weight 1 and the next
    one with the fractional weight that places the effective cutoff exactly at
    omega_c.

    Returns (n_matsubara,) weights in [0, 1].
    """
    if omega_c <= 0.0:
        raise ValueError(f"omega_c must be positive, got {omega_c}")
    x = 0.5 * omega_c / (np.pi * kT) - 0.5      # omega_c = pi kT (2x + 1)
    m_c = int(np.floor(x))
    frac = x - m_c
    w = np.zeros(n_matsubara, dtype=np.float64)
    if m_c >= 0:
        w[: min(m_c + 1, n_matsubara)] = 1.0
    if 0 <= m_c + 1 < n_matsubara:
        w[m_c + 1] = frac
    return w


def gap_kernel(lam_plus: np.ndarray, mu_matrix: np.ndarray,
               coulomb_w: np.ndarray, kT: float) -> np.ndarray:
    """
    The bracket common to the linearized and full gap equations:

        pi kT [ lambda^+(n, m; i, j) - 2 mu*_ij c_m ]

    Both solvers must use the identical combination -- the factor 2 on mu*
    (from folding a frequency-independent kernel onto n >= 0) and the cutoff
    weighting are exactly the sort of thing that would silently differ between
    two implementations and make their Tc disagree, so they share this.

    Returns (n_matsubara, n_matsubara, nb, nb).
    """
    return np.pi * kT * (lam_plus
                         - 2.0 * mu_matrix[None, None, :, :]
                         * coulomb_w[None, :, None, None])


def rescale_mu_star(mu_star, omega_c_from: float, omega_c_to: float):
    """
    Move mu* from one Coulomb cutoff to another.

    mu* is not a material constant: it is the bare Coulomb repulsion already
    logarithmically renormalised down from the Fermi scale to a cutoff,
    mu*(omega_c) = mu / [1 + mu ln(E_F/omega_c)] (Morel-Anderson). Eliminating
    mu between two cutoffs gives the cutoff-independent statement

        1/mu*(omega_2) = 1/mu*(omega_1) + ln(omega_1/omega_2)

    so lowering the cutoff lowers mu*. Quoting a mu* without its cutoff, or
    reusing one across cutoffs, is a real error: on MgB2 at fixed mu* = 0.11,
    moving omega_c from 3x to 10x the phonon maximum moves Tc from 30.4 to
    34.1 K.

    Rescaling removes most but not all of that: on CaC6 it cuts the drift over
    a 6.7x range of omega_c from +8.5% to -0.5%, on MgB2 from +6.9% to +3.4%.
    The residual is the Morel-Anderson form itself being approximate, and more
    so for a multiband system carrying one scalar mu* in every channel.

    Accepts a scalar or an (nb, nb) matrix; acts elementwise.
    """
    if omega_c_from <= 0.0 or omega_c_to <= 0.0:
        raise ValueError("both cutoffs must be positive")
    mu = np.asarray(mu_star, dtype=np.float64)
    denom = 1.0 + mu * np.log(omega_c_from / omega_c_to)
    if np.any(denom <= 0.0):
        raise ValueError(
            f"rescaling mu* from {omega_c_from} to {omega_c_to} passes through "
            f"the Morel-Anderson pole (1 + mu* ln(w1/w2) <= 0); the two cutoffs "
            f"are too far apart for this form to hold"
        )
    out = mu / denom
    return float(out) if out.ndim == 0 else out


def isotropic_average(a2f: np.ndarray, dos: np.ndarray | None = None) -> np.ndarray:
    """
    Collapse a band-resolved alpha^2F matrix to the single isotropic spectrum

        a2F(omega) = sum_ij N_i a2F_ij(omega) / sum_i N_i

    ``a2F_ij`` already carries the DOS of band j (its column index), so the
    weighting is on the row index i -- exactly the reference code's
    ``a2fmcm``. Equal weights are used when ``dos`` is None.

    Use this to compare a multiband model against a single-band Tc estimate
    (Allen-Dynes, `waw.analysis.eliashberg`), not as a substitute for solving
    the coupled equations: averaging first destroys the interband structure
    that makes MgB2's two gaps.
    """
    a = as_band_matrix(a2f)
    nb = a.shape[0]
    n = np.ones(nb) if dos is None else np.asarray(dos, dtype=np.float64)
    if n.shape != (nb,):
        raise ValueError(f"dos must have shape ({nb},); got {n.shape}")
    return np.einsum("i,ijw->w", n, a) / n.sum()
