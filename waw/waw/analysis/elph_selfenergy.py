"""
Electron self-energy from the electron-phonon interaction: how the phonons
renormalise the BANDS, rather than how they scatter the carriers.

This is the complement of `analysis.elph_boltzmann`. Both contract the same
validated vertex g(R_e, R_q); transport wants it on the Fermi surface with a
velocity weight, whereas here it becomes a frequency-dependent self-energy
that reshapes the dispersion itself.

The Fan-Migdal term (Giustino, Rev. Mod. Phys. 89, 015003 (2017), Eq. 128):

  Sigma_nk(w) = (1/Nq) sum_(q nu m) |g^nu_mn(k,q)|^2
                x [ (n_qnu + f_(m,k+q)) / (w - eps_(m,k+q) + w_qnu + i eta)
                  + (n_qnu + 1 - f_(m,k+q)) / (w - eps_(m,k+q) - w_qnu + i eta) ]

with n the Bose and f the Fermi factor. Im Sigma is the phonon-limited
linewidth; Re Sigma carries the mass enhancement and the ARPES kink.

WHAT IS AND IS NOT INCLUDED
---------------------------
Only Fan-Migdal. The Debye-Waller term is deliberately omitted, and the
omission is not a limitation for what this module is used for: DW is STATIC
(frequency-independent), so it contributes nothing to

  * Z = [1 - dRe Sigma/dw]^-1, hence nothing to the mass enhancement,
  * the dispersion kink,
  * Im Sigma, hence nothing to the linewidth.

It shifts bands rigidly, and so matters for absolute band positions and for
Allen-Heine-Cardona gap renormalisation -- neither of which this module
claims. Adding it properly needs the second-order (two-phonon) matrix
element, which DFPT does not supply directly; the rigid-ion reconstruction
from first-order couplings is a further approximation and is not attempted
here rather than guessed at.

No vertex corrections either, which for phonons is Migdal's theorem
(corrections of order w_D/E_F) rather than an admission. That protection does
NOT extend to magnons.

SPIN
----
Sigma is per spin channel and carries NO factor of two: the electron
propagates in one channel. This differs from `alpha2f`, where the factor
appears in both the coupling sum and N(eF) and cancels -- here there is no
N(eF) to cancel against.

Atomic units throughout (Hartree, Bohr); convert at the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR
from ..core.distributions import bose_einstein, fermi_dirac
from ..units import AMU_TO_ME, K_B_HARTREE
from .elph import (EPS_ACOUSTIC, _fixed_q_h_provider, band_eigensystem)


@dataclass
class ElphSelfEnergy:
    """Band-diagonal Fan-Migdal self-energy (atomic units)."""
    kpts:        np.ndarray   # (nk, 3) fractional
    energies:    np.ndarray   # (nE,) Hartree, the frequency grid
    sigma:       np.ndarray   # (nE, nk, nb) complex, Sigma_nk(w)
    eig:         np.ndarray   # (nk, nb) Hartree, bare bands
    temperature: float        # Kelvin
    eta:         float        # Hartree, the i*eta in the denominators


def fan_migdal_self_energy(
    kpts_target: np.ndarray,
    hr: HamiltonianR,
    g_R: np.ndarray, R_e: np.ndarray, degen_e: np.ndarray,
    R_q: np.ndarray, degen_q: np.ndarray,
    qpts: np.ndarray,
    omega_ph: np.ndarray, eigvec_ph: np.ndarray,
    masses_amu: np.ndarray, types: np.ndarray,
    energies: np.ndarray,
    *,
    fermi_energy: float,
    temperature: float = 0.0,
    eta: float = 1e-3,
    eps_acoustic: float = EPS_ACOUSTIC,
    q_weights: np.ndarray | None = None,
    bands: np.ndarray | None = None,
) -> ElphSelfEnergy:
    """
    Band-diagonal Sigma_nk(w) at the requested k-points.

    `kpts_target` is arbitrary -- a band path, a Fermi-surface sample, or a
    mesh -- because H(k+q) is interpolated rather than looked up. The q sum
    runs over `qpts` with `q_weights` (star multiplicities from
    `interfaces.ase.structure.irreducible_qpoints`, or uniform if omitted).

    Parameters
    ----------
    g_R, R_e, degen_e, R_q, degen_q : from `elph.wannier_transform_elph`.
    omega_ph, eigvec_ph : (nq, nmodes) Hartree and (nq, nmodes, nmodes), at
        `qpts`.
    energies : (nE,) Hartree, ABSOLUTE (same zero as `eig` and
        `fermi_energy`), ascending.
    temperature : Kelvin. T enters through both the Bose and Fermi factors.
    eta : Hartree. A numerical broadening, not a physical one -- Im Sigma is
        the physics. Keep it well below the phonon scale or it will smear the
        very structure being computed.
    bands : optional (nb_sel,) indices to restrict the OUTER band index n
        (the state whose self-energy is wanted). The inner sum over m always
        runs over every Wannier band.

    Returns
    -------
    ElphSelfEnergy with `sigma` of shape (nE, nk, nb_sel).
    """
    kpts_target = np.asarray(kpts_target, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    qpts = np.asarray(qpts, dtype=np.float64)
    if np.any(np.diff(energies) <= 0):
        raise ValueError("fan_migdal_self_energy: `energies` must be ascending.")
    nq, n_modes = omega_ph.shape
    if q_weights is None:
        q_weights = np.full(nq, 1.0 / nq)
    else:
        q_weights = np.asarray(q_weights, dtype=np.float64)
        if q_weights.shape != (nq,):
            raise ValueError(
                f"fan_migdal_self_energy: q_weights has shape "
                f"{q_weights.shape}, expected ({nq},)."
            )
        if not np.isclose(q_weights.sum(), 1.0, rtol=0, atol=1e-8):
            raise ValueError(
                f"fan_migdal_self_energy: q_weights sum to "
                f"{q_weights.sum():.10g}, not 1 -- they must be normalised "
                "star multiplicities."
            )

    kT = float(temperature) * K_B_HARTREE
    eig_k, _ = band_eigensystem(hr, kpts_target)
    nk, nw = eig_k.shape
    sel = np.arange(nw) if bands is None else np.asarray(bands, dtype=int)
    nE = len(energies)

    mass_mu_me = np.repeat(masses_amu[types], 3) * AMU_TO_ME
    _h_of = _fixed_q_h_provider(g_R, R_q, degen_q, qpts)
    _inv_deg_e = 1.0 / np.asarray(degen_e, dtype=np.float64)
    _R_e_T = np.asarray(R_e, dtype=np.float64).T

    sigma = np.zeros((nE, nk, len(sel)), dtype=np.complex128)
    _, U_k = band_eigensystem(hr, kpts_target)

    for iq in range(nq):
        eig_kq, U_kq = band_eigensystem(hr, kpts_target + qpts[iq])

        ph_k = np.exp(2j * np.pi * (kpts_target @ _R_e_T)) * _inv_deg_e[None, :]
        g_kq = np.tensordot(ph_k, _h_of(iq), axes=([1], [0]))

        alive = omega_ph[iq] > eps_acoustic
        pref = eigvec_ph[iq] / np.sqrt(mass_mu_me)[:, None]
        pref = pref / np.sqrt(2.0 * np.where(alive, omega_ph[iq], 1.0))[None, :]
        pref = pref * alive[None, :]
        g_mode = np.einsum("uv,kunm->kvnm", pref, g_kq, optimize=True)
        # (nk, nmodes, nw[k+q band m], nw[k band n]) after the gauge rotation
        g_band = np.einsum("kjm,kvjl,kln->kvmn", U_kq.conj(), g_mode, U_k,
                           optimize=True)
        g2 = (np.abs(g_band[:, :, :, sel]) ** 2).real   # (nk, nmodes, nw, nb)

        w_ph = omega_ph[iq]                             # (nmodes,)
        nB = bose_einstein(w_ph, kT)                    # (nmodes,)
        f_kq = fermi_dirac(eig_kq, fermi_energy, kT)    # (nk, nw)

        # numerators are (nk, nmodes, nw): emission and absorption branches
        num_p = nB[None, :, None] + f_kq[:, None, :]
        num_m = nB[None, :, None] + 1.0 - f_kq[:, None, :]
        # denominators (nE, nk, nmodes, nw)
        d = energies[:, None, None, None] - eig_kq[None, :, None, :] \
            + 1j * eta
        dp = d + w_ph[None, None, :, None]
        dm = d - w_ph[None, None, :, None]
        kern = num_p[None] / dp + num_m[None] / dm      # (nE, nk, nmodes, nw)
        kern = kern * alive[None, None, :, None]

        sigma += q_weights[iq] * np.einsum(
            "ekvm,kvmb->ekb", kern, g2, optimize=True)

    return ElphSelfEnergy(kpts=kpts_target, energies=energies, sigma=sigma,
                          eig=eig_k[:, sel], temperature=float(temperature),
                          eta=float(eta))


def mass_enhancement(se: ElphSelfEnergy, fermi_energy: float) -> np.ndarray:
    """
    lambda_nk = -dRe Sigma_nk(w)/dw at w = eF, so that Z = 1/(1 + lambda).

    Evaluated at the FERMI level rather than on shell: the mass enhancement
    that renormalises the Fermi velocity, and the quantity that should equal
    the lambda from `elph.alpha2f` when averaged over the Fermi surface --
    a real cross-check, since the two are different contractions of the same
    vertex (a frequency derivative here, a double Fermi-surface delta there).

    ZERO TEMPERATURE ONLY. This is not a convergence caveat, it is a
    statement about which axis the quantity lives on. The Fan-Migdal term

        (n_qnu + f_(m,k+q)) / (w - eps_(m,k+q) + w_qnu + i eta)

    has a pole at eps_(m,k+q) = w + w_qnu -- a state one phonon ABOVE w. At
    T = 0 the numerator vanishes there (n = 0 and the state is empty), so the
    pole is dead and dRe Sigma/dw is finite and eta-independent: measured on
    Al, lambda = 0.435 at every eta from 0.5 down to 0.05 meV. At T > 0 the
    Bose factor switches that numerator on, the pole becomes LIVE on the real
    axis, and the derivative acquires a 1/(w - eps)^2 divergence. It does not
    converge as eta -> 0, it blows up: on the same Al model at 300 K,
    lambda = -0.48, -3.1, -45.4 at eta = 0.5, 0.25, 0.10 meV.

    The finite-temperature mass enhancement is a MATSUBARA quantity,
    Z(i w_0) = 1 - Im Sigma(i w_0)/w_0 with w_0 = pi k_B T, which is finite at
    every T -- and is why Eliashberg theory is written on that axis. This
    function therefore refuses a finite-temperature self-energy rather than
    returning a number that looks plausible at one eta.

    Im Sigma is unaffected by any of this and is meaningful at all T.

    Returns (nk, nb).
    """
    if se.temperature > 0.0:
        raise ValueError(
            f"mass_enhancement: the self-energy was built at T = "
            f"{se.temperature:g} K, and -dRe Sigma/dw is a zero-temperature "
            "quantity -- at finite T the Bose factor activates real-axis "
            "poles and the derivative diverges as eta -> 0 rather than "
            "converging (see this function's docstring). Rebuild the "
            "self-energy at temperature=0 for the mass enhancement, or take "
            "Z = 1 - Im Sigma(i w_0)/w_0 on the Matsubara axis for finite T. "
            "Im Sigma from this object is valid at any temperature."
        )
    re = se.sigma.real
    d = np.gradient(re, se.energies, axis=0)            # (nE, nk, nb)
    i = int(np.argmin(np.abs(se.energies - fermi_energy)))
    if 0 < i < len(se.energies) - 1:
        x = se.energies[i - 1:i + 2]
        y = d[i - 1:i + 2]
        w = np.array([np.interp(fermi_energy, x, y[:, a, b])
                      for a in range(y.shape[1]) for b in range(y.shape[2])])
        return -w.reshape(y.shape[1], y.shape[2])
    return -d[i]


def fermi_surface_average_lambda(
    se: ElphSelfEnergy, fermi_energy: float, sigma_e: float,
) -> float:
    """
    N(eF)-weighted Fermi-surface average of `mass_enhancement`, i.e. the
    isotropic lambda that `elph.alpha2f` + `elph.lambda_from_a2f` returns.

    Weighting each state by its own delta(eps_nk - eF) is what makes the two
    comparable: alpha2F's numerator carries exactly that factor.

    DO NOT SUBSAMPLE THE K-POINTS. lambda_nk = sum_m |g|^2/(|eps_m - eF| +
    w_qnu)^2 has no upper bound -- it grows without limit as an intermediate
    state approaches eF with a soft mode -- so this average is heavy-tailed
    and a sparse sample is dominated by whichever outliers happen to land
    near the Fermi level. Measured on Nb at 32^3: taking every k inside the
    window (1461 states) the average is 1.211 and moves 0.7% under a
    +-0.5 meV shift in eF, matching alpha2F's 1.208 to 0.3%; thinning the
    SAME window to 900 or 300 states makes it swing by 10% over the same
    shift and report anywhere between 1.15 and 1.30. Pass every k in the
    window and let the Gaussian weight do the selecting.

    This function warns when the ten largest contributions carry more than
    five times their uniform share (10/N), which is the signature of that
    regime. The test is skipped below 100 states, where "the top ten" is a
    sizeable fraction of the sample by construction and says nothing.
    """
    import warnings

    from ..core.distributions import gaussian_smearing

    lam = mass_enhancement(se, fermi_energy)            # (nk, nb)
    w = gaussian_smearing(se.eig - fermi_energy, sigma_e)
    tot = w.sum()
    if not tot > 0.0:
        raise ValueError(
            "fermi_surface_average_lambda: no state within sigma_e of the "
            "Fermi level at these k-points -- the average is 0/0. Sample "
            "k-points on the Fermi surface, or widen sigma_e."
        )
    contrib = (lam * w).ravel()
    n_states = contrib.size
    if n_states >= 100:
        share = np.sort(contrib)[::-1][:10].sum() / contrib.sum()
        excess = share / (10.0 / n_states)
        if excess > 5.0:
            warnings.warn(
                f"fermi_surface_average_lambda: the ten largest contributions "
                f"carry {100*share:.0f}% of the total, {excess:.0f}x their "
                f"uniform share, so this average is dominated by a handful of "
                f"{n_states} states and is not converged. lambda_nk is "
                f"heavy-tailed; use every k inside the Fermi window rather "
                f"than a subsample.",
                RuntimeWarning, stacklevel=2,
            )
    return float(contrib.sum() / tot)
