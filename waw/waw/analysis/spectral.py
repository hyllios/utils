"""
Interacting Bloch spectral function from an arbitrary self-energy.

    A(k, E) = -(1/pi) Im Tr [ (E + i eta) I - H(k) - Sigma(k, E) ]^-1

Every many-body correction this package computes -- alloy disorder (CPA),
electron-phonon (Fan-Migdal), electron-magnon, GW quasiparticle shifts --
differs only in how Sigma is built. Once it exists, the Dyson inversion and
the spectral map are the same operation, so they live here rather than being
re-implemented per interaction.

Sigma may be

  * ``None``                     -- the bare band structure, A a set of
                                    Lorentzians of width ``eta``;
  * ``(nE, nw, nw)``             -- LOCAL (k-independent). This is what
                                    single-site CPA and single-site DMFT
                                    produce, and the shape `analysis.cpa`
                                    hands over;
  * ``(nE, nk, nw, nw)``         -- k-RESOLVED. Electron-phonon and
                                    electron-magnon self-energies are NOT
                                    local: Fan-Migdal carries explicit
                                    (k, q) structure and collapsing it to a
                                    local object throws away the dispersion
                                    kink that is usually the point of
                                    computing it.

A note on what is and is not a self-consistency problem. CPA's t-matrix loop
exists to perform a configurational average, and DMFT's loop to enforce a
local self-consistency condition; both must be iterated. Electron-phonon at
Migdal level is neither -- Sigma is built once from the DFT Green's function
and the DFPT phonons, and there is no loop at all. This module therefore
takes Sigma as data and asks no questions about where it came from.

Atomic units throughout (Hartree, Bohr); convert at the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR, operator_k


@dataclass
class SpectralFunction:
    """Interacting spectral function on a k-path (atomic units)."""
    kpath:    np.ndarray   # (nk, 3) crystal-coordinate k-points
    energies: np.ndarray   # (nE,) Hartree
    A:        np.ndarray   # (nk, nE) A(k, E) = -(1/pi) Im Tr G, states/Hartree
    eta:      float        # Hartree, the broadening added to E

    def band_resolved(self) -> np.ndarray:
        """Not available -- `A` is already the trace over the Wannier basis."""
        raise NotImplementedError(
            "SpectralFunction stores the trace; request `orbital_resolved=True` "
            "from bloch_spectral_function to keep the diagonal."
        )


def bloch_spectral_function(
    hr:           HamiltonianR,
    kpath:        np.ndarray,
    energies:     np.ndarray,
    self_energy:  np.ndarray | None = None,
    *,
    eta:              float = 1e-3,
    orbital_resolved: bool = False,
) -> SpectralFunction | tuple[SpectralFunction, np.ndarray]:
    """
    A(k, E) = -(1/pi) Im Tr [ (E + i eta) I - H(k) - Sigma(k, E) ]^-1.

    Parameters
    ----------
    hr : the (non-interacting) Wannier Hamiltonian.
    kpath : (nk, 3) fractional k-points.
    energies : (nE,) Hartree, the real-frequency grid.
    self_energy : ``None``, ``(nE, nw, nw)`` local, or ``(nE, nk, nw, nw)``
        k-resolved. Non-Hermitian is expected and required: its
        anti-Hermitian part is the lifetime, and a purely Hermitian Sigma
        leaves A a set of delta functions broadened only by `eta`.
    eta : Hartree. With a Sigma that already carries a finite Im part, keep
        this small -- it adds to the physical linewidth rather than
        replacing it, and a too-large `eta` will quietly dominate and make
        every interaction look the same.
    orbital_resolved : also return the (nk, nE, nw) diagonal of -Im G / pi,
        i.e. the projection onto each Wannier function.

    Returns
    -------
    SpectralFunction, or (SpectralFunction, (nk, nE, nw) array) if
    `orbital_resolved`.
    """
    energies = np.asarray(energies, dtype=np.float64)
    kpath = np.asarray(kpath, dtype=np.float64)
    nk, nE, nw = len(kpath), len(energies), hr.nw

    H_k = operator_k(hr.H_R, hr.R_vectors, hr.degen,
                     kpath).detach().cpu().numpy()          # (nk, nw, nw)

    sig = None
    if self_energy is not None:
        sig = np.asarray(self_energy, dtype=np.complex128)
        if sig.shape == (nE, nw, nw):
            sig = sig[:, None, :, :]                        # broadcast over k
        elif sig.shape != (nE, nk, nw, nw):
            raise ValueError(
                f"bloch_spectral_function: self_energy has shape {sig.shape}; "
                f"expected ({nE}, {nw}, {nw}) for a local self-energy or "
                f"({nE}, {nk}, {nw}, {nw}) for a k-resolved one, with "
                f"nE={nE} energies, nk={nk} k-points and nw={nw} Wannier "
                "functions."
            )

    I = np.eye(nw, dtype=np.complex128)
    A = np.empty((nk, nE), dtype=np.float64)
    A_orb = np.empty((nk, nE, nw), dtype=np.float64) if orbital_resolved else None
    for ie, E in enumerate(energies):
        M = (E + 1j * eta) * I[None] - H_k                  # (nk, nw, nw)
        if sig is not None:
            M = M - sig[ie]
        G = np.linalg.inv(M)
        diag = np.einsum("kii->ki", G)
        A[:, ie] = -diag.sum(axis=1).imag / np.pi
        if orbital_resolved:
            A_orb[:, ie, :] = -diag.imag / np.pi

    out = SpectralFunction(kpath=kpath, energies=energies, A=A, eta=eta)
    return (out, A_orb) if orbital_resolved else out


def quasiparticle_shift(
    eig:         np.ndarray,
    self_energy: np.ndarray,
    energies:    np.ndarray,
    *,
    linear: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    On-shell quasiparticle energies and renormalisation factors from a
    band-diagonal Sigma_nk(E).

    Solving E = eps_nk + Re Sigma_nk(E) exactly requires a root find per
    state; the linearised solution expands Sigma about eps_nk,

      E_nk = eps_nk + Z_nk Re Sigma_nk(eps_nk),
      Z_nk = [1 - dRe Sigma_nk/dE |_(eps_nk)]^-1,

    which is the standard quasiparticle approximation and is what the
    mass enhancement 1/Z = 1 + lambda refers to.

    Parameters
    ----------
    eig : (nk, nb) Hartree, the bare band energies.
    self_energy : (nE, nk, nb) Hartree, BAND-DIAGONAL Sigma_nk(E) on
        `energies`.
    energies : (nE,) Hartree, must be sorted ascending.
    linear : linearised (True) or a bisection root find on
        E - eps - Re Sigma(E) (False).

    Returns
    -------
    (E_qp, Z) both (nk, nb).
    """
    eig = np.asarray(eig, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    sig = np.asarray(self_energy)
    if sig.shape[0] != len(energies) or sig.shape[1:] != eig.shape:
        raise ValueError(
            f"quasiparticle_shift: self_energy shape {sig.shape} does not "
            f"match ({len(energies)},) + {eig.shape}."
        )
    if np.any(np.diff(energies) <= 0):
        raise ValueError("quasiparticle_shift: `energies` must be ascending.")

    re = sig.real
    nk, nb = eig.shape

    # One gradient over the whole (nE, nk, nb) block and one vectorised
    # linear interpolation, rather than an np.interp call per state: the
    # per-state loop this replaces cost ~2 nk nb Python-level calls, which on
    # a 46-band spinor model over a band path is tens of thousands.
    dre = np.gradient(re, energies, axis=0)
    j = np.clip(np.searchsorted(energies, eig) - 1, 0, len(energies) - 2)
    e0, e1 = energies[j], energies[j + 1]
    t = (eig - e0) / (e1 - e0)
    kk, bb = np.meshgrid(np.arange(nk), np.arange(nb), indexing="ij")
    s0 = re[j, kk, bb] * (1.0 - t) + re[j + 1, kk, bb] * t
    slope = dre[j, kk, bb] * (1.0 - t) + dre[j + 1, kk, bb] * t
    Z = 1.0 / (1.0 - slope)
    E_qp = eig + Z * s0
    if linear:
        return E_qp, Z

    # The root find genuinely is a search, but the sign changes of
    # f = E - eps - Re Sigma(E) can be located for every state at once.
    f = energies[:, None, None] - eig[None] - re
    sgn = np.signbit(f)
    cross = sgn[:-1] != sgn[1:]                      # (nE-1, nk, nb)
    dist = np.abs(energies[:-1, None, None] - eig[None])
    dist = np.where(cross, dist, np.inf)
    i = np.argmin(dist, axis=0)                      # (nk, nb)
    has = np.isfinite(np.min(dist, axis=0))
    x0, x1 = energies[i], energies[i + 1]
    f0 = f[i, kk, bb]
    f1 = f[i + 1, kk, bb]
    root = x0 - f0 * (x1 - x0) / (f1 - f0)
    return np.where(has, root, E_qp), Z
