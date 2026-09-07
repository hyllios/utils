"""
CRTA (constant relaxation time) ab initio Wigner Transport Equation: an
interband/"Zener tunneling" correction to the semiclassical Boltzmann
electrical conductivity.

  A. Cepellotti, B. Kozinsky, Materials Today Physics 19 (2021) 100412,
  Eq. 13.
  A. Cepellotti et al. (Phoebe), J. Phys. Mater. 5 (2022) 035003, Eq. 57.

The aiWTE keeps the off-diagonal elements of the electronic density
matrix (band-eigenstate representation), on top of the diagonal
semiclassical picture in `boltzmann.py`. Physically: carriers tunneling
between bands closer together in energy than their scattering linewidth
("Zener tunneling") contribute additional current beyond a per-band
relaxation-time picture. Decomposition:

    sigma_ij = sigma_ij^aiBTE + Delta_sigma_ij

sigma^aiBTE is `boltzmann.transport_coefficients(...).elcond`;
Delta_sigma_ij is `interband_conductivity_correction` here.

Real linewidths Gamma_bs(k) require ab initio electron-phonon scattering
(DFPT + Wannier-interpolated e-ph matrix elements), which this project
does not implement. This module uses the same CRTA already used for
sigma^aiBTE: Gamma_bs(k) = 1/relax_time for every band and k-point --
qualitatively captures the tunneling shape (bands closer than ~Gamma
couple; farther apart, they don't) but is not the ab initio linewidth.

The implemented formula matches Phoebe's `wigner_electron.cpp` source
rather than the papers' printed Eq. 13/57: the velocity product entering
Delta_sigma_ab is P_ab(b,b') = v_{a,bb'} * conj(v_{b,bb'}) (conjugated on
the second Cartesian index), not the Hermiticity-simplified
v_{a,bb'}*v_{b,bb'} the papers print; for a=b this makes P_aa=|v_{a,bb'}|^2
manifestly real and non-negative, unlike the papers' v_{bb'}^2. Closed
form implemented below:

    Delta_sigma_ab = -(2*gs/(V*Nk)) * Re[ sum_{k,b!=b'}
                        ratio(b,b') * P_ab(b,b') / xC(b,b') ]

    ratio(b,b') = [f(b)-f(b')] / [eps(b)-eps(b')]      (<=0 always)
    xC(b,b')    = Gamma_tot(b,b') + i*2*[eps(b')-eps(b)]
    Gamma_tot(b,b') = 1/tau_b + 1/tau_b'

(atomic units, gs = num_elec_per_state; e=1). The diagonal (a=b) is
provably non-negative for any Hamiltonian: P_aa>=0 real, ratio<=0
always, so every term in Delta_sigma_aa is >=0.

Not independently cross-validated against Phoebe's own numbers (no
electron-phonon linewidth pipeline here); validated via internal
invariants in `tests/test_wigner_transport.py` (reality of the sum, the
diagonal >=0 property, vanishing for well-separated bands, continuity
through the near-degenerate limit).

Atomic units throughout (e = hbar = 1): mu, kT in Hartree, relax_time in
atomic time (hbar/Hartree). `interband_conductivity_correction` returns
a (3, 3) array in the same convention as
`boltzmann.transport_coefficients`'s `elcond` -- add directly, then
convert with `waw.units.to_si_units(value, "electrical_conductivity")`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR
from ..core.distributions import fermi_dirac, dfermi_dE
from ..units import to_si_units
from .dos import _uniform_mesh
from .effective_mass import _h_and_k_derivatives_cartesian
from ._fourier_derivs import h_and_grad_cart_batch, KCHUNK


def velocity_matrix(hr: HamiltonianR, kpts_frac: np.ndarray, recip_lattice: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Full <n|dH/dk_a|m> matrix element on a k-list (Hartree*Bohr), including
    off-diagonal (interband) elements -- generalizes
    `boltzmann.band_velocities`, which keeps only the diagonal (n=m)
    Hellmann-Feynman part.

    Returns:
      eig : (nk, nw) Hartree
      vmat: (nk, nw, nw, 3) complex, Hartree*Bohr; vmat[k,n,m,a] = <n|dH/dk_a|m>
    """
    kpts_frac = np.asarray(kpts_frac, dtype=np.float64)
    nk = kpts_frac.shape[0]
    nw = hr.H_R.shape[-1]
    eig = np.empty((nk, nw), dtype=np.float64)
    vmat = np.empty((nk, nw, nw, 3), dtype=np.complex128)
    for lo in range(0, nk, KCHUNK):
        H0, grad_cart = h_and_grad_cart_batch(hr, kpts_frac[lo:lo + KCHUNK], recip_lattice)
        e, U = torch.linalg.eigh(H0)
        # <n|dH/dk_a|m> = sum_{p,q} conj(U[p,n]) grad_cart[a,p,q] U[q,m]
        V = torch.einsum('kpn,kapq,kqm->knma', U.conj(), grad_cart, U)   # (nc,nw,nw,3)
        eig[lo:lo + KCHUNK] = e.cpu().numpy()
        vmat[lo:lo + KCHUNK] = V.cpu().numpy()
    return eig, vmat


def _interband_ratio(eig: np.ndarray, mu: float, kT: float, degenerate_tol: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    [f(m)-f(n)] / [eps(m)-eps(n)] for every (k,n,m), continuous through
    eps(m)->eps(n) via the L'Hopital limit df/dE. Returns (ratio, dE)
    both shaped (nk, nw, nw), dE[k,n,m] = eps_m(k) - eps_n(k).
    """
    f = fermi_dirac(eig, mu, kT)
    dE = eig[:, None, :] - eig[:, :, None]        # [k,n,m] = eps_m - eps_n
    dF = f[:, None, :] - f[:, :, None]             # [k,n,m] = f_m - f_n

    degenerate = np.abs(dE) < degenerate_tol
    safe_dE = np.where(degenerate, 1.0, dE)
    ratio = np.where(degenerate, 0.0, dF / safe_dE)
    if np.any(degenerate):
        dfdE_n = dfermi_dE(eig, mu, kT)           # (nk, nw), indexed by n
        ratio = np.where(degenerate, dfdE_n[:, :, None], ratio)
    return ratio, dE


def interband_conductivity_correction(
    hr:                 HamiltonianR,
    real_lattice:       np.ndarray,
    recip_lattice:      np.ndarray,
    mesh:               tuple[int, int, int],
    mu:                 float,
    kT:                 float,
    relax_time:         float,
    num_elec_per_state: int = 2,
    degenerate_tol:     float = 1e-10,
) -> np.ndarray:
    """
    Delta_sigma_ij(mu, kT), CRTA: a constant linewidth Gamma = 1/relax_time
    for every band and k-point (see module docstring for the
    Phoebe-matching formula). Unrestricted double sum over ordered pairs
    (n, m), n != m.

    Args:
      mu, kT     : Hartree
      relax_time : atomic time units (hbar/Hartree)

    Returns a (3, 3) real array, same convention as
    `boltzmann.transport_coefficients`'s `elcond`. Diagonal is provably
    >= 0 -- asserted internally as a regression guard.
    """
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    V = abs(np.linalg.det(real_lattice))
    kpts = _uniform_mesh(mesh)
    nk = kpts.shape[0]
    eig, vmat = velocity_matrix(hr, kpts, recip_lattice)
    nw = eig.shape[1]

    Gamma_tot = 2.0 / relax_time   # tau^-1_n + tau^-1_m, constant Gamma for both

    ratio, dE = _interband_ratio(eig, mu, kT, degenerate_tol)   # dE[k,n,m] = eps_m - eps_n
    xC = Gamma_tot + 1j * 2.0 * dE                              # (nk, nw, nw) complex

    weight = ratio / xC
    diag = np.arange(nw)
    weight[:, diag, diag] = 0.0   # exclude b == b' (that's sigma^aiBTE's job)

    # P[n,m,a,b] = v(n,m,a) * conj(v(n,m,b)) -- see module docstring.
    P = np.einsum('knma,knmb->knmab', vmat, vmat.conj())       # (nk, nw, nw, 3, 3)
    total = np.einsum('knm,knmab->ab', weight, P)

    result = -(2.0 * num_elec_per_state / (V * nk)) * total
    imag_scale = np.abs(result.real).max() if np.abs(result.real).max() > 0 else 1.0
    assert np.abs(result.imag).max() < 1e-8 * imag_scale, \
        "Delta_sigma picked up a spurious imaginary part -- see module docstring's reality proof"
    diag_vals = np.diagonal(result.real)
    assert np.all(diag_vals > -1e-8 * max(np.abs(diag_vals).max(), 1e-30)), \
        "Delta_sigma's diagonal came out negative -- see module docstring's positivity proof"
    return result.real


@dataclass
class WignerConductivity:
    """sigma = aibte + interband, all (3, 3), atomic units (see module docstring)."""
    aibte:     np.ndarray
    interband: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.aibte + self.interband

    def to_si(self) -> dict:
        """Convert to SI (1/Ohm/m)."""
        return {
            "aibte":     to_si_units(self.aibte, "electrical_conductivity"),
            "interband": to_si_units(self.interband, "electrical_conductivity"),
            "total":     to_si_units(self.total, "electrical_conductivity"),
        }
