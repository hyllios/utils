"""
Electronic viscosity, CRTA (constant relaxation time), following Phoebe's
`src/observable/electron_viscosity.cpp` (`calcRTA`, the RTA path --
Cepellotti et al., J. Phys. Mater. 5 (2022) 035003). The relaxons/
hydrodynamic path there needs the full electron-phonon scattering-matrix
eigenvectors, which this project does not have -- same CRTA-vs-real-
linewidths scoping decision already made for `wigner_transport.py`.

    eta_abcd(mu, kT, tau) = (gs / (V*Nk)) * sum_{k,n}
        k_a * v_b(k,n) * k_c * v_d(k,n) * tau(k,n) * (-df/dE)(eps_kn; mu, kT)

k is the crystal wavevector folded into the first Brillouin zone (the
Wigner-Seitz cell of the reciprocal lattice, `_fold_to_first_bz`) --
unlike conductivity (gauge-invariant under k -> k+G, since only v
enters), viscosity involves k directly, so the BZ representative
matters. Phoebe folds k the same way (`Points::bzToWs`).

Uses the same per-(k,n) relaxation-time convention as `boltzmann.py`
(a constant, or a callable tau(eig), e.g. `dos_limited_relaxation_time`).
eta is real (k and v are both real, v via the non-degenerate
Hellmann-Feynman `band_velocities`, not the complex off-diagonal
`velocity_matrix` from `wigner_transport.py`). Diagonal components
eta_abab are provably >= 0 (each term is tau*(-df/dE)*(k_a*v_b)^2 >= 0);
general eta_abcd is not sign-definite.

UNITS. Atomic units (Hartree/Bohr/atomic-time, see boltzmann.py's module
docstring). `electronic_viscosity` returns a (3,3,3,3) array. Convert with
`waw.units.to_si_units(eta_au, "viscosity")` for Pa*s (kg/(m*s)), Phoebe's
own reported unit for eta_ijkl in Pi_ij = eta_ijkl * du_k/dr_l.

Conversion factor: Phoebe's `constants.h::viscosityAuToSi =
(1/E0)*hbar^2*(1/a0^3)*(1/(hbar/E0))` simplifies to `hbar/a0^3`,
independent of the atomic energy unit E0 (Hartree here, Rydberg there).
"""

from __future__ import annotations

import numpy as np

from ..core.hamiltonian import HamiltonianR
from ..core.distributions import minus_fermi_deriv
from ..units import HBAR_SI, BOHR_RADIUS_M, register_si_unit, register_from_si_unit
from .boltzmann import band_velocities
from .dos import _uniform_mesh

_G_SHELL = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)   # (27, 3) -- one shell of neighbouring reciprocal lattice points


def _fold_to_first_bz(kpts_frac: np.ndarray, recip_lattice: np.ndarray) -> np.ndarray:
    """
    Fold fractional k-points into the Wigner-Seitz cell of the reciprocal
    lattice (Cartesian, Bohr^-1) -- matches Phoebe's `Points::bzToWs`:
    centre the fractional coordinates first, then pick, among one shell
    of neighbouring G-vectors, the periodic image closest to the origin
    (needed for correctness on non-orthogonal reciprocal lattices, where
    simple per-component wrapping is not the true Wigner-Seitz cell).
    """
    kpts_frac = np.asarray(kpts_frac, dtype=np.float64)
    frac = kpts_frac - np.round(kpts_frac)          # centre near zero, (-0.5, 0.5]
    kcart = frac @ recip_lattice                    # (nk, 3)
    G_cart = _G_SHELL @ recip_lattice                # (27, 3)
    candidates = kcart[:, None, :] + G_cart[None, :, :]     # (nk, 27, 3)
    dist2 = (candidates ** 2).sum(axis=-1)
    best = np.argmin(dist2, axis=1)
    return candidates[np.arange(len(kcart)), best]


def electronic_viscosity(
    hr:                  HamiltonianR,
    real_lattice:        np.ndarray,
    recip_lattice:        np.ndarray,
    mesh:                tuple[int, int, int],
    mu:                  float,
    kT:                  float,
    relax_time,
    num_elec_per_state:  int = 2,
) -> np.ndarray:
    """
    eta_abcd(mu, kT), CRTA (see module docstring). `relax_time`: EITHER a
    constant (atomic time units) OR a callable eig_array -> tau_array
    (same convention as `boltzmann.transport_distribution_function`).

    Returns a (3, 3, 3, 3) real array, atomic units (see module
    docstring for the SI conversion via `to_si`).
    """
    real_lattice = np.asarray(real_lattice, dtype=np.float64)
    V = abs(np.linalg.det(real_lattice))
    kpts_frac = _uniform_mesh(mesh)
    nk = kpts_frac.shape[0]
    eig, vel = band_velocities(hr, kpts_frac, recip_lattice)   # (nk,nw), (nk,nw,3)
    kcart = _fold_to_first_bz(kpts_frac, recip_lattice)        # (nk, 3)

    if callable(relax_time):
        tau_kn = np.asarray(relax_time(eig), dtype=np.float64)
        if tau_kn.shape != eig.shape:
            raise ValueError(f"relax_time(eig) must return shape {eig.shape}, got {tau_kn.shape}")
    else:
        tau_kn = np.full(eig.shape, float(relax_time))

    weight = tau_kn * minus_fermi_deriv(eig, mu, kT)           # (nk, nw), >= 0

    # kv[k,n,a,b] = k_a * v_b(k,n)
    kv = np.einsum('ka,knb->knab', kcart, vel)                 # (nk, nw, 3, 3)
    eta = np.einsum('kn,knab,kncd->abcd', weight, kv, kv) * (num_elec_per_state / (V * nk))

    diag = np.array([eta[a, b, a, b] for a in range(3) for b in range(3)])
    assert np.all(diag >= -1e-8 * max(np.abs(diag).max(), 1e-30)), \
        "eta_abab came out negative -- see module docstring's diagonal positivity proof"
    return eta


@register_si_unit("viscosity")
def _viscosity_to_si(eta_au):
    """Convert `electronic_viscosity`'s result to SI (Pa*s = kg/(m*s))."""
    return np.asarray(eta_au) * HBAR_SI / BOHR_RADIUS_M ** 3


@register_from_si_unit("viscosity")
def _viscosity_from_si(eta_si):
    """Inverse of `_viscosity_to_si`."""
    return np.asarray(eta_si) * BOHR_RADIUS_M ** 3 / HBAR_SI
