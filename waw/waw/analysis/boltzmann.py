"""
Semiclassical Boltzmann transport, constant relaxation-time approximation
(wannier90's BoltzWann module), from a Wannier Hamiltonian:

  G. Pizzi, D. Volja, B. Kozinsky, M. Fornari, N. Marzari,
  "BoltzWann: A code for the evaluation of thermoelectric and electronic
  transport properties with a maximally-localized Wannier functions basis",
  Comp. Phys. Comm. 185, 422 (2014); DOI:10.1016/j.cpc.2013.09.015

wannier90: `postw90/boltzwann.F90`, `wan_ham.F90`. Band velocities use the
non-degenerate Hellmann-Feynman formula dE_n/dk_a = Re<n|dH/dk_a|n>
(wannier90's default `use_degen_pert=.false.`) -- inaccurate exactly at
band degeneracies.

`transport_distribution_function`'s `relax_time` also accepts a callable
tau(E) for an energy-dependent relaxation time; see
`dos_limited_relaxation_time` for a tau(E) ~ 1/D(E) model.

Units: atomic units throughout -- energies Hartree, k/velocities Bohr^-1 /
Hartree*Bohr, relaxation time in hbar/Hartree. `seebeck_reduced` and
`kappa_moment` are Kelvin-free reduced forms (see `transport_coefficients`).

Conversion to SI via `waw.units.to_si_units(value, quantity, ...)`,
registered for `"electrical_conductivity"`/`"seebeck"`/
`"thermal_conductivity"`/`"thermoelectric_conductivity"`, matching
wannier90's `*_elcond.dat`/`*_seebeck.dat`/`*_kappa.dat`/`*_sigmas.dat`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR
from ..core.distributions import minus_fermi_deriv, gaussian_smearing
from ..units import (
    E_CHARGE, HBAR_SI, BOHR_RADIUS_M, HARTREE_TO_J, K_B_SI,
    register_si_unit, register_from_si_unit,
)
from .dos import density_of_states, _uniform_mesh
from .effective_mass import _h_and_k_derivatives_cartesian
from ._fourier_derivs import h_and_grad_cart_batch, KCHUNK

# wannier90 constants.F90: min_smearing_binwidth_ratio, smearing_cutoff
_MIN_SMEARING_BINWIDTH_RATIO = 2.0
_SMEARING_CUTOFF = 10.0


@dataclass
class TDF:
    """Transport distribution function (atomic units)."""
    energies: np.ndarray   # (nE,) Hartree
    tdf:      np.ndarray   # (nE, 3, 3) Hartree * atomic_time / Bohr, per cell volume


@dataclass
class BoltzmannTransport:
    """
    Boltzmann transport coefficients on a (mu, kT) grid, atomic units --
    see `transport_coefficients`'s docstring for exactly what each
    reduced tensor means and how it maps to physical (SI) quantities.
    """
    mu:               np.ndarray   # (n_mu,) Hartree
    kT:               np.ndarray   # (n_T,) Hartree
    elcond:           np.ndarray   # (n_mu, n_T, 3, 3) Hartree * atomic_time / Bohr
    seebeck_reduced:  np.ndarray   # (n_mu, n_T, 3, 3) dimensionless
    kappa_moment:     np.ndarray   # (n_mu, n_T, 3, 3) Hartree * atomic_time / Bohr


def band_velocities(hr: HamiltonianR, kpts_frac: np.ndarray, recip_lattice: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Band energies and Hellmann-Feynman band velocities on an arbitrary set
    of k-points (non-degenerate formula; see module docstring).

    Args:
      hr           : HamiltonianR
      kpts_frac    : (nk, 3) crystal-coordinate k-points
      recip_lattice: (3, 3) rows = reciprocal lattice vectors, Bohr^-1
                     (2*pi*inv(real_lattice).T convention, as elsewhere)

    Returns:
      eig: (nk, nw) Hartree
      vel: (nk, nw, 3) Hartree*Bohr -- dE_n/dk_cart_a = Re<n|dH/dk_cart_a|n>
    """
    kpts_frac = np.asarray(kpts_frac, dtype=np.float64)
    nk = kpts_frac.shape[0]
    nw = hr.H_R.shape[-1]
    eig = np.empty((nk, nw), dtype=np.float64)
    vel = np.empty((nk, nw, 3), dtype=np.float64)
    for lo in range(0, nk, KCHUNK):
        H0, grad_cart = h_and_grad_cart_batch(hr, kpts_frac[lo:lo + KCHUNK], recip_lattice)
        e, U = torch.linalg.eigh(H0)                                   # (nc,nw),(nc,nw,nw)
        # <i|dH/dk_a|i> for every band i, every Cartesian a
        Gdiag = torch.einsum('kni,kanm,kmi->kai', U.conj(), grad_cart, U)   # (nc,3,nw)
        eig[lo:lo + KCHUNK] = e.cpu().numpy()
        vel[lo:lo + KCHUNK] = Gdiag.real.transpose(-1, -2).cpu().numpy()    # (nc,nw,3)
    return eig, vel


def transport_distribution_function(
    hr:                  HamiltonianR,
    real_lattice:        np.ndarray,
    recip_lattice:       np.ndarray,
    mesh:                tuple[int, int, int],
    energies:            np.ndarray,
    relax_time:          float,
    num_elec_per_state:  int = 2,
    smearing:            float = 0.0,
) -> TDF:
    """
    Transport distribution function Sigma_ij(E) (wannier90's TDF), on a
    uniform Gamma-centred BZ mesh:

        Sigma_ij(E) = (tau/V) sum_{n,k} v_i(n,k) v_j(n,k) delta(E - eps_nk)

    matching `boltzwann.F90`'s `TDF_kpt` + the kweight/cell_volume
    normalization in `boltzwann_main`. `energies` must be a uniformly
    spaced grid (Hartree); the delta function is a "nearest energy bin"
    histogram when smearing/binwidth < 2 (wannier90's
    `min_smearing_binwidth_ratio`, the default), else a Gaussian kernel
    exp(-(dE/smearing)^2)/(smearing*sqrt(pi)) (`smr_type=gauss`; other
    smearing types are not implemented).

    Args:
      relax_time          : EITHER a constant, atomic time units (hbar/Hartree,
                            wannier90's own CRTA convention) OR a callable
                            eig_array -> tau_array (same shape, atomic time
                            units) for an energy-dependent relaxation time,
                            evaluated per band/k-point before binning --
                            see `dos_limited_relaxation_time` for the
                            tau(E) ~ 1/D(E) "DOS-limited" model
      num_elec_per_state  : per-band occupation factor (2 for non-spinor
                            WFs, matching wannier90's default)
      smearing            : Hartree; 0.0 (default) reproduces wannier90's
                            own default (unsmeared TDF)

    Returns TDF(energies, tdf) with tdf[:, i, j] symmetric, in
    Hartree * atomic_time / Bohr (see module docstring).
    """
    energies = np.asarray(energies, dtype=np.float64)
    nE = energies.shape[0]
    binwidth = energies[1] - energies[0]

    cell_volume = abs(np.linalg.det(np.asarray(real_lattice, dtype=np.float64)))
    kpts = _uniform_mesh(mesh)
    nk = kpts.shape[0]
    eig, vel = band_velocities(hr, kpts, recip_lattice)
    nw = eig.shape[1]

    if callable(relax_time):
        tau_kn = np.asarray(relax_time(eig), dtype=np.float64)
        if tau_kn.shape != eig.shape:
            raise ValueError(f"relax_time(eig) must return shape {eig.shape}, got {tau_kn.shape}")
    else:
        tau_kn = np.full(eig.shape, float(relax_time))

    # tau folded in per (k, n); a no-op reshuffle for constant tau
    outer = np.einsum('kni,knj->knij', vel, vel) * tau_kn[:, :, None, None]   # (nk, nw, 3, 3)

    tdf = np.zeros((nE, 3, 3), dtype=np.float64)
    do_smearing = smearing / binwidth >= _MIN_SMEARING_BINWIDTH_RATIO
    if do_smearing:
        # Methfessel-Paxton n=0 == gaussian_smearing's sigma under sigma = smearing/sqrt(2)
        x = energies[:, None, None] - eig[None, :, :]        # (nE, nk, nw)
        mask = np.abs(x / smearing) <= _SMEARING_CUTOFF
        kernel = np.where(mask, gaussian_smearing(x, smearing / np.sqrt(2)), 0.0)
        tdf = np.einsum('ekn,knij->eij', kernel, outer) * num_elec_per_state
    else:
        # unsmeared: nearest-bin histogram (wannier90's default)
        idx = np.rint((eig - energies[0]) / (energies[-1] - energies[0]) * (nE - 1)).astype(np.int64)
        idx = np.clip(idx, 0, nE - 1)
        weighted = outer * (num_elec_per_state / binwidth)   # (nk, nw, 3, 3)
        flat_idx = idx.reshape(-1)
        flat_w = weighted.reshape(-1, 3, 3)
        np.add.at(tdf, flat_idx, flat_w)

    tdf = tdf / (nk * cell_volume)
    return TDF(energies=energies, tdf=tdf)


def dos_limited_relaxation_time(
    dos_energies:   np.ndarray,
    dos_values:     np.ndarray,
    tau_ref:        float,
    e_ref:          float,
    dos_floor_frac: float = 1e-3,
):
    """
    tau(E) = tau_ref * D(e_ref) / D(E) -- phenomenological "DOS-limited
    scattering" model (Garmroudi et al., arXiv:2501.04891): scattering
    rate 1/tau(E) proportional to the density of final states, so sharp
    DOS features give short lifetimes and sparse regions give long ones.
    Qualitative, not first-principles.

    D(E) is linearly interpolated from (dos_energies, dos_values),
    typically `dos.density_of_states`'s output (Hartree grid,
    states/Hartree). Floored at `dos_floor_frac * max(dos_values)` to
    avoid a divide-by-zero in a gap or DOS tail.

    Returns a callable eig_array -> tau_array (atomic time units), for
    `transport_distribution_function`'s `relax_time` argument.
    """
    dos_energies = np.asarray(dos_energies, dtype=np.float64)
    dos_values = np.asarray(dos_values, dtype=np.float64)
    floor = dos_floor_frac * dos_values.max()
    d_ref = max(float(np.interp(e_ref, dos_energies, dos_values)), floor)

    def tau(eig: np.ndarray) -> np.ndarray:
        d = np.interp(eig, dos_energies, dos_values)
        d = np.maximum(d, floor)
        return tau_ref * d_ref / d

    return tau


def transport_coefficients(tdf: TDF, mu_list: np.ndarray, kT_list: np.ndarray) -> BoltzmannTransport:
    """
    Reduce a TDF(E) to transport-coefficient tensors on a (mu, kT) grid
    (Riemann sum, step = the TDF energy-grid spacing), matching
    `boltzwann_main`'s integration before its SI unit conversion:

      elcond(mu,kT)          = int dE (-df/dE) TDF(E) dE
      sigmaS_moment(mu,kT)   = int dE (-df/dE) TDF(E) (E-mu) dE
      kappa_moment(mu,kT)    = int dE (-df/dE) TDF(E) (E-mu)^2 dE
      seebeck_reduced(mu,kT) = -elcond(mu,kT)^-1 @ [sigmaS_moment(mu,kT)/kT]

    `seebeck_reduced` is dimensionless, equal to the physical Seebeck
    coefficient divided by +k_B/e (see `to_si`) -- using kT in Hartree
    rather than T in Kelvin lets it be computed in pure atomic units.

    `kappa_moment` is wannier90's "K coefficient", not the full electronic
    thermal conductivity (kappa = K - S^2*sigma*T); `to_si` returns K in
    SI (W/m/K) mirroring wannier90's `*_kappa.dat`.
    """
    mu_list = np.asarray(mu_list, dtype=np.float64)
    kT_list = np.asarray(kT_list, dtype=np.float64)
    E = tdf.energies
    step = E[1] - E[0]
    n_mu, n_T = len(mu_list), len(kT_list)

    elcond = np.zeros((n_mu, n_T, 3, 3))
    seebeck_reduced = np.zeros((n_mu, n_T, 3, 3))
    kappa_moment = np.zeros((n_mu, n_T, 3, 3))

    for iT, kT in enumerate(kT_list):
        for imu, mu in enumerate(mu_list):
            w = minus_fermi_deriv(E, mu, kT)                       # (nE,)
            base = np.einsum('e,eij->ij', w, tdf.tdf) * step        # (3,3)
            moment1 = np.einsum('e,e,eij->ij', w, E - mu, tdf.tdf) * step
            moment2 = np.einsum('e,e,eij->ij', w, (E - mu) ** 2, tdf.tdf) * step

            elcond[imu, iT] = base
            kappa_moment[imu, iT] = moment2

            det = np.linalg.det(base)
            if det == 0.0:
                seebeck_reduced[imu, iT] = 0.0
            else:
                inv = np.linalg.inv(base)
                seebeck_reduced[imu, iT] = -(inv @ (moment1 / kT))

    return BoltzmannTransport(mu=mu_list, kT=kT_list, elcond=elcond,
                              seebeck_reduced=seebeck_reduced, kappa_moment=kappa_moment)


@register_si_unit("electrical_conductivity")
def _elcond_to_si(elcond_atomic):
    """
    `transport_coefficients`'s `elcond` (Hartree*atomic_time/Bohr) -> SI
    electrical conductivity, 1/(Ohm*m):

        elcond_SI = e^2 * elcond_au / (hbar_SI * BOHR_RADIUS_M)
    """
    return np.asarray(elcond_atomic) * (E_CHARGE ** 2 / (HBAR_SI * BOHR_RADIUS_M))


@register_from_si_unit("electrical_conductivity")
def _elcond_from_si(elcond_si):
    """Inverse of `_elcond_to_si`."""
    return np.asarray(elcond_si) / (E_CHARGE ** 2 / (HBAR_SI * BOHR_RADIUS_M))


@register_si_unit("seebeck")
def _seebeck_to_si(seebeck_reduced):
    """
    `transport_coefficients`'s dimensionless `seebeck_reduced` -> SI
    Seebeck coefficient, V/K: `seebeck_SI = (k_B_SI/e) * seebeck_reduced`.
    Sign convention: negative S for electron-like carriers, positive for
    hole-like.
    """
    return (K_B_SI / E_CHARGE) * np.asarray(seebeck_reduced)


@register_from_si_unit("seebeck")
def _seebeck_from_si(seebeck_si):
    """Inverse of `_seebeck_to_si`."""
    return (E_CHARGE / K_B_SI) * np.asarray(seebeck_si)


@register_si_unit("thermal_conductivity")
def _kappa_to_si(kappa_moment_atomic, *, kT_values):
    """
    `transport_coefficients`'s `kappa_moment` (Hartree*atomic_time/Bohr,
    wannier90's "K coefficient", NOT the full electronic thermal
    conductivity kappa = K - S^2*sigma*T) -> SI, W/(m*K):

        kappa_SI = (HARTREE_TO_J*k_B_SI)/(hbar_SI*BOHR_RADIUS_M) *
                   (kappa_moment_au / kT_values)

    `kT_values` is `bt.kT` (Hartree, not a separately-supplied Kelvin
    temperature -- the formula divides by kT in Hartree directly).
    """
    kT_values = np.asarray(kT_values, dtype=np.float64)
    kappa_si = np.asarray(kappa_moment_atomic) * (HARTREE_TO_J * K_B_SI / (HBAR_SI * BOHR_RADIUS_M))
    return kappa_si / kT_values[None, :, None, None]


@register_from_si_unit("thermal_conductivity")
def _kappa_from_si(kappa_si, *, kT_values):
    """Inverse of `_kappa_to_si` -- same `kT_values` (Hartree) kwarg."""
    kT_values = np.asarray(kT_values, dtype=np.float64)
    kappa_moment_atomic = np.asarray(kappa_si) * kT_values[None, :, None, None]
    return kappa_moment_atomic / (HARTREE_TO_J * K_B_SI / (HBAR_SI * BOHR_RADIUS_M))


@register_si_unit("thermoelectric_conductivity")
def _sigma_seebeck_to_si(elcond_atomic, *, seebeck_reduced):
    """
    The `sigma*Seebeck` thermoelectric conductivity (wannier90's
    *_sigmas.dat), SI A/(m*K): `sigmaS_moment_au/kT` is recovered as
    `-elcond_au @ seebeck_reduced` (`seebeck_reduced`'s own defining
    relation, `transport_coefficients`'s docstring), then converted via
    `sigma_S_SI = e*k_B_SI/(hbar_SI*BOHR_RADIUS_M) * (sigmaS_moment_au/kT)`.
    """
    elcond_atomic = np.asarray(elcond_atomic)
    seebeck_reduced = np.asarray(seebeck_reduced)
    sigmaS_moment_over_kT = -np.einsum('mtij,mtjk->mtik', elcond_atomic, seebeck_reduced)
    return sigmaS_moment_over_kT * (E_CHARGE * K_B_SI / (HBAR_SI * BOHR_RADIUS_M))


@register_from_si_unit("thermoelectric_conductivity")
def _sigma_seebeck_from_si(sigmaS_si, *, seebeck_reduced):
    """
    Inverse of `_sigma_seebeck_to_si`. Not a plain rescaling: solves
    `sigmaS_moment_over_kT = -elcond_atomic @ seebeck_reduced` for
    `elcond_atomic` given the same `seebeck_reduced`, i.e.
    `elcond_atomic = -sigmaS_moment_over_kT @ inv(seebeck_reduced)`.
    Requires `seebeck_reduced` invertible at every (mu, T) point.
    """
    sigmaS_moment_over_kT = np.asarray(sigmaS_si) / (E_CHARGE * K_B_SI / (HBAR_SI * BOHR_RADIUS_M))
    inv_seebeck = np.linalg.inv(np.asarray(seebeck_reduced))
    return -np.einsum('mtik,mtkl->mtil', sigmaS_moment_over_kT, inv_seebeck)


def boltzmann_dos(
    hr: HamiltonianR, mesh: tuple[int, int, int], energies: np.ndarray,
    smearing: float, num_elec_per_state: int = 2,
):
    """
    BoltzWann's companion DOS (`boltz_calc_also_dos`): Gaussian-smeared
    density of states on the BoltzWann k-mesh, normalized so that
    int DOS(E) dE = num_elec_per_state * num_wann.

    Delegates to `dos.density_of_states`: wannier90's Gaussian kernel
    exp(-(x/w)^2)/(w*sqrt(pi)) (Methfessel-Paxton n=0) matches
    `density_of_states`'s exp(-x^2/2sigma^2)/(sigma*sqrt(2pi)) under
    sigma = w/sqrt(2); only `smr_type=gauss` is supported.

    Returns a `dos.DOS` (energies Hartree, dos states/Hartree) already
    scaled by `num_elec_per_state`, unlike `density_of_states`'s own
    output.
    """
    result = density_of_states(hr, mesh=mesh, energies=energies, sigma=smearing / np.sqrt(2))
    result.dos = result.dos * num_elec_per_state
    return result
