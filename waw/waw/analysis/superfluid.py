"""
Superfluid weight and superconducting penetration depth, from a Wannier
Hamiltonian:

  Hiorth, Gutierrez-Amigo, Cavignac, Haule, Marques, Torma,
  "Ab-initio superfluid weight and superconducting penetration depth",
  arXiv:2603.10955 (2026).

Assumes multiband mean-field BCS pairing under time-reversal symmetry and
uniform pairing (a single scalar gap Delta on every orbital, no interband
pairing, block-diagonal BdG Hamiltonian per band -- paper's Eqs. 3-5).
Multi-gap materials (e.g. MgB2's sigma/pi gaps) need this function called
once per gap value and combined per Fermi-surface sheet; not automated
here.

Unlike the paper's finite-difference scheme over DFT wavefunctions, this
module uses the analytic H(R) Fourier derivatives shared with
effective_mass.py/topology.py (`_fourier_derivs`), so band velocities and
interband quantum-geometric matrix elements are exact, with no
restriction to the ab-initio mesh.

Units: Eqs. (4)-(7) as published carry no explicit e^2/hbar^2 prefactor
(compare the standard London form j = -(n_s e^2/m*) A). `superfluid_weight`
computes the reduced-units tensor as literally written, atomic units
(Hartree, Cartesian Bohr^-1, divided by cell volume in Bohr^3 per the
paper's explicit 1/V); `waw.units.to_si_units(D_reduced,
"superfluid_weight")` applies the derived e^2/hbar^2 factor to get SI
(A^2 s^2 / (kg m^3), matching n_s e^2/m*) for `penetration_depth` (Eq. 17).

The e^2/hbar^2 prefactor follows from the Peierls substitution
k -> k + (e/hbar) A: since D is built from two k-derivatives, converting
them to A-derivatives contributes (e/hbar)^2 overall. Consistent with
dimensional analysis and with the identical prefactor in the normal-state
Drude weight (Scalapino, White & Zhang, PRL 68, 2830 (1992), ref. [9] of
Hiorth et al.). The combinatorial factors inside the brackets of
Eqs. (4)-(5) themselves are taken as published, not independently
re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR
from ..core.distributions import gaussian_smearing
from ..units import (
    E_CHARGE, HBAR_SI, BOHR_RADIUS_M, HARTREE_TO_J,
    register_si_unit, register_from_si_unit,
)
from ._fourier_derivs import h_and_k_derivatives_frac, h_and_grad_frac_batch, KCHUNK


@dataclass
class SuperfluidWeight:
    """Superfluid weight tensor and its conventional/geometric decomposition."""
    conventional: np.ndarray   # (3, 3) Hartree/Bohr, "reduced units" -- see module docstring
    geometric:    np.ndarray   # (3, 3), same units
    total:        np.ndarray   # (3, 3) = conventional + geometric
    delta:        float        # Hartree, the (uniform) pairing gap used
    mu:           float        # Hartree, chemical potential used
    kT:           float        # Hartree; 0.0 = T=0 limit formulas (Eqs. 6-7)


def _per_k_contributions(
    hr: HamiltonianR, k_frac: np.ndarray, recip_lattice: np.ndarray,
    delta: float, mu: float, kT: float, conv_sigma: float | None = None,
) -> tuple:
    """
    D^{mu nu}_conv(k), D^{mu nu}_geom(k), without the 1/V prefactor
    (Eqs. 4-5 of Hiorth et al., general temperature; kT<=0 uses the exact
    T=0 limit, Eqs. 6-7).

    If conv_sigma is given, D_conv uses the exact Delta -> 0 limit of its
    prefactor, Delta^2/(eps^2+Delta^2)^{3/2} -> 2 delta(eps), represented
    by a Gaussian of numerical width conv_sigma (independent of the
    physical Delta) -- see `superfluid_weight_small_gap`. D_geom stays on
    the exact, Delta-dependent formula.

    Everything (hr, delta, mu, kT, conv_sigma) is in atomic units.
    """
    H0, grad_frac, _ = h_and_k_derivatives_frac(hr, k_frac)
    eigvals, eigvecs = np.linalg.eigh(H0)
    eps = eigvals - mu
    nw = len(eps)

    inv_recip = np.linalg.inv(recip_lattice)
    grad_cart = np.einsum('ja,anm->jnm', inv_recip, grad_frac)          # (3, nw, nw), Wannier basis
    G = np.einsum('ni,jnm,mk->jik', eigvecs.conj(), grad_cart, eigvecs)  # (3, nw, nw), eigenbasis

    E = np.sqrt(eps ** 2 + delta ** 2)
    if kT <= 0:
        occ = 1.0 / E
        occ_deriv = np.zeros(nw)
    else:
        beta = 1.0 / kT
        occ = np.tanh(beta * E / 2) / E
        occ_deriv = -beta / (2 * np.cosh(beta * E / 2) ** 2)

    conv = np.zeros((3, 3))
    if conv_sigma is None:
        for m in range(nw):
            prefac = (occ[m] + occ_deriv[m]) * (delta ** 2 / E[m] ** 2)
            v_m = G[:, m, m].real   # Hellmann-Feynman band velocity, exact
            conv += prefac * np.outer(v_m, v_m)
    else:
        gauss = 2.0 * gaussian_smearing(eps, conv_sigma)
        for m in range(nw):
            v_m = G[:, m, m].real
            conv += gauss[m] * np.outer(v_m, v_m)

    geom = np.zeros((3, 3), dtype=np.complex128)
    for m in range(nw):
        for n in range(nw):
            if m == n:
                continue
            denom = eps[n] + eps[m]
            if abs(denom) < 1e-10 or abs(eps[m] - eps[n]) < 1e-10:
                continue   # ill-defined at eps_n = -eps_m or a degeneracy
            weight = (occ[m] - occ[n]) * delta ** 2 * (eps[n] - eps[m]) / denom
            # <d_mu m|n><n|d_nu m> = G[mu,m,n] G[nu,n,m] / (eps_m - eps_n)^2 (1st-order
            # perturbation theory for |d_mu m> = sum_{p!=m} |p><p|dH/dk_mu|m>/(eps_m-eps_p))
            term = np.outer(G[:, m, n], G[:, n, m]) / (eps[m] - eps[n]) ** 2
            geom += weight * (term + term.conj().T)

    return conv, geom.real


def _batch_k_contributions_sum(
    hr: HamiltonianR, kpts_frac: np.ndarray, recip_lattice: np.ndarray,
    delta: float, mu: float, kT: float, conv_sigma: float | None = None,
) -> tuple:
    """
    Same physics as `_per_k_contributions`, vectorized over a k-batch and
    summed over it (torch, all bands/pairs at once instead of a per-k,
    per-band Python loop). Returns (conv_sum, geom_sum), each (3, 3) real
    numpy, summed (not averaged) over the batch.
    """
    H0, grad_frac = h_and_grad_frac_batch(hr, kpts_frac)   # (nk,nw,nw), (nk,3,nw,nw)
    eigvals, eigvecs = torch.linalg.eigh(H0)               # (nk,nw), (nk,nw,nw)
    eps = eigvals - mu                                     # (nk,nw)
    nw = eps.shape[-1]

    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice), dtype=grad_frac.dtype)
    grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad_frac)             # (nk,3,nw,nw)
    G = torch.einsum('kni,kjnm,kmp->kjip', eigvecs.conj(), grad_cart, eigvecs)  # (nk,3,nw,nw)

    E = torch.sqrt(eps ** 2 + delta ** 2)
    if kT <= 0:
        occ = 1.0 / E
        occ_deriv = torch.zeros_like(occ)
    else:
        beta = 1.0 / kT
        occ = torch.tanh(beta * E / 2) / E
        occ_deriv = -beta / (2 * torch.cosh(beta * E / 2) ** 2)

    v = torch.diagonal(G, dim1=-2, dim2=-1).real   # (nk,3,nw) band velocities <m|dH/dk|m>
    if conv_sigma is None:
        prefac = (occ + occ_deriv) * (delta ** 2 / E ** 2)          # (nk,nw)
    else:
        prefac = 2.0 * torch.exp(-0.5 * (eps / conv_sigma) ** 2) / (conv_sigma * np.sqrt(2 * np.pi))
    conv_sum = torch.einsum('km,kjm,klm->jl', prefac, v, v)

    # Off-diagonal (m != n) geometric term, masked for the ill-defined
    # eps_n = -eps_m / degenerate cases (see _per_k_contributions).
    eps_m, eps_n = eps[:, :, None], eps[:, None, :]                  # (nk,nw,1)/(nk,1,nw)
    occ_m, occ_n = occ[:, :, None], occ[:, None, :]
    denom = eps_n + eps_m
    diff  = eps_m - eps_n
    off_diag = ~torch.eye(nw, dtype=torch.bool, device=eps.device)
    mask = off_diag[None, :, :] & (denom.abs() >= 1e-10) & (diff.abs() >= 1e-10)
    diff2_safe = torch.where(mask, diff ** 2, torch.ones_like(diff))
    weight = torch.where(mask, (occ_m - occ_n) * delta ** 2 * (eps_n - eps_m) / denom,
                          torch.zeros_like(denom))
    A = (weight / diff2_safe).to(G.dtype)[:, None, :, :] * G   # (nk,3,nw,nw): A[k,j,m,n]
    term_sum = torch.einsum('kjmn,klnm->jl', A, G)             # sum_{k,m,n} A[k,j,m,n]*G[k,l,n,m]
    geom_sum = term_sum + term_sum.conj().transpose(-1, -2)

    return conv_sum.cpu().numpy(), geom_sum.real.cpu().numpy()


def superfluid_weight_at_k(
    hr: HamiltonianR, k_frac: np.ndarray, real_lattice: np.ndarray,
    delta: float, mu: float, kT: float = 0.0, conv_sigma: float | None = None,
) -> tuple:
    """
    Band-resolved conventional/geometric superfluid weight contributions
    at a single k-point (Eqs. 4-5 of Hiorth et al.), including the 1/V
    prefactor. Useful for plotting along a high-symmetry path the way
    Fig. 2 of the paper does.

    conv_sigma: see `superfluid_weight_small_gap` -- if given, D_conv uses
    the Delta -> 0 Gaussian-smeared limit instead of the exact formula.

    Returns (D_conv_k, D_geom_k), each (3, 3), in Hartree/Bohr.
    """
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    V = abs(np.linalg.det(real_lattice))
    conv, geom = _per_k_contributions(hr, np.asarray(k_frac, dtype=np.float64),
                                       recip_lattice, delta, mu, kT, conv_sigma)
    return conv / V, geom / V


def superfluid_weight(
    hr:           HamiltonianR,
    real_lattice: np.ndarray,
    delta:        float,
    mu:           float,
    mesh:         tuple = (10, 10, 10),
    kT:           float = 0.0,
    conv_sigma:   float | None = None,
) -> SuperfluidWeight:
    """
    Brillouin-zone-averaged superfluid weight tensor, decomposed into
    conventional and geometric contributions (Eqs. 4-5 of Hiorth et al.,
    T=0 limit by default).

    D^{mu nu}_total = (1/N_k) sum_k D^{mu nu}_k, where each D^{mu nu}_k
    already includes the paper's explicit 1/V (unit cell volume) factor
    -- this normalization follows from sum_k f(k) -> V N_k int(d^3k/(2pi)^3)
    f(k) for a uniform mesh of N_k points spanning the BZ.

    Args:
      hr           : HamiltonianR
      real_lattice : (3, 3) real-space lattice rows, Bohr
      delta        : Hartree, the uniform pairing gap
      mu           : Hartree, chemical potential (not computed here -- supply
                     from a separate band-filling/Fermi-level calculation)
      mesh         : (N1, N2, N3) uniform BZ mesh
      kT           : Hartree; 0.0 (default) uses the exact T=0 limit formulas
      conv_sigma   : Hartree or None. If given, D_conv is computed via the
                     Delta -> 0 limit (see `superfluid_weight_small_gap`)
                     instead of the exact formula -- prefer calling that
                     function directly, this is exposed here mainly so
                     `superfluid_weight_at_k` and `superfluid_weight`
                     share one implementation.

    With the exact formula (conv_sigma=None), the conventional term's
    prefactor Delta^2/(eps^2+Delta^2)^{3/2} is sharply peaked within an
    energy window ~Delta around the Fermi level, so small gaps need a much
    denser mesh than typical DOS/band-structure use (this does a plain
    BZ sum, not the paper's kernel-regression acceleration for coarse
    meshes). `superfluid_weight_small_gap` avoids this when Delta << bandwidth.
    """
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    V = abs(np.linalg.det(real_lattice))

    N1, N2, N3 = mesh
    i, j, k = np.meshgrid(np.arange(N1), np.arange(N2), np.arange(N3), indexing="ij")
    kpts_frac = np.stack([i.ravel() / N1, j.ravel() / N2, k.ravel() / N3], axis=-1)

    conv_total = np.zeros((3, 3))
    geom_total = np.zeros((3, 3))
    for lo in range(0, len(kpts_frac), KCHUNK):
        conv, geom = _batch_k_contributions_sum(
            hr, kpts_frac[lo:lo + KCHUNK], recip_lattice, delta, mu, kT, conv_sigma)
        conv_total += conv
        geom_total += geom

    nk = N1 * N2 * N3
    conv_total = conv_total / (nk * V)
    geom_total = geom_total / (nk * V)

    return SuperfluidWeight(
        conventional=conv_total, geometric=geom_total,
        total=conv_total + geom_total, delta=delta, mu=mu, kT=kT,
    )


def superfluid_weight_small_gap(
    hr:           HamiltonianR,
    real_lattice: np.ndarray,
    delta:        float,
    mu:           float,
    sigma:        float,
    mesh:         tuple = (20, 20, 20),
) -> SuperfluidWeight:
    """
    Superfluid weight in the small-gap limit (Delta << bandwidth), where
    the conventional term's prefactor is replaced by its exact Delta -> 0
    distributional limit:

        Delta^2 / (eps^2 + Delta^2)^(3/2)  ->  2 delta(eps)     (Delta -> 0)

    D_conv then reduces to a Fermi-surface/DOS-type integral,

        D^{mu nu}_conv(Delta -> 0) = (2/V) rho(mu) <v_mu v_nu>_{eps=mu}

    which converges with mesh density like an ordinary DOS calculation
    (see `dos.density_of_states`) instead of needing mesh resolution
    ~ Delta. Evaluated via Gaussian smearing of numerical width `sigma`
    (decoupled from the physical Delta, which enters only D_geom).

    D_geom is not reformulated -- it has no sharp Delta-scale feature and
    is evaluated with the ordinary exact formula at the true Delta.

    Not appropriate when Delta is comparable to or exceeds the relevant
    bandwidth (e.g. flat-band materials) -- use `superfluid_weight` there.
    """
    return superfluid_weight(hr, real_lattice, delta, mu, mesh=mesh, kT=0.0,
                              conv_sigma=sigma)


@register_si_unit("superfluid_weight")
def reduced_to_si(D_reduced: np.ndarray | float) -> np.ndarray:
    """
    Convert a superfluid-weight tensor from this module's reduced units
    (Hartree/Bohr, as returned by `superfluid_weight`/
    `superfluid_weight_at_k`) to SI units (A^2 s^2 / (kg m^3), matching
    n_s e^2/m*) via the derived e^2/hbar^2 prefactor (see module
    docstring). Registered as `waw.units.to_si_units(D_reduced,
    "superfluid_weight")`.
    """
    D_reduced = np.asarray(D_reduced, dtype=np.float64)
    return D_reduced * (E_CHARGE ** 2 / HBAR_SI ** 2) * HARTREE_TO_J / BOHR_RADIUS_M


@register_from_si_unit("superfluid_weight")
def si_to_reduced(D_si: np.ndarray | float) -> np.ndarray:
    """
    Inverse of `reduced_to_si`: SI (A^2 s^2 / (kg m^3)) back to reduced
    units (Hartree/Bohr). Registered as `waw.units.from_si_units(D_si,
    "superfluid_weight")`.
    """
    D_si = np.asarray(D_si, dtype=np.float64)
    return D_si / ((E_CHARGE ** 2 / HBAR_SI ** 2) * HARTREE_TO_J / BOHR_RADIUS_M)


def penetration_depth(Ds: np.ndarray | float, mu0: float = 4 * np.pi * 1e-7) -> np.ndarray:
    """
    London penetration depth from the superfluid weight (Eq. 17):

        lambda_L = (mu_0 Ds)^(-1/2)

    Ds must already be in SI units (A^2 s^2 / (kg m^3), i.e. the same
    units as n_s e^2/m*) -- use `reduced_to_si` to convert the output of
    `superfluid_weight`/`superfluid_weight_at_k` first.

    Ds should be a scalar or a single diagonal component (e.g.
    Ds_xx, Ds_zz for an anisotropic material, as in the paper's
    Kagome-material analysis), not the full (3, 3) tensor -- Eq. 17
    isn't meaningful applied element-wise to off-diagonal entries.
    """
    Ds = np.asarray(Ds, dtype=np.float64)
    return 1.0 / np.sqrt(mu0 * Ds)
