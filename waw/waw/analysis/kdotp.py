"""
k.p expansion coefficients (postw90 ``berry_task = kdotp``, Wannier90
tutorial 33, BC2N).

Quasi-degenerate (Lowdin) k.p perturbation theory, expanding the
Wannier-interpolated Hamiltonian H(k) around a single k-point to 2nd order,
projected onto a caller-chosen band subset "A". Transcribed from
wannier90's ``berry.F90::berry_get_kdotp``:

    0th order: eig0[n]              = E_n(k0)                      n in A
    1st order: first_order[n,m,a]   = <n|dH/dk_a|m>(k0)             n,m in A
    2nd order: second_order[n,m,a,b] =
        0.5 * <n|d^2H/dk_a dk_b|m>(k0)
        + 0.5 * sum_{r not in A} <n|dH/dk_a|r><r|dH/dk_b|m> *
                (1/(E_n - E_r) + 1/(E_m - E_r))

All matrix elements are taken in the H(k0) EIGENBASIS (the "H-gauge"): n, m,
r index eigenstates of H(k0), not the raw Wannier basis. `dH/dk`/`d^2H/dk^2`
come from the same Fourier-series differentiation every other analysis
module uses (`_fourier_derivs.h_and_hess_cart_batch`); the H-gauge rotation
is the exact einsum pattern `shift_current.py` already uses for its own
`HH_da_bar`/`HH_dadb_bar`.

Related to, but distinct from, `effective_mass.degenerate_effective_mass`,
which assumes the band group is (near-)exactly degenerate and uses a
single group-averaged reference energy E0 -- valid at a true band
extremum only. This module keeps E_n/E_m separate (per-band-pair energy
denominators), exact for any subset A, degenerate or not.

Units: atomic throughout (Hartree / Hartree*Bohr / Hartree*Bohr^2), per
CLAUDE.md -- eV/Angstrom-registered converters below for notebook display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..core.hamiltonian import HamiltonianR
from ..units import (
    HARTREE_TO_EV, BOHR_TO_ANG, EV_TO_HARTREE, ANG_TO_BOHR,
    register_eva_unit, register_from_eva_unit,
)
from ._fourier_derivs import h_and_hess_cart_batch


@dataclass
class KdotpResult:
    """k.p expansion coefficients around a single k-point, for `bands`."""
    kpoint:       np.ndarray   # (3,) fractional k-point the expansion is around
    bands:        tuple        # 0-based band indices into H(kpoint)'s ascending eigenbasis
    eig0:         np.ndarray   # (nA,) real, Hartree -- E_n(k0), 0th order
    first_order:  np.ndarray   # (nA, nA, 3) complex, Hartree*Bohr -- <n|dH/dk_a|m>
    second_order: np.ndarray   # (nA, nA, 3, 3) complex, Hartree*Bohr^2


def kdotp_coefficients(
    hr: HamiltonianR, recip_lattice: np.ndarray, kpoint, bands,
) -> KdotpResult:
    """
    k.p expansion of the Wannier-interpolated Hamiltonian around `kpoint`,
    projected onto the band subset `bands`.

    Args:
      hr, recip_lattice: HamiltonianR + reciprocal lattice (Bohr^-1)
      kpoint: (3,) fractional k-point (postw90's ``kdotp_kpoint``)
      bands : 0-based band indices into H(kpoint)'s ascending eigenbasis --
              the "A" subset (postw90's ``kdotp_bands``, 1-based there).
              Every other band contributes to the 2nd-order virtual-state
              correction as intermediate state "r"; no cutoff on how close
              r may sit to A (formula assumes A is well isolated).
    """
    kpt = np.asarray(kpoint, dtype=np.float64).reshape(1, 3)
    H0, grad_cart, hess_cart = h_and_hess_cart_batch(hr, kpt, recip_lattice)
    H0, grad_cart, hess_cart = H0[0], grad_cart[0], hess_cart[0]   # drop the size-1 k axis

    eig, UU = torch.linalg.eigh(H0)
    HH_da_bar = torch.einsum('ni,anm,mj->aij', UU.conj(), grad_cart, UU)        # (3,nw,nw)
    HH_dadb_bar = torch.einsum('ni,abnm,mj->abij', UU.conj(), hess_cart, UU)    # (3,3,nw,nw)

    eig_np = eig.detach().cpu().numpy()
    HH_da_np = HH_da_bar.detach().cpu().numpy()
    HH_dadb_np = HH_dadb_bar.detach().cpu().numpy()

    nw = eig_np.shape[0]
    bands = tuple(bands)
    band_set = set(bands)
    remote = [r for r in range(nw) if r not in band_set]

    eig0 = eig_np[list(bands)]

    nA = len(bands)
    first_order = np.zeros((nA, nA, 3), dtype=np.complex128)
    second_order = np.zeros((nA, nA, 3, 3), dtype=np.complex128)

    for i, n in enumerate(bands):
        for j, m in enumerate(bands):
            first_order[i, j, :] = HH_da_np[:, n, m]
            for a in range(3):
                for b in range(3):
                    val = 0.5 * HH_dadb_np[a, b, n, m]
                    for r in remote:
                        val += 0.5 * HH_da_np[a, n, r] * HH_da_np[b, r, m] * (
                            1.0 / (eig_np[n] - eig_np[r]) + 1.0 / (eig_np[m] - eig_np[r])
                        )
                    second_order[i, j, a, b] = val

    return KdotpResult(kpoint=kpt[0], bands=bands, eig0=eig0,
                        first_order=first_order, second_order=second_order)


@register_eva_unit("kdotp_first_order")
def _kdotp_first_order_to_eva(value):
    return value * HARTREE_TO_EV * BOHR_TO_ANG


@register_from_eva_unit("kdotp_first_order")
def _kdotp_first_order_from_eva(value):
    return value * EV_TO_HARTREE * ANG_TO_BOHR


@register_eva_unit("kdotp_second_order")
def _kdotp_second_order_to_eva(value):
    return value * HARTREE_TO_EV * BOHR_TO_ANG ** 2


@register_from_eva_unit("kdotp_second_order")
def _kdotp_second_order_from_eva(value):
    return value * EV_TO_HARTREE * ANG_TO_BOHR ** 2
