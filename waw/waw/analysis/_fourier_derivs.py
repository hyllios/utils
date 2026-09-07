"""
Shared low-level helper: analytic k-derivatives of the Wannier Bloch
Hamiltonian H(k) = sum_R e^{2 pi i k.R} H(R) / degen(R), in fractional
(crystal) k-space.

Used by both effective_mass.py (which converts to Cartesian via the
reciprocal lattice, since masses are metric-dependent) and topology.py
(which stays in fractional space throughout, since Berry curvature and
Chern numbers are purely topological and metric-independent).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..core.hamiltonian import HamiltonianR

# k-points processed per batch in the vectorized analysis paths -- bounds peak
# memory of the (chunk, 3, nw, nw) intermediates while still handing each chunk
# to torch as one multi-threaded batched op.
KCHUNK = 4096


def h_and_grad_cart_batch(
    hr: HamiltonianR, kpts_frac: np.ndarray, recip_lattice: np.ndarray,
) -> tuple[Tensor, Tensor]:
    """
    Batched H(k) and Cartesian dH/dk (Bohr^-1) for a k-stack, as torch tensors
    -- the multi-core analogue of `effective_mass._h_and_k_derivatives_cartesian`
    (gradient only). ``grad_cart = inv(recip_lattice) . grad_frac`` (the same
    Jacobian), each component Hermitized.

    Returns (H0 (nk,nw,nw), grad_cart (nk,3,nw,nw)), both complex.
    """
    H0, grad_frac = h_and_grad_frac_batch(hr, kpts_frac)
    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice),
                                dtype=grad_frac.dtype, device=grad_frac.device)
    grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad_frac)
    grad_cart = 0.5 * (grad_cart + grad_cart.conj().transpose(-1, -2))
    return H0, grad_cart


def h_and_grad_frac_batch(hr: HamiltonianR, kpts_frac: np.ndarray) -> tuple[Tensor, Tensor]:
    """
    Batched H(k) and dH/dk_frac for a whole stack of k-points, as torch
    tensors -- the multi-core analogue of `h_and_k_derivatives_frac` (which
    is per-k numpy). One `(nk, nR)` phase matrix drives batched einsums, and
    the downstream `torch.linalg.eigh` over the `(nk, nw, nw)` stack uses all
    BLAS threads (see `waw.set_num_threads`), so per-k Python loops are gone.

    Hessian is not returned (topology only needs the gradient); callers that
    need it still use the per-k `h_and_k_derivatives_frac`.

    Returns (H0, grad_frac):
      H0        : (nk, nw, nw)    complex, Hermitized H(k)
      grad_frac : (nk, 3, nw, nw) complex, dH/dk_frac_a, each Hermitized
    """
    H_R   = hr.H_R.to(dtype=torch.complex128)                 # (nR, nw, nw)
    device = H_R.device
    R     = torch.as_tensor(hr.R_vectors, dtype=torch.float64, device=device)  # (nR, 3)
    degen = torch.as_tensor(hr.degen, dtype=torch.float64, device=device)      # (nR,)
    k     = torch.as_tensor(np.asarray(kpts_frac, dtype=np.float64), device=device)  # (nk, 3)

    phase = torch.exp(2j * np.pi * (k @ R.T)) / degen         # (nk, nR)

    H0 = torch.einsum('kr,rnm->knm', phase, H_R)
    H0 = 0.5 * (H0 + H0.conj().transpose(-1, -2))

    grad_coeff = phase.unsqueeze(-1) * (2j * np.pi * R)       # (nk, nR, 3)
    grad = torch.einsum('kra,rnm->kanm', grad_coeff, H_R)     # (nk, 3, nw, nw)
    grad = 0.5 * (grad + grad.conj().transpose(-1, -2))

    return H0, grad


def h_and_hess_frac_batch(hr: HamiltonianR, kpts_frac: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
    """
    Batched H(k), dH/dk_frac, and d^2H/dk_frac^2 for a k-stack -- the
    vectorized-over-a-k-mesh analogue of `h_and_k_derivatives_frac`'s per-k
    Hessian (needed by shift-current's `HH_dadb_bar`, tutorial 25; the plain
    gradient alone, `h_and_grad_frac_batch`, doesn't return it).

    Same Fourier-series differentiation pattern as `h_and_grad_frac_batch`,
    one more derivative: `coeff_ab(R) = -(2*pi)^2 R_a R_b`.

    Returns (H0, grad_frac, hess_frac):
      H0        : (nk, nw, nw)       complex, Hermitized
      grad_frac : (nk, 3, nw, nw)    complex, dH/dk_frac_a, each Hermitized
      hess_frac : (nk, 3, 3, nw, nw) complex, d^2H/dk_frac_a dk_frac_b, each Hermitized
    """
    H_R   = hr.H_R.to(dtype=torch.complex128)                 # (nR, nw, nw)
    device = H_R.device
    R     = torch.as_tensor(hr.R_vectors, dtype=torch.float64, device=device)  # (nR, 3)
    degen = torch.as_tensor(hr.degen, dtype=torch.float64, device=device)      # (nR,)
    k     = torch.as_tensor(np.asarray(kpts_frac, dtype=np.float64), device=device)  # (nk, 3)

    phase = torch.exp(2j * np.pi * (k @ R.T)) / degen         # (nk, nR)

    H0 = torch.einsum('kr,rnm->knm', phase, H_R)
    H0 = 0.5 * (H0 + H0.conj().transpose(-1, -2))

    grad_coeff = phase.unsqueeze(-1) * (2j * np.pi * R)       # (nk, nR, 3)
    grad = torch.einsum('kra,rnm->kanm', grad_coeff, H_R)     # (nk, 3, nw, nw)
    grad = 0.5 * (grad + grad.conj().transpose(-1, -2))

    RR = -(2 * np.pi) ** 2 * torch.einsum('ra,rb->rab', R, R)     # (nR, 3, 3)
    hess_coeff = phase[:, :, None, None] * RR[None, :, :, :]       # (nk, nR, 3, 3)
    hess = torch.einsum('krab,rnm->kabnm', hess_coeff, H_R)        # (nk, 3, 3, nw, nw)
    hess = 0.5 * (hess + hess.conj().transpose(-1, -2))

    return H0, grad, hess


def h_and_hess_cart_batch(
    hr: HamiltonianR, kpts_frac: np.ndarray, recip_lattice: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Batched H(k), Cartesian dH/dk, and Cartesian d^2H/dk^2 (Bohr^-1/Bohr^-2)
    for a k-stack -- the Cartesian-Jacobian-transformed counterpart of
    `h_and_hess_frac_batch`, same `inv(recip_lattice)` transform
    `h_and_grad_cart_batch`/`effective_mass.py`'s own Hessian rotation use
    (`hess_cart = inv(recip).inv(recip).hess_frac`, contracted on both
    derivative indices since a Hessian has two).

    Returns (H0 (nk,nw,nw), grad_cart (nk,3,nw,nw), hess_cart (nk,3,3,nw,nw)).
    """
    H0, grad_frac, hess_frac = h_and_hess_frac_batch(hr, kpts_frac)
    inv_recip = torch.as_tensor(np.linalg.inv(recip_lattice),
                                dtype=grad_frac.dtype, device=grad_frac.device)
    grad_cart = torch.einsum('ja,kanm->kjnm', inv_recip, grad_frac)
    grad_cart = 0.5 * (grad_cart + grad_cart.conj().transpose(-1, -2))

    hess_cart = torch.einsum('ja,lb,kabnm->kjlnm', inv_recip, inv_recip, hess_frac)
    hess_cart = 0.5 * (hess_cart + hess_cart.conj().transpose(-1, -2))
    return H0, grad_cart, hess_cart


def h_and_k_derivatives_frac(hr: HamiltonianR, k0_frac: np.ndarray) -> tuple:
    """
    Analytic H(k), dH/dk_frac and d^2H/dk_frac^2 at k0_frac, from the
    H(R) Fourier series (exact, no finite-difference step needed since we
    have matrix elements, not just scalar band energies).

    Returns (H0, grad_frac, hess_frac):
      H0        : (nw, nw) complex, Hermitized
      grad_frac : (3, nw, nw) complex, dH/dk_frac_a, each Hermitized
      hess_frac : (3, 3, nw, nw) complex, d^2H/dk_frac_a dk_frac_b, Hermitized
    """
    H_R   = hr.H_R.detach().cpu().numpy()
    R     = hr.R_vectors.astype(np.float64)
    degen = hr.degen.astype(np.float64)

    phase = np.exp(2j * np.pi * (R @ k0_frac)) / degen   # (nR,)

    H0 = np.einsum('r,rnm->nm', phase, H_R)
    H0 = (H0 + H0.conj().T) / 2

    grad_frac = np.einsum('r,ra,rnm->anm', phase, 2j * np.pi * R, H_R)
    for a in range(3):
        grad_frac[a] = (grad_frac[a] + grad_frac[a].conj().T) / 2

    nw = H0.shape[0]
    hess_frac = np.empty((3, 3, nw, nw), dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            coeff = -(2 * np.pi) ** 2 * R[:, a] * R[:, b]
            hess_frac[a, b] = np.einsum('r,rnm->nm', phase * coeff, H_R)
            hess_frac[a, b] = (hess_frac[a, b] + hess_frac[a, b].conj().T) / 2

    return H0, grad_frac, hess_frac
