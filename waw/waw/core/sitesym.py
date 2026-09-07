"""
Symmetry-adapted Wannier functions (SAWF).

R. Sakuma, "Symmetry-adapted Wannier functions in the maximal
localization procedure", PRB 87, 235109 (2013). Transcribed from
wannier90's src/sitesym.F90 (`site_symmetry = .true.` / `write_dmn`).

The crystal's point-group symmetry operations act simultaneously on:
  * the k-mesh (each operation R permutes the mesh, k -> Rk)
  * the ab-initio band basis at each k (`d_matrix_band`, from the .dmn file)
  * the target Wannier-function basis (`d_matrix_wann`)

Only the irreducible k-points (`ir2ik`) carry independent numerical work;
every other k-point's V/U/Z is derived from an irreducible representative
via the group action.

Two symmetrization primitives recur throughout (matching wannier90's own
naming):
  * "extract" at an irreducible k: build a subspace invariant under k's
    own little group (stabilizer) and re-orthonormalize
    (`symmetrize_u_irr`, `extract_symmetrized_subspace`).
  * "broadcast" an irreducible-k quantity to the whole mesh via the group
    action: X(Rk) = d_left(R,k) X(k) d_right(R,k)^dagger.

Dimensionless/gauge machinery only -- no eV/Bohr unit conversions, matching
core.disentangle/core.optim's atomic-unit convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class SiteSymmetry:
    """
    Symmetry data for SAWF, as PyTorch tensors -- `interfaces.wannier90.
    io.read_dmn`'s output, converted from numpy.

    nsymmetry, nkptirr, num_kpts, num_bands, num_wann : int
    ik2ir  : (num_kpts,) long   0-based irreducible-rep index of each k
    ir2ik  : (nkptirr,)  long   0-based full-BZ k-index of each irreducible rep
    kptsym : (nsymmetry, nkptirr) long  full-BZ k-index symmetry `isym` maps
             irreducible rep `ir` to
    d_matrix_wann : (num_wann, num_wann, nsymmetry, nkptirr) complex
    d_matrix_band : (num_bands, num_bands, nsymmetry, nkptirr) complex
    """
    nsymmetry:     int
    nkptirr:       int
    num_kpts:      int
    num_bands:     int
    num_wann:      int
    ik2ir:         Tensor
    ir2ik:         Tensor
    kptsym:        Tensor
    d_matrix_wann: Tensor
    d_matrix_band: Tensor


def site_symmetry_from_dmn(dmn: dict, dtype=torch.complex128, device=None) -> SiteSymmetry:
    """Build a `SiteSymmetry` from `interfaces.wannier90.io.read_dmn`'s dict."""
    return SiteSymmetry(
        nsymmetry=dmn["nsymmetry"], nkptirr=dmn["nkptirr"],
        num_kpts=dmn["num_kpts"], num_bands=dmn["num_bands"], num_wann=dmn["num_wann"],
        ik2ir=torch.as_tensor(dmn["ik2ir"], dtype=torch.long, device=device),
        ir2ik=torch.as_tensor(dmn["ir2ik"], dtype=torch.long, device=device),
        kptsym=torch.as_tensor(dmn["kptsym"], dtype=torch.long, device=device),
        d_matrix_wann=torch.as_tensor(dmn["d_matrix_wann"], dtype=dtype, device=device),
        d_matrix_band=torch.as_tensor(dmn["d_matrix_band"], dtype=dtype, device=device),
    )


# ---------------------------------------------------------------------------
# Broadcast bookkeeping: for every k in the mesh, which (ir, isym) reaches it
# ---------------------------------------------------------------------------

def _build_broadcast_table(kptsym: Tensor, ir2ik: Tensor, num_kpts: int) -> tuple[Tensor, Tensor]:
    """
    For every k-point, find one (ir, isym) pair with kptsym[isym, ir] == k,
    preferring the first ir and, within that ir, the first isym (isym=0 is
    the identity) -- matching wannier90's "first symmetry to reach an
    unvisited k wins" bookkeeping.

    Returns (src_ir, src_isym), each (num_kpts,) long, fully assigned.
    """
    nsym, nkptirr = kptsym.shape
    kptsym_np = kptsym.cpu().numpy()
    ir2ik_np = ir2ik.cpu().numpy()

    src_ir = -np.ones(num_kpts, dtype=np.int64)
    src_isym = -np.ones(num_kpts, dtype=np.int64)
    for ir in range(nkptirr):
        for isym in range(nsym):
            k = kptsym_np[isym, ir] if isym > 0 else ir2ik_np[ir]
            if src_ir[k] == -1:
                src_ir[k] = ir
                src_isym[k] = isym
    if (src_ir < 0).any():
        missing = np.nonzero(src_ir < 0)[0]
        raise ValueError(f"dis_spheres/site-symmetry k-points not reached by any symmetry: {missing}")
    return (torch.as_tensor(src_ir, dtype=torch.long),
            torch.as_tensor(src_isym, dtype=torch.long))


def broadcast_matrix(
    M_irr: Tensor, d_left: Tensor, d_right: Tensor,
    kptsym: Tensor, ir2ik: Tensor, num_kpts: int,
) -> Tensor:
    """
    Broadcast a quantity known at every irreducible k to the full mesh via
    the group action: M(Rk) = d_left(R,k) . M(k) . d_right(R,k)^dagger.

    Args:
      M_irr  : (nkptirr, dl, dr) complex, at the irreducible representatives
      d_left : (dl, dl, nsymmetry, nkptirr) complex (`d_matrix_band` during
               disentanglement, `d_matrix_wann` during spread minimization)
      d_right: (dr, dr, nsymmetry, nkptirr) complex (`d_matrix_wann`)
      kptsym, ir2ik : as `SiteSymmetry`

    Returns M_full: (num_kpts, dl, dr) complex.
    """
    src_ir, src_isym = _build_broadcast_table(kptsym, ir2ik, num_kpts)
    Dl = d_left[:, :, src_isym, src_ir].permute(2, 0, 1)     # (nk, dl, dl)
    Dr = d_right[:, :, src_isym, src_ir].permute(2, 0, 1)    # (nk, dr, dr)
    M_src = M_irr[src_ir]                                     # (nk, dl, dr)
    return torch.matmul(torch.matmul(Dl, M_src), Dr.conj().transpose(-1, -2))


# ---------------------------------------------------------------------------
# Self-consistent stabilizer symmetrization at one irreducible k
# ---------------------------------------------------------------------------

def _polar_orthonormalize(M: Tensor) -> Tensor:
    """
    Loewdin/polar orthonormalization: the closest semi-unitary matrix to M
    (same construction as `core.disentangle._init_V_amn`'s U_loewdin, and
    wannier90's own `orthogonalize_u`, an SVD-based polar decomposition).
    M: (n, m) with n >= m. Returns (n, m) with columns orthonormal.
    """
    P, _, Qh = torch.linalg.svd(M, full_matrices=False)
    return P @ Qh


def symmetrize_u_irr(
    U_irr: Tensor, sitesym: SiteSymmetry, ir: int,
    d_left: Tensor | None = None,
    n_iter: int = 100, eps: float = 1e-9,
) -> Tensor:
    """
    Self-consistent stabilizer symmetrization of U at ONE irreducible k
    (wannier90's `symmetrize_ukirr`): iterate

        U <- orthonormalize( (1/n_stab) sum_{R: Rk=k} d_left(R,k) U D(R,k)^dagger )

    over the stabilizer subgroup (symmetries R with kptsym[R,ir] == ir2ik[ir]),
    until self-consistent. A trivial stabilizer (only the identity) returns
    the plain orthonormalization of U_irr immediately.

    Args:
      U_irr  : (d_left_dim, num_wann) complex, current value at this
               irreducible k (d_left_dim = num_bands during disentanglement,
               num_wann during spread minimization)
      sitesym: SiteSymmetry
      ir     : which irreducible representative (0-based)
      d_left : (d_left_dim, d_left_dim, nsymmetry, nkptirr), defaults to
               sitesym.d_matrix_band (the disentanglement-phase convention;
               pass sitesym.d_matrix_wann explicitly for the spread-
               minimization phase, where both dims are num_wann)

    Returns the symmetrized, orthonormal U (same shape as U_irr).
    """
    if d_left is None:
        d_left = sitesym.d_matrix_band
    ik = int(sitesym.ir2ik[ir])
    stab = [isym for isym in range(sitesym.nsymmetry)
            if int(sitesym.kptsym[isym, ir]) == ik]
    if len(stab) == 1:
        return _polar_orthonormalize(U_irr)

    # Stabilizer matrices don't change across iterations, only U does --
    # hoist them out of the n_iter loop and batch the sum as two matmuls.
    n_stab = len(stab)
    stab_idx = torch.as_tensor(stab, dtype=torch.long, device=U_irr.device)
    Dl_H = d_left[:, :, stab_idx, ir].permute(2, 0, 1).conj().transpose(-1, -2)   # (n_stab, dl, dl)
    Dw = sitesym.d_matrix_wann[:, :, stab_idx, ir].permute(2, 0, 1)               # (n_stab, dw, dw)

    U = U_irr
    for _ in range(n_iter):
        usum = torch.matmul(torch.matmul(Dl_H, U), Dw).sum(dim=0) / n_stab
        diff = (U - usum).abs().sum().item()
        U = _polar_orthonormalize(usum)
        if diff < eps:
            break
    return U


# ---------------------------------------------------------------------------
# Disentanglement: symmetrized Z-matrix + irreducible-k extraction
# ---------------------------------------------------------------------------

def symmetrize_zmatrix(Z_full: Tensor, sitesym: SiteSymmetry) -> Tensor:
    """
    Symmetrize the disentanglement Z-matrix at every irreducible k
    (wannier90's `sitesym_symmetrize_zmatrix`):

        Z(k) <- sum_R d(R,k)^dagger Z(Rk) d(R,k)   (over the full symmetry
                                                     group, k = ir2ik[ir])

    The sum is built from every orbit member's Z (each visited once,
    globally across all irreducible reps), plus the representative's own
    stabilizer-folded contribution, normalized by the stabilizer order.

    Args:
      Z_full : (num_kpts, num_bands, num_bands) complex, the ordinary
               (not yet symmetrized) Z-matrix at every k
               (`core.disentangle._build_Z`'s output)
      sitesym: SiteSymmetry

    Returns Z_irr: (nkptirr, num_bands, num_bands) complex, symmetrized.
    """
    nkptirr, nsym = sitesym.nkptirr, sitesym.nsymmetry
    d_band = sitesym.d_matrix_band
    kptsym = sitesym.kptsym.cpu().numpy()
    ir2ik = sitesym.ir2ik.cpu().numpy()

    Z_irr = torch.stack([Z_full[ir2ik[ir]].clone() for ir in range(nkptirr)])
    lfound = np.zeros(sitesym.num_kpts, dtype=bool)

    for ir in range(nkptirr):
        ik = ir2ik[ir]
        lfound[ik] = True
        for isym in range(1, nsym):
            irk = kptsym[isym, ir]
            if lfound[irk]:
                continue
            lfound[irk] = True
            D = d_band[:, :, isym, ir]
            Z_irr[ir] = Z_irr[ir] + D.conj().transpose(-1, -2) @ Z_full[irk] @ D

        stab = [isym for isym in range(1, nsym) if kptsym[isym, ir] == ik]
        if stab:
            Z_snapshot = Z_irr[ir].clone()
            for isym in stab:
                D = d_band[:, :, isym, ir]
                Z_irr[ir] = Z_irr[ir] + D.conj().transpose(-1, -2) @ Z_snapshot @ D
        n_stab = int((kptsym[:, ir] == ik).sum())
        Z_irr[ir] = Z_irr[ir] / n_stab

    return Z_irr


def _solve_2x2_generalized_eigh_max(H: Tensor, S: Tensor) -> Tensor:
    """
    Solve the 2x2 generalized Hermitian eigenproblem H v = w S v (S positive
    definite) via Cholesky reduction (A x = w B x, B = L L^dagger =>
    L^-1 A L^-dagger y = w y, x = L^-dagger y); returns only the eigenvector
    for the LARGEST eigenvalue -- all `extract_symmetrized_subspace`'s
    band-by-band steepest-ascent loop needs.
    """
    L = torch.linalg.cholesky(S)
    Linv = torch.linalg.inv(L)
    A = Linv @ H @ Linv.conj().transpose(-1, -2)
    _, Y = torch.linalg.eigh(A)   # ascending
    V = Linv.conj().transpose(-1, -2) @ Y
    return V[:, -1]


def extract_symmetrized_subspace(
    Z: Tensor, sitesym: SiteSymmetry, nw: int, ir: int,
    U_init: Tensor | None = None, d_left: Tensor | None = None,
    n_iter: int = 50, conv_tol: float = 1e-10,
) -> Tensor:
    """
    Extract the nw-dimensional subspace at irreducible rep `ir` maximizing
    Tr(U^dagger Z U) subject to U^dagger U = I and stabilizer covariance --
    wannier90's `sitesym_dis_extract_symmetry`, a band-by-band 2x2
    generalized-eigenvalue steepest-ascent iteration.

    Each outer iteration: deltaU = ZU - U(U^dagger Z U) is the steepest-
    ascent direction for Tr(U^dagger Z U) projected orthogonal to U's span;
    converged when deltaU ~ 0. Otherwise, for each column i independently,
    solve the 2x2 generalized eigenvalue problem in span{U[:,i], deltaU[:,i]}
    and move to the Rayleigh-quotient-maximizing combination (an exact
    per-orbital line search, no step size). This band-by-band update can
    rotate within a degenerate eigenspace of Z as stabilizer covariance
    demands, which a single global `eigh` cannot (its basis choice within a
    degenerate eigenspace is arbitrary, not necessarily equivariant). After
    each sweep, re-symmetrize via `symmetrize_u_irr`.

    Args:
      Z      : (n, n) complex, already symmetrized (`symmetrize_zmatrix`'s
               output at this `ir`) and restricted to whatever band subspace
               the caller wants -- `n` need not be `sitesym.num_bands`.
      ir     : which irreducible representative to extract at (0-based)
      U_init : (n, nw) complex, optional initial guess (e.g. warm-started
               from the previous disentanglement sweep); None builds a
               fresh top-nw `eigh(Z)` guess.
      d_left : (n, n, nsymmetry, nkptirr) complex, matching `Z`'s band
               subspace. Defaults to `sitesym.d_matrix_band` (only valid
               when `Z` is unrestricted, dimension num_bands).

    Returns U: (n, nw) complex, orthonormal columns, stabilizer-covariant.
    """
    if d_left is None:
        d_left = sitesym.d_matrix_band
    if U_init is None:
        _, eigvecs = torch.linalg.eigh(Z)   # ascending
        U_init = eigvecs[:, -nw:]
    U = symmetrize_u_irr(U_init, sitesym, ir, d_left=d_left)

    prev_norm = None
    for _ in range(n_iter):
        ZU = Z @ U
        lam = U.conj().transpose(-1, -2) @ ZU
        deltaU = ZU - U @ lam
        delta_norm = deltaU.abs().sum().item()
        if delta_norm < conv_tol:
            break
        # Loewdin renormalization introduces a small residual floor deltaU
        # can't beat -- stop once progress stalls there.
        if prev_norm is not None and delta_norm > prev_norm * 0.999:
            break
        prev_norm = delta_norm

        U_new = U.clone()
        for i in range(nw):
            u_i, du_i = U[:, i], deltaU[:, i]
            sp22 = torch.vdot(du_i, du_i).real
            if sp22.abs() < 1e-10:
                continue   # column already converged
            Zdu_i = Z @ du_i
            hp11 = torch.vdot(u_i, ZU[:, i]).real.to(Z.dtype)
            hp12 = torch.vdot(ZU[:, i], du_i)
            hp22 = torch.vdot(du_i, Zdu_i).real.to(Z.dtype)
            sp11 = torch.vdot(u_i, u_i).real.to(Z.dtype)
            sp12 = torch.vdot(u_i, du_i)
            sp22c = sp22.to(Z.dtype)

            H2 = torch.stack([torch.stack([hp11, hp12]),
                              torch.stack([hp12.conj(), hp22])])
            S2 = torch.stack([torch.stack([sp11, sp12]),
                              torch.stack([sp12.conj(), sp22c])])
            v = _solve_2x2_generalized_eigh_max(H2, S2)
            U_new[:, i] = v[0] * u_i + v[1] * du_i

        U = symmetrize_u_irr(U_new, sitesym, ir, d_left=d_left)

    return U


# ---------------------------------------------------------------------------
# Spread minimization: gradient reduction to the irreducible wedge
# ---------------------------------------------------------------------------

def reduce_gradient_to_irr(
    G_full: Tensor, sitesym: SiteSymmetry, d_left: Tensor | None = None,
) -> Tensor:
    """
    Reduce a full-mesh (Euclidean or Riemannian) gradient to the
    irreducible wedge -- wannier90's `sitesym_symmetrize_gradient`, modes 1
    and 2 applied back-to-back.

    Mode 1: at each irreducible k, sum every orbit member's gradient rotated
    back into the representative's frame,

        G(k) <- sum_R Dl(R,k)^dagger G(Rk) Dw(R,k)   (R over the whole
                                                       group; k = ir2ik[ir])

    Mode 2: fold in the little-group stabilizer average at k itself
    (symmetries R with Rk = k, including the identity),

        G(k) <- (1/n_stab) sum_{R: Rk=k} Dl(R,k)^dagger G(k) Dw(R,k)

    `d_left` must match whatever `_symmetrize_and_broadcast` used to build
    U, since G lives in the same space as U. Defaults to `d_matrix_wann`
    (both indices Wannier-gauge, after genuine disentanglement); pass
    `d_matrix_band` for isolated bands.

    Args:
      G_full : (num_kpts, num_wann, num_wann) complex
      sitesym: SiteSymmetry

    Returns G_irr: (nkptirr, num_wann, num_wann) complex.
    """
    if d_left is None:
        d_left = sitesym.d_matrix_wann
    nkptirr, nsym = sitesym.nkptirr, sitesym.nsymmetry
    d_wann = sitesym.d_matrix_wann
    kptsym = sitesym.kptsym.cpu().numpy()
    ir2ik = sitesym.ir2ik.cpu().numpy()

    G_irr = torch.stack([G_full[ir2ik[ir]].clone() for ir in range(nkptirr)])
    lfound = np.zeros(sitesym.num_kpts, dtype=bool)

    for ir in range(nkptirr):
        ik = ir2ik[ir]
        lfound[ik] = True
        for isym in range(1, nsym):
            irk = kptsym[isym, ir]
            if lfound[irk]:
                continue
            lfound[irk] = True
            Dl = d_left[:, :, isym, ir]
            Dw = d_wann[:, :, isym, ir]
            G_irr[ir] = G_irr[ir] + Dl.conj().transpose(-1, -2) @ G_full[irk] @ Dw

    for ir in range(nkptirr):
        ik = ir2ik[ir]
        stab = [isym for isym in range(1, nsym) if kptsym[isym, ir] == ik]
        if not stab:
            continue
        g = G_irr[ir]
        total = g.clone()
        for isym in stab:
            Dl = d_left[:, :, isym, ir]
            Dw = d_wann[:, :, isym, ir]
            total = total + Dl.conj().transpose(-1, -2) @ g @ Dw
        G_irr[ir] = total / (len(stab) + 1)

    return G_irr
