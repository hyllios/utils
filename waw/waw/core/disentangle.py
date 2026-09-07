"""
Subspace disentanglement for entangled bands.

Implements the iterative subspace selection algorithm from Souza, Marzari &
Vanderbilt (PRB 65, 035109, 2001).  When there are more bands than Wannier
functions (nb > nw), we must first select an optimal nw-dimensional subspace
at each k-point before running the standard spread minimization.

The objective is to minimize Omega_I over all semi-unitary V(k):

    Omega_I = (1/Nk) sum_{k,b} w_b [ Nw - ||V(k)† M(k,b) V(k+b)||_F^2 ]

subject to:
  * V(k)† V(k) = I_{nw}             (semi-unitarity)
  * V(k) spans the outer window      (only bands with eig in [E_out_min, E_out_max])
  * V(k) contains the frozen window  (bands with eig in [E_froz_min, E_froz_max]
                                       are included at every k-point)

Algorithm (coordinate descent over k-points):
  At each step and each k, fix V(k') for k' ≠ k and update V(k) by:
    1. Compute Z(k) = sum_b w_b M(k,b) P(k+b) M†(k,b)
       where P(k) = V(k) V(k)† is the current projection.
    2. Project Z(k) onto the free-outer subspace (outer ∩ non-frozen bands).
    3. Take the top (nw - n_frozen(k)) eigenvectors of the projected Z as the
       new free columns of V(k).
    4. Keep the n_frozen(k) frozen columns fixed.
  Repeat until Omega_I converges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .spread import rotate_overlaps
from .sitesym import SiteSymmetry, broadcast_matrix, symmetrize_zmatrix, extract_symmetrized_subspace
from .optim import _riemannian_gradient, _qr_retract, _real_inner


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DisentangleResult:
    """Output of the disentanglement step."""
    V:         Tensor              # (nk, nb, nw)  semi-unitary mixing matrix
    omega_i:   float               # final Omega_I in Bohr^2
    lwindow:   Tensor = None       # (nk, nb) bool  band-in-outer-window mask
    ndimwin:   Tensor = None       # (nk,) long     bands in outer window per k
    history:   list[float] = field(default_factory=list)
    converged: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _omega_i(V: Tensor, Mmn: Tensor, wb: Tensor, kb_idx: Tensor) -> float:
    """
    Omega_I of the current subspace selection V.

    Omega_I = (1/Nk) sum_{k,b} w_b [Nw - ||M_tilde^{k,b}||_F^2], using the
    full Frobenius norm (all n,m pairs), not just the diagonal: ||M_tilde||_F^2
    = Tr(P_k P_{k+b}) is invariant under any unitary gauge rotation
    V(k) -> V(k)W(k), unlike the diagonal-only sum, and matches what the
    Z-matrix iteration below actually minimizes.
    """
    nk = V.shape[0]
    nw = V.shape[2]
    M_sub = rotate_overlaps(V, Mmn, kb_idx)   # (nk, nnb, nw, nw)
    frob2 = M_sub.abs().pow(2).sum(dim=(-1, -2))   # (nk, nnb)
    return torch.einsum("b,kb->", wb, nw - frob2).item() / nk


def _build_Z(V: Tensor, Mmn: Tensor, wb: Tensor, kb_idx: Tensor,
             half_shell: bool = False) -> Tensor:
    """
    Compute Z(k) = sum_b w_b M(k,b) P(k+b) M†(k,b) for all k.

    Returns Z: (nk, nb, nb) Hermitian matrices.
    P(k+b) = V(k+b) @ V†(k+b) is the projection onto the current subspace.

    half_shell: set for Gamma-only data whose .nnkp stores only one of each
    +-b pair (wannier90's "b-vectors reduced by half" trick, wb doubled by
    the shell-weight solve). The MV spread functional itself is exact under
    that weight-doubling (every term is +-b symmetric via M(k,-b) = M(k,b)†
    at Gamma), but this Z map is not: the missing -b neighbour contributes
    M† P M, which differs from M P M†. Without the completion, the
    iteration converges to points that are not stationary points of
    Omega_I. The 1/2 compensates the doubled wb.
    """
    # P at each k+b: (nk, nnb, nb, nb)
    V_kb  = V[kb_idx]                                          # (nk, nnb, nb, nw)
    P_kb  = torch.matmul(V_kb, V_kb.conj().transpose(-1, -2))  # (nk, nnb, nb, nb)

    # Z_kb = M @ P_kb @ M†: (nk, nnb, nb, nb)
    MP    = torch.matmul(Mmn, P_kb)                            # (nk, nnb, nb, nb)
    Z_kb  = torch.matmul(MP, Mmn.conj().transpose(-1, -2))    # (nk, nnb, nb, nb)

    if half_shell:
        # Add the missing -b partners: M(k,-b) P(k-b) M(k,-b)† = M† P M
        # (Gamma-only => nk == 1, so P(k-b) is the same projector as P(k+b)).
        Mh    = Mmn.conj().transpose(-1, -2)
        Z_rev = torch.matmul(torch.matmul(Mh, P_kb), Mmn)
        Z_kb  = 0.5 * (Z_kb + Z_rev)

    # Z[k] = sum_b w_b Z_kb[k,b]
    return (wb.to(Z_kb.dtype)[None, :, None, None] * Z_kb).sum(dim=1)  # (nk, nb, nb)


# ---------------------------------------------------------------------------
# V initialization helpers
# ---------------------------------------------------------------------------

def _init_V_index(
    outer_mask:  torch.Tensor,
    frozen_mask: torch.Tensor,
    nk:          int,
    nb:          int,
    nw:          int,
    dtype,
) -> torch.Tensor:
    """
    Baseline initialization: frozen bands first, then the first nw-nf
    free outer-window bands in index order.
    """
    V = torch.zeros(nk, nb, nw, dtype=dtype)
    for ik in range(nk):
        froz_idx     = frozen_mask[ik].nonzero(as_tuple=True)[0]
        free_out_idx = (outer_mask[ik] & ~frozen_mask[ik]).nonzero(as_tuple=True)[0]
        nf = len(froz_idx)
        for j, idx in enumerate(froz_idx):
            V[ik, idx, j] = 1.0
        for j, idx in enumerate(free_out_idx[: nw - nf]):
            V[ik, idx, nf + j] = 1.0
    return V


def _init_V_amn(
    Amn:         Tensor,
    outer_mask:  torch.Tensor,
    frozen_mask: torch.Tensor,
    nk:          int,
    nb:          int,
    nw:          int,
) -> torch.Tensor:
    """
    Projection-based initialization from the trial orbitals Amn, following
    wannier90's `dis_project` + `dis_proj_froz`.

    At each k-point:
      * Loewdin/polar projection over the full outer window (frozen rows
        included): SVD A_win = P S Q^dag of Amn restricted to window bands,
        U_loewdin = P Q^dag -- the closest set of nw orthonormal vectors to
        the nw projected trial orbitals.
      * Frozen columns -- standard band unit vectors (band fully included).
      * Free columns -- top (nw - nf) eigenvectors of the trial-subspace
        projector P_s = U_loewdin U_loewdin^dag restricted to the free
        (non-frozen) window bands. Equivalent to W90's QPQ construction
        (Q_froz P_s Q_froz), since QPQ's frozen rows/columns are
        identically zero and its nonzero-eigenvalue eigenvectors are
        exactly the eigenvectors of the free-restricted block.

    The frozen-window treatment differs from a naive "SVD of the
    frozen-excluded rows of Amn": trial orbitals whose weight sits mostly
    on frozen bands still contribute unit weight to P_s after Loewdin
    normalization, so their non-frozen tails steer the free-column choice.
    On hard metallic cases the two inits can land in different
    self-consistent basins of the Z-matrix iteration.

    Without frozen bands the two schemes give the same initial subspace
    (P Q^dag spans the same space as the top-nw left singular vectors).

    Because frozen and free outer-window bands are disjoint index sets,
    the resulting V satisfies V†V = I_nw exactly by construction.
    """
    V = torch.zeros(nk, nb, nw, dtype=Amn.dtype)
    for ik in range(nk):
        win_idx      = outer_mask[ik].nonzero(as_tuple=True)[0]
        froz_idx     = frozen_mask[ik].nonzero(as_tuple=True)[0]
        free_out_idx = (outer_mask[ik] & ~frozen_mask[ik]).nonzero(as_tuple=True)[0]
        nf  = len(froz_idx)
        nfr = nw - nf

        # Frozen columns: standard band unit vectors
        for j, idx in enumerate(froz_idx):
            V[ik, idx, j] = 1.0

        if nfr == 0:
            continue

        A_win = Amn[ik, win_idx, :]   # (ndimwin, nw_trials)

        if A_win.numel() == 0 or A_win.shape[1] == 0:
            for j, idx in enumerate(free_out_idx[:nfr]):
                V[ik, idx, nf + j] = 1.0
            continue

        try:
            # Loewdin/polar projection over the full window (dis_project)
            P, _, Qh = torch.linalg.svd(A_win, full_matrices=False)
            U_loewdin = P @ Qh                       # (ndimwin, nw_trials)

            # Trial-subspace projector restricted to free bands (dis_proj_froz)
            free_in_win = (~frozen_mask[ik][win_idx]).nonzero(as_tuple=True)[0]
            P_s_free = U_loewdin[free_in_win] @ U_loewdin[free_in_win].conj().T
            eigvals, eigvecs = torch.linalg.eigh(P_s_free)   # ascending
            n_avail = eigvecs.shape[1]
            n_take  = min(nfr, n_avail)
            V[ik][free_out_idx, nf : nf + n_take] = eigvecs[:, -n_take:].flip(-1)
            # If fewer eigenvectors than needed, fill remainder by index
            for j in range(n_take, nfr):
                if j < len(free_out_idx):
                    V[ik, free_out_idx[j], nf + j] = 1.0
        except Exception:
            for j, idx in enumerate(free_out_idx[:nfr]):
                V[ik, idx, nf + j] = 1.0

    return V


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def _check_frozen_is_symmetry_closed(frozen_mask, outer_mask, sitesym, tol=1e-6):
    """Preconditions for combining a frozen window with site symmetry.

    Two of them, and both are properties of the DATA rather than of the code:

    1. Every k in a star must freeze the same number of bands. Since
       eps(Sk, n) = eps(k, n) this is automatic for an energy window, and it
       fails only if the window edge sits within numerical noise of a band, so a
       violation means the window should be moved, not that the code is wrong.

    2. The frozen subspace must be INVARIANT under the band representation of
       EVERY symmetry element, not merely the stabilizer's. The stabilizer is
       what makes restricting Z to the free block meaningful; the rest of the
       group is what makes `broadcast_matrix` preserve the constraint, since
       span V(Rk) = d_band(R) span V(k) contains the frozen space only if
       d_band(R) maps it to itself. Checking only the stabilizer lets the frozen
       band be lost at image k-points -- measured, not hypothetical.

       Physically this holds because eps(Sk, n) = eps(k, n) forces d_band to be
       block diagonal over degenerate groups, so an energy window can only fail
       it by cutting through a multiplet.
    """
    nfr = frozen_mask.sum(dim=1)
    for ir in range(sitesym.nkptirr):
        ik = int(sitesym.ir2ik[ir])
        star = [int(sitesym.kptsym[isym, ir]) for isym in range(sitesym.nsymmetry)]
        bad = sorted({int(nfr[j]) for j in star})
        if len(bad) > 1:
            raise ValueError(
                f"sitesym + frozen bands: the star of irreducible k-point {ik} "
                f"freezes different numbers of bands at different members "
                f"({bad}). Degenerate partners are being split by the frozen "
                f"window edge; move it into a gap.")
        froz = frozen_mask[ik]
        free = outer_mask[ik] & ~frozen_mask[ik]
        if not (froz.any() and free.any()):
            continue
        fi = froz.nonzero(as_tuple=True)[0]
        ei = free.nonzero(as_tuple=True)[0]
        for isym in range(sitesym.nsymmetry):
            D = sitesym.d_matrix_band[:, :, isym, ir]
            leak = max(D[fi][:, ei].abs().max().item(),
                       D[ei][:, fi].abs().max().item())
            if leak > tol:
                jk = int(sitesym.kptsym[isym, ir])
                where = ("the stabilizer element" if jk == ik
                         else f"the element mapping it to k-point {jk}")
                raise ValueError(
                    f"sitesym + frozen bands: at k-point {ik} {where} {isym} "
                    f"mixes frozen with non-frozen bands (|d_band| leak "
                    f"{leak:.2e} > {tol:g}), so the frozen subspace is not "
                    f"symmetry invariant. Restricting Z to the free block, and "
                    f"broadcasting the result over the star, both stop making "
                    f"sense. The frozen window is cutting a degenerate "
                    f"multiplet; move its edge into a gap.")


def _check_subspace_cut_is_clean(evals, nfr, ik, tol=1e-8):
    """The top-nfr eigenspace of a symmetrized Z is stabilizer invariant only if
    the cut does not fall inside a degenerate eigenvalue. Without a gap, `eigh`
    picks an arbitrary basis of the multiplet and the selected subspace is not
    equivariant -- silently breaking the symmetry the run asked for."""
    if nfr >= len(evals):
        return
    gap = float(evals[-nfr] - evals[-nfr - 1])
    if gap < tol:
        raise ValueError(
            f"sitesym + frozen bands: at k-point {ik} the Z-matrix eigenvalue "
            f"cut selecting {nfr} free states falls inside a degenerate "
            f"multiplet (gap {gap:.2e} < {tol:g}). The chosen subspace would "
            f"not be symmetry equivariant. Change nw, or the windows, so the "
            f"cut lands in a gap.")


def disentangle(
    Mmn:            Tensor,
    eig:            Tensor,
    wb:             Tensor,
    kb_idx:         Tensor,
    nw:             int,
    Amn:            Tensor | None              = None,
    outer_window:   tuple[float, float] | None = None,
    frozen_window:  tuple[float, float] | None = None,
    proj_min:       float | None = None,
    proj_max:       float | None = None,
    n_iter:         int   = 200,
    conv_tol:       float = 1e-10,
    conv_window:    int   = 3,
    mix_ratio:      float = 0.5,
    bvecs:          Tensor | None = None,
    kpts:           Tensor | None = None,
    real_lattice:   np.ndarray | None = None,
    dis_spheres:    list[tuple[float, float, float, float]] | None = None,
    dis_spheres_first_wann: int = 0,
    sitesym:        SiteSymmetry | None = None,
) -> DisentangleResult:
    """
    Iterative subspace disentanglement (SMV01).

    For isolated bands (nb == nw) this is a no-op: V = I is returned immediately.

    Args:
      Mmn          : (nk, nnb, nb, nb) complex  raw overlaps from DFT
      eig          : (nk, nb)          real     band eigenvalues (Hartree)
      wb           : (nnb,)            real     shell weights
      kb_idx       : (nk, nnb)         long     neighbour k-index table
      nw           : int               number of Wannier functions
      Amn          : (nk, nb, nw) complex, optional.
                     Trial projection matrix from the .amn file.  When
                     provided, the initial V(k) is set to the top
                     (nw - n_frozen) left singular vectors of
                     Amn[k, free_outer_bands, :], aligning the starting
                     subspace with the trial WF projections.  When None
                     (default), the first nw free outer-window bands are
                     used (original index-based fallback).
      outer_window : (E_min, E_max) in Hartree.  Only bands inside are used.
                     None means all bands.
      frozen_window: (E_min, E_max) in Hartree.  Bands inside are frozen
                     (always included in the subspace).  None means no
                     frozen bands.
      proj_min, proj_max : wannier90's `dis_proj_min`/`dis_proj_max`
                     (projectability-based disentanglement).
                     projs[k,i] = sum_j |Amn[k,i,j]|^2 (requires Amn).
                     Bands with projs >= proj_max are frozen (in addition
                     to frozen_window); bands with projs < proj_min are
                     dropped from the outer window entirely (unless
                     already frozen). Bands with proj_min <= projs <
                     proj_max remain ordinary disentanglement candidates.
                     Combines with (does not replace) the energy windows;
                     None (default) disables projectability filtering.
      sitesym      : core.sitesym.SiteSymmetry, optional (`site_symmetry =
                     .true.`). When given, the Z-matrix eigenvector
                     extraction runs only over the irreducible k-wedge
                     (`sitesym.ir2ik`) and is symmetrized under the
                     crystal's point-group action at every sweep
                     (`core.sitesym.symmetrize_zmatrix`/
                     `extract_symmetrized_subspace`), then broadcast to the
                     rest of the mesh -- every other k-point's V is derived,
                     not independently optimized. Frozen bands are not yet
                     supported together with sitesym; when they are, the
                     symmetrization is done on the SUBSPACE (d_band only)
                     rather than on the frame, because Omega_I is
                     gauge-invariant and d_matrix_wann cannot act on a subset
                     of the nw columns. Two data preconditions are checked
                     (`_check_frozen_is_symmetry_closed`,
                     `_check_subspace_cut_is_clean`); the Wannier-side gauge is
                     then fixed by `optim.minimize_spread_symmetrized`.
                     None (default) disables this.
      n_iter       : maximum number of sweeps through all k-points
      conv_tol     : convergence threshold on |Omega_I[t] - Omega_I[t-conv_window]|
      conv_window  : window size for convergence check
      mix_ratio    : damping of the Z-matrix fixed-point iteration
                     (wannier90's `dis_mix_ratio`, same default 0.5):
                     Z_in(n) = mix*Z(V_{n-1}) + (1-mix)*Z_in(n-1), with
                     Z_in(1) = Z(V_0) undamped. 1.0 recovers the plain
                     undamped iteration, which can converge to a worse
                     Omega_I fixed point on hard cases.
      bvecs        : (nk, nnb, 3) Cartesian b-vectors, optional. Used only
                     to auto-detect Gamma-only half-shell data (a b-vector
                     set not symmetric under negation), which needs the
                     completed Z map -- see ``_build_Z``. None assumes
                     ordinary full-shell data.
      kpts, real_lattice, dis_spheres, dis_spheres_first_wann :
                     wannier90's `dis_spheres`: k-space-localized
                     disentanglement, for manifolds (e.g. correlated t2g/eg
                     orbitals) that are cleanly isolated almost everywhere
                     but hybridize with other bands only near specific
                     k-points. `dis_spheres` is a list of (kx, ky, kz,
                     radius) tuples, fractional centre + Cartesian-
                     reciprocal radius (Bohr^-1). At k-points inside at
                     least one sphere, disentanglement proceeds as usual
                     (energy windows / proj_min/proj_max, if given). At
                     k-points outside every sphere, disentanglement is
                     skipped entirely: exactly `nw` bands starting at
                     `dis_spheres_first_wann` (0-based) are taken directly,
                     overriding any outer/frozen/projectability mask at
                     that k-point. `kpts` (nk, 3) fractional and
                     `real_lattice` (3, 3) Bohr are required together with
                     `dis_spheres`; None (default) disables this and uses
                     ordinary disentanglement everywhere.

    Returns:
      DisentangleResult with V (nk, nb, nw), omega_i, history, converged flag.
    """
    nk, nnb, nb, _ = Mmn.shape

    # ---- No-op for isolated bands ----------------------------------------
    if nb == nw:
        V = torch.eye(nb, dtype=Mmn.dtype).unsqueeze(0).expand(nk, -1, -1).clone()
        oi = _omega_i(V, Mmn, wb, kb_idx)
        lwindow = torch.ones(nk, nb, dtype=torch.bool)
        ndimwin = torch.full((nk,), nb, dtype=torch.long)
        return DisentangleResult(
            V=V, omega_i=oi, lwindow=lwindow, ndimwin=ndimwin,
            history=[oi], converged=True,
        )

    # ---- Build band-selection masks --------------------------------------
    if outer_window is not None:
        eout_min, eout_max = outer_window
        outer_mask = (eig >= eout_min) & (eig <= eout_max)   # (nk, nb) bool
    else:
        outer_mask = torch.ones(nk, nb, dtype=torch.bool)

    if frozen_window is not None:
        efroz_min, efroz_max = frozen_window
        frozen_mask = (eig >= efroz_min) & (eig <= efroz_max) & outer_mask
    else:
        frozen_mask = torch.zeros(nk, nb, dtype=torch.bool)

    # ---- Projectability disentanglement (Wannier90 dis_windows_proj) -----
    if proj_min is not None or proj_max is not None:
        if Amn is None:
            raise ValueError(
                "proj_min/proj_max (projectability disentanglement) require "
                "Amn -- projectability is computed from the trial-projection "
                "matrix, projs[k,i] = sum_j |Amn[k,i,j]|^2."
            )
        projs = Amn.abs().pow(2).sum(dim=-1)   # (nk, nb)
        if proj_max is not None:
            frozen_mask = frozen_mask | ((projs >= proj_max) & outer_mask)
        if proj_min is not None:
            outer_mask = outer_mask & ((projs >= proj_min) | frozen_mask)

    # ---- dis_spheres: k-space-localized disentanglement (Wannier90 tutorial20) ---
    if dis_spheres is not None:
        if kpts is None or real_lattice is None:
            raise ValueError("dis_spheres requires kpts and real_lattice.")
        recip_lattice = 2.0 * np.pi * np.linalg.inv(np.asarray(real_lattice)).T
        k_np = kpts.detach().cpu().numpy() if isinstance(kpts, torch.Tensor) else np.asarray(kpts)

        in_sphere = np.zeros(nk, dtype=bool)
        for kx, ky, kz, radius in dis_spheres:
            center = np.array([kx, ky, kz])
            dk_frac = k_np - center[None, :]
            dk_frac = dk_frac - np.rint(dk_frac)          # nearest periodic image
            dk_cart = dk_frac @ recip_lattice              # (nk, 3) Bohr^-1
            in_sphere |= (dk_cart ** 2).sum(axis=1) < radius ** 2

        outside = torch.from_numpy(~in_sphere)
        if outside.any():
            first = dis_spheres_first_wann
            fixed_mask = torch.zeros(nb, dtype=torch.bool)
            fixed_mask[first:first + nw] = True
            outer_mask[outside] = fixed_mask
            frozen_mask[outside] = fixed_mask

    n_outer = outer_mask.sum(dim=1)   # (nk,)
    n_froz  = frozen_mask.sum(dim=1)  # (nk,)

    if not (n_outer >= nw).all():
        raise ValueError(
            f"Outer window too narrow: some k-points have fewer than nw={nw} "
            f"bands inside.  Minimum is {n_outer.min().item()}."
        )
    if not (n_froz <= nw).all():
        raise ValueError(
            f"Frozen window too wide: some k-points have more than nw={nw} "
            f"frozen bands.  Maximum is {n_froz.max().item()}."
        )
    # Zero-freedom trap (silent until 2026-07-28, and it cost trigonal Te four
    # sessions): if the frozen window already contains nw bands at (nearly)
    # every k, disentanglement has no freedom anywhere -- the model is simply
    # "the lowest nw bands of the window", whatever they are. On Te the outer
    # window included the 5s manifold, so frozen s + p-valence == nw == 18 at
    # EVERY k and the returned "p model" never contained a single p-conduction
    # band. Everything downstream (SHC above E_F, SAC quadrupole sums) was
    # built on the wrong subspace and no error was raised. A plot makes this
    # obvious -- see `waw.vis.plot_wannierization_windows` -- but plots only
    # help when someone looks, so say it here too.
    frac_full = float((n_froz == nw).sum()) / len(n_froz)
    if frac_full > 0.9:
        import warnings
        warnings.warn(
            f"disentangle: the frozen window contains exactly nw={nw} bands at "
            f"{100 * frac_full:.0f}% of k-points -- disentanglement has (almost) "
            "no freedom, and the model is just the lowest-nw-band slice of the "
            "outer window. If a detached manifold (semicore s, ...) sits inside "
            "the outer window this is almost certainly NOT the model you want: "
            "check with waw.vis.plot_wannierization_windows.",
            RuntimeWarning, stacklevel=3,
        )
    sitesym_frozen = sitesym is not None and bool((n_froz > 0).any())
    if sitesym_frozen:
        _check_frozen_is_symmetry_closed(frozen_mask, outer_mask, sitesym)

    # ---- Initialize V ----------------------------------------------------
    if Amn is not None:
        V = _init_V_amn(Amn, outer_mask, frozen_mask, nk, nb, nw)
    else:
        V = _init_V_index(outer_mask, frozen_mask, nk, nb, nw, Mmn.dtype)

    # ---- Half-shell detection (Gamma-only .nnkp; see _build_Z) -----------
    half_shell = False
    if bvecs is not None:
        b0 = bvecs[0]                                    # (nnb, 3)
        dists = (b0[:, None, :] + b0[None, :, :]).norm(dim=-1)   # |b_i + b_j|
        has_partner = (dists < 1e-8).any(dim=1)
        half_shell = not bool(has_partner.all())

    # ---- Main iteration -------------------------------------------------
    history = []
    Z_in = None   # damped Z-matrix state (Wannier90's czmat_in)

    # Fast path: no frozen bands and no outer-window exclusion (e.g. pure-SCDM
    # runs with outer_window=frozen_window=None) means free_out_idx is the
    # same arange(nb) at every k, so the per-k eigh loop below collapses into
    # one batched torch.linalg.eigh over the whole (nk, nb, nb) Z at once --
    # avoids nk sequential small (<=nb x nb) eigendecompositions, each too
    # small for BLAS threading to help, dominated by Python/dispatch overhead.
    uniform_free = sitesym is None and not bool(frozen_mask.any()) and bool(outer_mask.all())

    for sweep in range(n_iter):
        history.append(_omega_i(V, Mmn, wb, kb_idx))

        Z_out = _build_Z(V, Mmn, wb, kb_idx, half_shell)   # (nk, nb, nb) — vectorised
        if Z_in is None or mix_ratio >= 1.0:
            Z_in = Z_out
        else:
            Z_in = mix_ratio * Z_out + (1.0 - mix_ratio) * Z_in
        Z = Z_in

        if sitesym is not None:
            # Symmetrize Z over the whole mesh (needs every orbit member),
            # then extract each irreducible k's subspace from its own
            # symmetrized slice via the band-by-band steepest-ascent
            # iteration (wannier90's `sitesym_dis_extract_symmetry`),
            # restricted to that k's outer-window bands -- see
            # core.sitesym.extract_symmetrized_subspace. Warm-started from
            # this k's own previous-sweep V, not recomputed from scratch.
            Z_irr = symmetrize_zmatrix(Z, sitesym)   # (nkptirr, nb, nb)
            for ir in range(sitesym.nkptirr):
                ik = int(sitesym.ir2ik[ir])
                if sitesym_frozen:
                    # WITH frozen bands the free columns are a SUBSET of the nw,
                    # and `extract_symmetrized_subspace` cannot act on a subset:
                    # it rotates columns by d_matrix_wann, which is nw x nw. The
                    # way out is that Omega_I does not depend on the gauge at
                    # all, only on span(V(k)) -- and a SUBSPACE transforms under
                    # d_band alone, d_wann cancelling in V V^dag. So symmetrize
                    # the subspace, not the frame: Z_free is already stabilizer
                    # invariant, hence so are its eigenspaces, and a plain eigh
                    # gives an invariant top-nfr subspace provided the cut does
                    # not fall inside a degenerate multiplet (checked). d_wann
                    # belongs to the gauge stage (`minimize_spread_symmetrized`),
                    # which is where nw x nw is the right size.
                    froz_idx = frozen_mask[ik].nonzero(as_tuple=True)[0]
                    free_out_idx = (outer_mask[ik]
                                    & ~frozen_mask[ik]).nonzero(as_tuple=True)[0]
                    nf = len(froz_idx)
                    nfr = nw - nf
                    if nfr > 0:
                        Z_free = Z_irr[ir][free_out_idx][:, free_out_idx]
                        evals, evecs = torch.linalg.eigh(Z_free)
                        _check_subspace_cut_is_clean(evals, nfr, ik)
                        V[ik, :, nf:] = 0.0
                        V[ik][free_out_idx, nf:] = evecs[:, -nfr:]
                else:
                    free_out_idx = outer_mask[ik].nonzero(as_tuple=True)[0]
                    Z_free = Z_irr[ir][free_out_idx][:, free_out_idx]
                    d_free = sitesym.d_matrix_band[free_out_idx][:, free_out_idx]
                    U_init = V[ik][free_out_idx, :]

                    v_free = extract_symmetrized_subspace(
                        Z_free, sitesym, nw, ir, U_init=U_init, d_left=d_free,
                    )

                    V[ik, :, :] = 0.0
                    V[ik][free_out_idx, :] = v_free

            V_irr = torch.stack([V[int(sitesym.ir2ik[ir])]
                                 for ir in range(sitesym.nkptirr)])
            # Same reason: with frozen bands only the SUBSPACE is being
            # symmetrized, so the images are generated with d_wann = I. That
            # leaves the frame at the image k-points in whatever gauge d_band
            # induces -- irrelevant to Omega_I, and fixed afterwards by the
            # symmetrized spread minimization.
            d_wann = (torch.eye(nw, dtype=sitesym.d_matrix_wann.dtype)
                      .reshape(nw, nw, 1, 1)
                      .expand_as(sitesym.d_matrix_wann).contiguous()
                      if sitesym_frozen else sitesym.d_matrix_wann)
            V = broadcast_matrix(
                V_irr, sitesym.d_matrix_band, d_wann,
                sitesym.kptsym, sitesym.ir2ik, nk,
            )
        elif uniform_free:
            # Batched over all k at once: free_out_idx == arange(nb) everywhere.
            _, eigvecs = torch.linalg.eigh(Z)          # (nk, nb, nb), ascending
            V = eigvecs[:, :, -nw:].clone()             # (nk, nb, nw)
        else:
            for ik in range(nk):
                froz_idx     = frozen_mask[ik].nonzero(as_tuple=True)[0]
                free_out_idx = (outer_mask[ik] & ~frozen_mask[ik]).nonzero(as_tuple=True)[0]
                nf   = len(froz_idx)
                nfr  = nw - nf   # free columns to determine

                if nfr == 0:
                    continue   # all columns frozen; nothing to optimise

                # Extract Z restricted to the free-outer subspace
                Z_free = Z[ik][free_out_idx][:, free_out_idx]   # (n_free_out, n_free_out)

                # Top nfr eigenvectors (eigh returns ascending order)
                _, eigvecs = torch.linalg.eigh(Z_free)           # (n_free_out, n_free_out)
                v_free = eigvecs[:, -nfr:]                        # (n_free_out, nfr)

                # Write back into V[ik]: zero out free columns then fill
                V[ik, :, nf:] = 0.0
                V[ik][free_out_idx, nf:] = v_free

        # Convergence check
        if len(history) >= conv_window:
            delta = abs(history[-conv_window] - history[-1])
            if delta < conv_tol:
                oi = _omega_i(V, Mmn, wb, kb_idx)
                return DisentangleResult(
                    V=V, omega_i=oi, lwindow=outer_mask, ndimwin=n_outer,
                    history=history, converged=True,
                )

    oi = _omega_i(V, Mmn, wb, kb_idx)
    return DisentangleResult(
        V=V, omega_i=oi, lwindow=outer_mask, ndimwin=n_outer,
        history=history, converged=False,
    )


# ---------------------------------------------------------------------------
# Joint (whole-mesh) Riemannian CG disentanglement
#
# Alternative to the per-k Z-matrix coordinate descent above: each sweep
# there, V(k)'s subspace is picked from an independent top-eigenvector
# problem, with no mechanism preventing two neighbouring k from settling on
# different, both locally self-consistent, branches when the relevant
# eigenvalues are nearly degenerate. This minimizes the SAME Omega_I jointly
# over the whole V tensor by Riemannian nonlinear CG instead, removing that
# per-k independence without changing what is being minimized.
#
# Reuses `core.optim`'s manifold-geometry primitives (`_riemannian_gradient`,
# `_qr_retract`), built for the spread-minimization stage's U(k) in U(nw) --
# both formulas are dimension-agnostic and apply identically to V's
# rectangular Stiefel manifold St(nb, nw).
# ---------------------------------------------------------------------------

def _omega_i_diff(V: Tensor, Mmn: Tensor, wb: Tensor, kb_idx: Tensor) -> Tensor:
    """
    Differentiable Omega_I(V): same formula as `_omega_i` above, without the
    final ``.item()`` that detaches it from the autograd graph.
    """
    nk = V.shape[0]
    nw = V.shape[2]
    M_sub = rotate_overlaps(V, Mmn, kb_idx)                     # (nk, nnb, nw, nw)
    frob2 = M_sub.abs().pow(2).sum(dim=(-1, -2))                # (nk, nnb)
    return torch.einsum("b,kb->", wb.to(frob2.dtype), nw - frob2) / nk


def _omega_i_and_grad(V: Tensor, Mmn: Tensor, wb: Tensor, kb_idx: Tensor) -> tuple[float, Tensor]:
    V = V.detach().requires_grad_(True)
    omega = _omega_i_diff(V, Mmn, wb, kb_idx)
    omega.backward()
    return omega.item(), V.grad.detach()


def _gram_schmidt_masked(free_only: Tensor, free_col_mask: Tensor) -> Tensor:
    """
    Orthonormalize the free columns of `free_only` (nk, nb, nw) among
    themselves, in column order, via explicit modified Gram-Schmidt --
    NOT `torch.linalg.qr`, which was tried first and rejected: a frozen
    slot's column is exactly zero (masked out), and Householder QR is then
    free to fill that degenerate column with an ARBITRARY completion vector
    to finish the orthonormal basis (verified empirically: it picked a
    vector at whatever row is first in the matrix, e.g. `e_0`) -- that
    filler vector still has to be orthogonal to every other column QR
    produces, which means it silently steals a genuine degree of freedom
    from the TRUE free columns even though the degenerate column itself is
    later discarded. A masked column that is skipped (left exactly zero,
    contributing nothing to the projection of later columns) rather than
    arbitrarily completed avoids this failure mode entirely.

    `free_col_mask` (nk, nw) marks which columns are real (True) vs a
    masked-out frozen slot (False, left exactly zero, never normalized).
    """
    nw = free_only.shape[-1]
    cols = []
    for j in range(nw):
        v = free_only[..., j]
        for c in cols:
            proj = (c.conj() * v).sum(dim=-1, keepdim=True)     # (nk, 1)
            v = v - proj * c
        norm = v.norm(dim=-1, keepdim=True)                      # (nk, 1)
        safe_norm = torch.where(norm > 1e-300, norm, torch.ones_like(norm))
        is_active = free_col_mask[:, j:j + 1]                    # (nk, 1)
        v = torch.where(is_active, v / safe_norm, torch.zeros_like(v))
        cols.append(v)
    return torch.stack(cols, dim=-1)


def _retract_disentangle(
    V: Tensor, step: Tensor,
    frozen_block: Tensor, free_col_mask: Tensor, outer_row_mask: Tensor,
) -> Tensor:
    """
    QR-retract V+step, then explicitly restore the two constraints a plain
    `_qr_retract` is not guaranteed to preserve exactly:

      * Frozen columns must stay EXACTLY their fixed unit vectors, not just
        close -- Householder QR (`torch.linalg.qr`'s algorithm) does not in
        general reproduce an already-unit-norm input column unchanged in its
        output. Done by hand instead: hard-reset the frozen columns, project
        their component back out of the free columns, then re-orthonormalize
        the free block among itself via `_gram_schmidt_masked` (see its
        docstring for why plain `torch.linalg.qr` is unsafe here).
      * Outer-window row exclusion IS already preserved automatically through
        any QR step (a linear combination of columns that are all exactly
        zero at an excluded row is itself exactly zero there), so the final
        multiply-by-mask is a cheap belt-and-braces safety net, not
        load-bearing.
    """
    V_trial = _qr_retract(V, step)

    frozen_col = (~free_col_mask).unsqueeze(-2)                 # (nk, 1, nw) bool
    V_trial = torch.where(frozen_col.expand_as(V_trial), frozen_block, V_trial)

    free_only = V_trial * free_col_mask.unsqueeze(-2)
    overlap = torch.matmul(frozen_block.conj().transpose(-1, -2), free_only)   # (nk, nw, nw)
    free_only = free_only - torch.matmul(frozen_block, overlap)

    Q = _gram_schmidt_masked(free_only, free_col_mask)

    V_out = torch.where(frozen_col.expand_as(V_trial), frozen_block, Q)
    V_out = V_out * outer_row_mask.unsqueeze(-1).to(V_out.dtype)
    return V_out


def disentangle_joint(
    Mmn:            Tensor,
    eig:            Tensor,
    wb:             Tensor,
    kb_idx:         Tensor,
    nw:             int,
    Amn:            Tensor | None              = None,
    outer_window:   tuple[float, float] | None = None,
    frozen_window:  tuple[float, float] | None = None,
    n_iter:         int   = 200,
    conv_tol:       float = 1e-10,
    conv_window:    int   = 3,
    lr:             float = 1.0,
    num_cg_steps:   int   = 5,
    cg_max_ratio:   float = 3.0,
) -> DisentangleResult:
    """
    Joint Riemannian CG minimization of Omega_I over the whole `V` at once --
    see the module-level comment above `_omega_i_diff` for the motivation.

    Does NOT support `sitesym`, `dis_spheres`, or `proj_min`/`proj_max`: a
    deliberately scoped-down first version. `disentangle()` remains the only
    path for those.

    Args: as `disentangle()`'s correspondingly-named parameters (same units:
    `eig`/`outer_window`/`frozen_window` in Hartree). `lr`, `num_cg_steps`,
    `cg_max_ratio` are the CG line-search/reset hyperparameters -- same
    defaults and meaning as `core.optim._minimize_cg`.

    Returns:
      DisentangleResult with V (nk, nb, nw), omega_i, history, converged flag.
    """
    nk, nnb, nb, _ = Mmn.shape

    if nb == nw:
        V = torch.eye(nb, dtype=Mmn.dtype).unsqueeze(0).expand(nk, -1, -1).clone()
        oi = _omega_i(V, Mmn, wb, kb_idx)
        lwindow = torch.ones(nk, nb, dtype=torch.bool)
        ndimwin = torch.full((nk,), nb, dtype=torch.long)
        return DisentangleResult(
            V=V, omega_i=oi, lwindow=lwindow, ndimwin=ndimwin,
            history=[oi], converged=True,
        )

    if outer_window is not None:
        eout_min, eout_max = outer_window
        outer_mask = (eig >= eout_min) & (eig <= eout_max)
    else:
        outer_mask = torch.ones(nk, nb, dtype=torch.bool)

    if frozen_window is not None:
        efroz_min, efroz_max = frozen_window
        frozen_mask = (eig >= efroz_min) & (eig <= efroz_max) & outer_mask
    else:
        frozen_mask = torch.zeros(nk, nb, dtype=torch.bool)

    n_outer = outer_mask.sum(dim=1)
    n_froz  = frozen_mask.sum(dim=1)
    if not (n_outer >= nw).all():
        raise ValueError(
            f"Outer window too narrow: some k-points have fewer than nw={nw} "
            f"bands inside.  Minimum is {n_outer.min().item()}."
        )
    if not (n_froz <= nw).all():
        raise ValueError(
            f"Frozen window too wide: some k-points have more than nw={nw} "
            f"frozen bands.  Maximum is {n_froz.max().item()}."
        )

    if Amn is not None:
        V = _init_V_amn(Amn, outer_mask, frozen_mask, nk, nb, nw)
    else:
        V = _init_V_index(outer_mask, frozen_mask, nk, nb, nw, Mmn.dtype)

    # Frozen columns are placed first by both initializers above -- column j
    # at k is frozen iff j < n_froz[k].
    free_col_mask  = torch.arange(nw)[None, :] >= n_froz[:, None]     # (nk, nw) bool
    frozen_block   = V * (~free_col_mask).unsqueeze(-2)                # constant: fixed frozen columns
    outer_row_mask = outer_mask.to(V.dtype)                            # (nk, nb), 1.0/0.0

    def retract(V_cur: Tensor, step: Tensor) -> Tensor:
        step = step * outer_row_mask.unsqueeze(-1) * free_col_mask.unsqueeze(-2)
        return _retract_disentangle(V_cur, step, frozen_block, free_col_mask, outer_row_mask)

    history   = []
    converged = False
    d_prev    = None
    gcnorm0   = 0.0
    ncg       = 0

    for t in range(n_iter):
        omega_val, G_euc = _omega_i_and_grad(V, Mmn, wb, kb_idx)
        history.append(omega_val)

        G_masked = G_euc * outer_row_mask.unsqueeze(-1) * free_col_mask.unsqueeze(-2)
        G_riem = _riemannian_gradient(V, G_masked)
        gcnorm1 = _real_inner(G_riem, G_riem).item()

        if t == 0 or ncg >= num_cg_steps:
            beta = 0.0
            ncg = 0
        elif gcnorm0 > torch.finfo(torch.float64).eps:
            beta = gcnorm1 / gcnorm0
            if beta > cg_max_ratio:
                beta, ncg = 0.0, 0
            else:
                ncg += 1
        else:
            beta, ncg = 0.0, 0
        gcnorm0 = gcnorm1

        direction = -G_riem if beta == 0.0 else -G_riem + beta * d_prev
        doda0 = _real_inner(G_riem, direction).item()
        if doda0 >= 0.0:
            direction = -G_riem
            beta, ncg = 0.0, 0
            doda0 = _real_inner(G_riem, direction).item()
            if doda0 >= 0.0:
                direction = -direction
                doda0 = -doda0

        V_trial = retract(V, lr * direction)
        with torch.no_grad():
            omega_trial = _omega_i_diff(V_trial, Mmn, wb, kb_idx).item()

        denom = omega_trial - omega_val
        c = (denom - doda0 * lr) / lr**2 if abs(lr) > 0 else 0.0
        use_quadratic = abs(c) > torch.finfo(torch.float64).eps
        if use_quadratic:
            alpha = -0.5 * doda0 / c
            if doda0 * alpha >= 0.0:
                use_quadratic = False

        if use_quadratic:
            step_len = alpha
            V_cand = retract(V, step_len * direction)
            with torch.no_grad():
                omega_cand = _omega_i_diff(V_cand, Mmn, wb, kb_idx).item()
        else:
            step_len = lr
            V_cand, omega_cand = V_trial, omega_trial

        # Backtracking safety net: the parabolic-fit step is only a local
        # model and is not guaranteed to improve on omega_val -- it can
        # overshoot, especially right after a CG direction reset (this is
        # exactly what `test_omega_i_non_increasing` caught: an uncontrolled
        # jump partway through a run that only recovered several sweeps
        # later). Halve the step length until it does, or fall back to a
        # no-op sweep (keeps the whole history non-increasing by
        # construction) and reset CG momentum so the next sweep restarts
        # from a fresh steepest-descent direction instead of compounding a
        # bad one.
        backtracks = 0
        while omega_cand > omega_val + 1e-12 and backtracks < 20:
            step_len *= 0.5
            V_cand = retract(V, step_len * direction)
            with torch.no_grad():
                omega_cand = _omega_i_diff(V_cand, Mmn, wb, kb_idx).item()
            backtracks += 1

        if omega_cand <= omega_val + 1e-12:
            V = V_cand.detach()
        else:
            beta, ncg = 0.0, 0
            direction = -G_riem

        d_prev = direction

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    oi = _omega_i(V, Mmn, wb, kb_idx)
    return DisentangleResult(
        V=V, omega_i=oi, lwindow=outer_mask, ndimwin=n_outer,
        history=history, converged=converged,
    )
