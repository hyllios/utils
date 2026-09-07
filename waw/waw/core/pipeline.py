"""
Core Wannierization driver — atomic units only.

Chains the numerical engine end to end, on already-loaded data:

  1. Dis   — iterative subspace disentanglement   (skipped when nb == nw)
  2. Init  — SVD gauge initialization
  3. Min   — Riemannian spread minimization (with optional global search)
  4. H(R)  — real-space Wannier Hamiltonian via Fourier transform

Atomic units throughout: energy windows and H(R) in Hartree, spreads/centres
in Bohr^2/Bohr. No files are read/written and no eV/Angstrom appears here —
see ``waw.interfaces.wannier90.pipeline.wannierize`` for the file/eV wrapper.

Entry point:
    result = wannierize(wdata, nw, mp_grid, real_lattice_bohr)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .types       import WannierData
from .disentangle import DisentangleResult, disentangle, disentangle_joint
from .global_optim import global_minimize_spread
from .hamiltonian  import HamiltonianR, compute_hr, hopping_range
from .init         import svd_init
from .optim        import SpreadResult, minimize_spread_symmetrized, minimize_spread_slwf
from .sitesym      import SiteSymmetry
from .spread       import (
    rotate_overlaps, compute_spread_from_M_tilde, compute_slwf_spread_from_M_tilde,
    compute_ss_spread_from_M_tilde, _guided_phase, _canonical_b_permutation,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WannierResult:
    """
    Complete output of the Wannierization driver (atomic units throughout).

    Attributes
    ----------
    spread         : SpreadResult      final spread minimization (U_final, Omega, centres)
    hr             : HamiltonianR      real-space Hamiltonian H(R), in Hartree
    dis            : DisentangleResult subspace selection; None for isolated bands
    wdata          : WannierData       input data (Mmn, Amn, eig, …)
    spreads_bohr2  : (nw,) float64    per-WF spread σ_n² in Bohr²
    omega_final    : float             Ω recomputed at U_final (Bohr²; reference for spreads sum)
    centres_bohr   : (nw, 3) float64  final Wannier centres in Bohr (consistent with omega_final)
    m_tilde        : Tensor            final gauge-rotated overlaps M̃ (nk, nnb, nw, nw)
    """

    spread:        SpreadResult
    hr:            HamiltonianR
    dis:           DisentangleResult | None
    wdata:         WannierData
    spreads_bohr2: np.ndarray
    omega_final:   float = 0.0
    centres_bohr:  np.ndarray | None = None
    m_tilde:       Tensor | None = None


# ---------------------------------------------------------------------------
# Persisting a Wannierization for reuse (skip the expensive re-run)
# ---------------------------------------------------------------------------

def save_wannier_result(result: "WannierResult", path) -> "Path":
    """Save the analysis-relevant part of a Wannierization to a ``.npz``.

    Stores everything downstream analysis needs -- the real-space Hamiltonian
    ``H(R)`` (``result.hr``), the Wannier centres, the spreads, and
    ``Omega``/``Omega_I`` -- so a later session can ``load_wannier_result``
    and go straight to band interpolation, surface spectral functions, ARPES,
    etc. without redoing the (expensive) disentanglement + MLWF minimization.

    The **gauge** (``spread.U_final`` and, when the bands were disentangled,
    ``dis.V``) is stored too when present: it is small (``(nk, nw, nw)`` and
    ``(nk, nb, nw)``) but it is what lets a reloaded result be used for
    quantities defined on the ORIGINAL Bloch states rather than only on the
    interpolated model -- above all ``W = V @ U_final``, the rotation
    `analysis.elph.wannier_transform_elph` needs to take the electron-phonon
    matrix element from the Bloch to the Wannier gauge. Without it an el-ph
    notebook could not resume from a cache at all.

    The heavy inputs (``wdata`` overlaps, ``m_tilde``) are intentionally NOT
    stored -- ``wdata`` in particular is gigabytes. Round-trips ``result.hr`` /
    ``centres_bohr`` / ``omega_final`` / ``spreads_bohr2`` / ``dis.omega_i``
    exactly.
    """
    from pathlib import Path
    path = Path(path)
    hr = result.hr
    H_R = hr.H_R.detach().cpu().numpy() if hasattr(hr.H_R, "detach") else np.asarray(hr.H_R)
    omega_i = np.nan
    if result.dis is not None and getattr(result.dis, "omega_i", None) is not None:
        omega_i = float(result.dis.omega_i)
    centres = (np.asarray(result.centres_bohr) if result.centres_bohr is not None
               else np.zeros((0, 3)))

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    gauge = {}
    U_final = getattr(result.spread, "U_final", None) if result.spread is not None else None
    if U_final is not None:
        gauge["U_final"] = _np(U_final)
    V = getattr(result.dis, "V", None) if result.dis is not None else None
    if V is not None:
        gauge["dis_V"] = _np(V)

    # the interpolation is centre-aware (use_ws_distance) only when the model
    # knows its centres, lattice and mesh -- store them, or a reloaded model
    # silently falls back to the plain Wigner-Seitz sum
    ws_meta = {}
    if hr.real_lattice is not None:
        ws_meta["real_lattice"] = np.asarray(hr.real_lattice, dtype=np.float64)
    if hr.mp_grid is not None:
        ws_meta["mp_grid"] = np.asarray(hr.mp_grid, dtype=np.int64)

    np.savez_compressed(
        path,
        H_R=H_R,
        R_vectors=np.asarray(hr.R_vectors, dtype=np.int64),
        degen=np.asarray(hr.degen, dtype=np.int64),
        nw=np.int64(hr.nw),
        centres_bohr=centres,
        **ws_meta,
        spreads_bohr2=np.asarray(result.spreads_bohr2, dtype=np.float64),
        omega_final=np.float64(result.omega_final),
        omega_i=np.float64(omega_i),
        **gauge,
    )
    # np.savez appends .npz if missing; return the actual file path
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_wannier_result(path) -> "WannierResult":
    """Reload a Wannierization saved by :func:`save_wannier_result`.

    Returns a ``WannierResult`` whose ``hr``, ``centres_bohr``,
    ``omega_final``, ``spreads_bohr2`` and ``dis.omega_i`` are populated (the
    fields any analysis routine uses), plus the **gauge** when the file carries
    it: ``spread.U_final`` and ``dis.V`` come back as tensors, so
    ``W = V @ U_final`` can be rebuilt for `analysis.elph`.

    ``spread`` / ``dis`` are lightweight stand-ins exposing only those fields
    (``wdata`` and ``m_tilde`` are never stored and stay ``None``). Files
    written before the gauge was stored load fine, with ``spread`` ``None`` --
    callers that need the gauge should check for that and redo the
    minimization.
    """
    from types import SimpleNamespace
    import warnings
    d = np.load(path)
    centres = d["centres_bohr"]
    centres = centres if centres.shape[0] else None
    stored_files = set(d.files)
    hr = HamiltonianR(
        H_R=torch.from_numpy(d["H_R"]),
        R_vectors=d["R_vectors"].astype(np.int64),
        degen=d["degen"].astype(np.int64),
        nw=int(d["nw"]),
        centres=centres,
        real_lattice=(d["real_lattice"] if "real_lattice" in stored_files else None),
        mp_grid=(tuple(int(x) for x in d["mp_grid"])
                 if "mp_grid" in stored_files else None),
    )
    if hr.real_lattice is None or hr.mp_grid is None:
        warnings.warn(
            f"{path}: saved before the lattice/mesh were stored, so this model "
            "cannot build its use_ws_distance data and every interpolation from "
            "it falls back to the plain Wigner-Seitz sum. That is a real "
            "difference -- on bcc Nb it was 3797 vs 52 meV at E_F. Re-save it "
            "from a fresh wannierize, or pass `ws=` explicitly.",
            RuntimeWarning, stacklevel=2)
    omega_i = float(d["omega_i"])
    stored = stored_files

    spread = None
    if "U_final" in stored:
        spread = SimpleNamespace(U_final=torch.from_numpy(d["U_final"]))
    dis = None
    if not np.isnan(omega_i) or "dis_V" in stored:
        V = torch.from_numpy(d["dis_V"]) if "dis_V" in stored else None
        dis = SimpleNamespace(omega_i=(None if np.isnan(omega_i) else omega_i), V=V)
    return WannierResult(
        spread=spread, hr=hr, dis=dis, wdata=None,
        spreads_bohr2=d["spreads_bohr2"],
        omega_final=float(d["omega_final"]),
        centres_bohr=centres, m_tilde=None,
    )


# ---------------------------------------------------------------------------
# Internal helper: per-WF spreads
# ---------------------------------------------------------------------------

def _per_wf_spreads(
    M_tilde: Tensor,
    wb:      Tensor,
    bvecs:   Tensor,
    centres: Tensor,
    rguide:  Tensor | None = None,
    use_ss_functional: bool = False,
) -> np.ndarray:
    """
    Per-WF total spread σ_n² in Bohr².

    Decomposition (MV97):
      σ_n² = Ω_{I+OD,n} + Ω_{D,n}
      Ω_{I+OD,n} = (1/Nk) Σ_{k,b} w_b (1 − |M̃_nn|²)
      Ω_{D,n}    = (1/Nk) Σ_{k,b} w_b (−Im ln M̃_nn − b·r̄_n)²

    Ω_{I+OD,n} absorbs both the gauge-invariant Ω_I and off-diagonal Ω_OD
    contributions for WF n, so Σ_n σ_n² = Ω_I + Ω_OD + Ω_D = Ω.

    ``rguide``, when given (guiding_centres enabled), must be the same
    reference used for ``centres`` or the two use inconsistent phase-branch
    conventions.

    ``use_ss_functional``: Ω_{I+OD,n} is shared with the MV functional
    (see `_ss_spread_from_M_tilde`); only Ω_{D,n} is replaced by the SS
    per-orbital k-variance of M_nn, disaggregated by n instead of summed.
    """
    nk = M_tilde.shape[0]
    diag = M_tilde.diagonal(dim1=-2, dim2=-1)             # (nk, nnb, nw)
    iod_n = torch.einsum("b,kbn->n", wb, 1.0 - diag.abs().pow(2)) / nk

    if use_ss_functional:
        nw = diag.shape[-1]
        perm = _canonical_b_permutation(bvecs)
        diag_c = torch.gather(diag, 1, perm.unsqueeze(-1).expand(-1, -1, nw))   # (nk, nnb, nw)
        mean_M = diag_c.mean(dim=0)
        mean_abs2 = diag_c.abs().pow(2).mean(dim=0)
        variance = mean_abs2 - mean_M.abs().pow(2)          # (nnb, nw)
        od_n = torch.einsum("b,bn->n", wb, variance)
    else:
        phase = _guided_phase(diag, bvecs, rguide) if rguide is not None else torch.angle(diag)
        b_dot_r = torch.einsum("kba,na->kbn", bvecs, centres)  # (nk, nnb, nw)
        od_n = torch.einsum("b,kbn->n", wb, (-phase - b_dot_r).pow(2)) / nk

    return (iod_n + od_n).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------

def wannierize(
    wdata:         WannierData,
    nw:            int,
    mp_grid:       tuple[int, int, int],
    real_lattice:  np.ndarray,
    *,
    outer_window:  tuple[float, float] | None   = None,
    frozen_window: tuple[float, float] | None   = None,
    proj_min:      float | None                 = None,
    proj_max:      float | None                 = None,
    dis_spheres:   list[tuple[float, float, float, float]] | None = None,
    dis_spheres_first_wann: int                 = 0,
    sitesym:       SiteSymmetry | None          = None,
    slwf_num:       int | None                  = None,
    slwf_constrain: bool                        = False,
    slwf_target_centres: Tensor | None          = None,
    slwf_lambda:    float                       = 1.0,
    dis_solver:    str                          = "zmatrix",
    dis_n_iter:    int                          = 200,
    dis_conv_tol:  float                        = 1e-10,
    dis_mix_ratio: float                        = 0.5,
    n_iter:        int                          = 1000,
    conv_tol:      float                        = 1e-10,
    conv_window:   int                          = 5,
    n_restarts:    int                          = 1,
    n_hops:        int                          = 0,
    hop_strength:  float                        = 0.3,
    optimizer:     str                          = "cg",
    lr:            float                        = 3e-2,
    seed:          int                          = 0,
    n_workers:     int | None                   = None,
    guiding_centres: bool                       = False,
    guide_refresh: int                          = 10,
    use_ss_functional: bool                     = False,
    verbose:       bool                         = True,
) -> WannierResult:
    """
    Run the Wannierization engine on a loaded WannierData, in atomic units.

    Disentangles (if nb > nw), minimizes the spread functional, and builds the
    real-space Hamiltonian H(R).  All energies are Hartree, all lengths Bohr.

    Parameters
    ----------
    wdata : WannierData
        Loaded overlaps/projections/eigenvalues (eig in Hartree).
    nw : int
        Number of Wannier functions.
    mp_grid : (N1, N2, N3)
        Monkhorst-Pack grid, needed to build H(R).
    real_lattice : (3, 3) float64
        Real-space lattice vectors as rows, in Bohr.
    outer_window, frozen_window : (E_min, E_max) in Hartree, optional
        Disentanglement windows.  None means all bands / no frozen bands.
    proj_min, proj_max : float, optional
        Wannier90's `dis_proj_min`/`dis_proj_max` (projectability-based
        disentanglement). Combines with (does not replace)
        outer_window/frozen_window; requires wdata.Amn. See
        ``core.disentangle.disentangle``.
    dis_spheres, dis_spheres_first_wann :
        Wannier90's `dis_spheres`/`dis_spheres_first_wann` (k-space-
        localized disentanglement). See ``core.disentangle.disentangle``.
    sitesym : core.sitesym.SiteSymmetry, optional (Wannier90's
        `site_symmetry = .true.`). When given, disentanglement AND spread
        minimization both run over the irreducible k-wedge only
        (``core.disentangle.disentangle``'s `sitesym`, `core.optim.
        minimize_spread_symmetrized`), broadcasting to the rest of the
        mesh via the crystal's point-group action. `n_restarts`/`n_hops`
        are not supported together with `sitesym` (global restarts/
        basin-hopping aren't implemented for the symmetrized optimizers)
        -- `n_restarts` is silently forced to 1 in that case.
        `guiding_centres`/`guide_refresh` are supported and forwarded as
        usual. For isolated bands (V is None), U's left index transforms
        under `d_matrix_band`, not `d_matrix_wann` -- see
        `core.optim.minimize_spread_symmetrized`. Frozen bands
        (`frozen_window`/`proj_max`) ARE supported together with `sitesym`:
        the disentanglement then symmetrizes the subspace rather than the
        frame, since Omega_I is gauge invariant and `d_matrix_wann` cannot act
        on a subset of the columns -- see ``disentangle`` for the two data
        preconditions it checks. Default optimizer is `'cg'`; `'adam'` can
        get stuck in symmetrized minimizations (see `core.optim.
        minimize_spread_symmetrized`).
    slwf_num : int, optional (Wannier90's `slwf_num`). Selectively
        localized Wannier functions: only the first `slwf_num` of `nw`
        Wannier functions are localized (`core.optim.minimize_spread_slwf`,
        `core.spread.compute_slwf_spread`); the rest are "spectator" WFs
        excluded from the spread functional. Mutually exclusive with
        `sitesym` (raises if both are given). `n_restarts`/`n_hops` are
        not supported here either (forced to a single plain
        `minimize_spread_slwf` run, no global search).
    slwf_constrain, slwf_target_centres, slwf_lambda :
        Wannier90's `slwf_constrain`/`slwf_centres`/`slwf_lambda`: when
        `slwf_constrain=True`, additionally pin the OWF centres to
        `slwf_target_centres` (Bohr, shape `(slwf_num, 3)`; required in
        that case -- not defaulted to trial-projection centres). `slwf_lambda`
        is the Lagrange multiplier (wannier90's `lambda_loc`, ignored when
        `slwf_constrain=False`).
    dis_solver : str
        `'zmatrix'` (default): the original Souza-Marzari-Vanderbilt per-k
        coordinate descent (``core.disentangle.disentangle``) -- picks each
        k's subspace independently each sweep from a top-eigenvector
        problem, with no explicit mechanism to keep neighbouring k's choices
        consistent when the relevant eigenvalues are nearly degenerate.
        `'joint_cg'`: minimizes the SAME Omega_I jointly over the whole
        mesh at once by Riemannian nonlinear CG
        (``core.disentangle.disentangle_joint``), removing that per-k
        independence. Does not support `sitesym`, `dis_spheres`, or
        `proj_min`/`proj_max` (raises `ValueError` if combined with any of
        those); `dis_mix_ratio` does not apply to it either (CG's own
        momentum replaces the Z-matrix damping).
    dis_n_iter, dis_conv_tol, n_iter, conv_tol, conv_window, n_restarts,
    n_hops, hop_strength, optimizer, lr, seed, n_workers :
        Optimizer controls (see ``global_minimize_spread`` / ``disentangle``).
    guiding_centres, guide_refresh :
        Enable Wannier90-style guiding centres in the spread minimization
        (prevents the MLWF "runaway centre" branch-cut pathology on
        systems prone to it -- small/few-k periodic metals). Off by
        default; see ``optim.minimize_spread``.
    use_ss_functional : bool (Wannier90's `use_ss_functional`).
        Use the Stengel-Spaldin alternative localization functional
        instead of the ordinary Marzari-Vanderbilt one -- see
        ``core.spread._ss_spread_from_M_tilde``/``core.optim.
        minimize_spread``. Mutually exclusive with `sitesym` (raises)
        and ignored by the SLWF path (which has its own spread functional).
    verbose : bool
        If True (default), print progress (Ω in Bohr²) to stdout.

    Returns
    -------
    WannierResult
    """
    if slwf_num is not None and sitesym is not None:
        raise ValueError("slwf_num and sitesym are mutually exclusive (not supported together).")
    if slwf_num is not None and slwf_constrain and slwf_target_centres is None:
        raise ValueError("slwf_target_centres is required when slwf_constrain=True.")
    if use_ss_functional and sitesym is not None:
        raise ValueError("use_ss_functional and sitesym are mutually exclusive (not supported together).")
    if dis_solver not in ("zmatrix", "joint_cg"):
        raise ValueError(f"dis_solver must be 'zmatrix' or 'joint_cg', got {dis_solver!r}.")
    if dis_solver == "joint_cg":
        if sitesym is not None:
            raise ValueError("dis_solver='joint_cg' does not support sitesym yet.")
        if dis_spheres is not None:
            raise ValueError("dis_solver='joint_cg' does not support dis_spheres yet.")
        if proj_min is not None or proj_max is not None:
            raise ValueError("dis_solver='joint_cg' does not support proj_min/proj_max yet.")

    def log(msg: str) -> None:
        if verbose:
            print(f"[waw] {msg}")

    log(f"nk={wdata.nk}  nb={wdata.nb}  nw={nw}  nnb={wdata.nnb}")

    # ------------------------------------------------------------------
    # 1. Disentanglement (skipped when nb == nw)
    # ------------------------------------------------------------------
    dis_result: DisentangleResult | None = None

    if wdata.nb > nw:
        log(f"Disentangling ({dis_solver}): nb={wdata.nb} → nw={nw} ...")
        if dis_solver == "joint_cg":
            dis_result = disentangle_joint(
                wdata.Mmn, wdata.eig, wdata.wb, wdata.kb_idx, nw,
                Amn           = wdata.Amn,
                outer_window  = outer_window,
                frozen_window = frozen_window,
                n_iter        = dis_n_iter,
                conv_tol      = dis_conv_tol,
            )
        else:
            dis_result = disentangle(
                wdata.Mmn, wdata.eig, wdata.wb, wdata.kb_idx, nw,
                Amn           = wdata.Amn,
                outer_window  = outer_window,
                frozen_window = frozen_window,
                proj_min      = proj_min,
                proj_max      = proj_max,
                n_iter        = dis_n_iter,
                conv_tol      = dis_conv_tol,
                mix_ratio     = dis_mix_ratio,
                bvecs         = wdata.bvecs,
                kpts          = wdata.kpts,
                real_lattice  = real_lattice,
                dis_spheres   = dis_spheres,
                dis_spheres_first_wann = dis_spheres_first_wann,
                sitesym       = sitesym,
            )
        status = "converged" if dis_result.converged else "not converged"
        log(f"  Omega_I = {dis_result.omega_i:.6f} Bohr²  ({status})")
        V = dis_result.V    # (nk, nb, nw)
        # Project overlaps into the nw-dimensional subspace
        Mmn_opt  = rotate_overlaps(V, wdata.Mmn, wdata.kb_idx)   # (nk, nnb, nw, nw)
        Amn_proj = torch.bmm(
            V.conj().transpose(-1, -2), wdata.Amn
        )                                                          # (nk, nw, nw)
    else:
        log("Isolated bands — no disentanglement needed.")
        V        = None
        Mmn_opt  = wdata.Mmn                                       # (nk, nnb, nw, nw)
        Amn_proj = wdata.Amn                                       # (nk, nb=nw, nw)

    # ------------------------------------------------------------------
    # 2. SVD initialization
    # ------------------------------------------------------------------
    U_init = svd_init(Amn_proj)   # (nk, nw, nw)

    with torch.no_grad():
        omega0 = compute_spread_from_M_tilde(
            rotate_overlaps(U_init, Mmn_opt, wdata.kb_idx), wdata.wb, wdata.bvecs
        )[0]
    log(f"SVD init Omega = {omega0.item():.4f} Bohr²")

    # ------------------------------------------------------------------
    # 3. Spread minimization
    # ------------------------------------------------------------------
    if sitesym is not None:
        log(f"Minimizing spread  ({optimizer.upper()}, symmetrized, "
            f"nkptirr={sitesym.nkptirr} of {sitesym.num_kpts}) ...")
        U_irr_init = U_init[sitesym.ir2ik]   # (nkptirr, nw, nw)
        # Isolated bands (V is None): U's left index is the raw candidate-
        # band manifold, not Wannier gauge, even though nb == nw -- it
        # transforms under d_matrix_band, not d_matrix_wann.
        d_left = sitesym.d_matrix_band if V is None else None
        spread_result = minimize_spread_symmetrized(
            U_irr_init, sitesym, Mmn_opt, wdata.wb, wdata.bvecs, wdata.kb_idx,
            optimizer   = optimizer,
            lr          = lr,
            n_iter      = n_iter,
            conv_tol    = conv_tol,
            conv_window = conv_window,
            guiding_centres = guiding_centres,
            guide_refresh   = guide_refresh,
            d_left      = d_left,
        )
    elif slwf_num is not None:
        log(f"Minimizing spread  ({optimizer.upper()}, SLWF, "
            f"slwf_num={slwf_num} of {nw}, constrain={slwf_constrain}) ...")
        spread_result = minimize_spread_slwf(
            U_init, Mmn_opt, wdata.wb, wdata.bvecs, wdata.kb_idx, slwf_num,
            constrain      = slwf_constrain,
            target_centres = slwf_target_centres,
            lambda_        = slwf_lambda,
            optimizer   = optimizer,
            lr          = lr,
            n_iter      = n_iter,
            conv_tol    = conv_tol,
            conv_window = conv_window,
            guiding_centres = guiding_centres,
            guide_refresh   = guide_refresh,
        )
    else:
        log(
            f"Minimizing spread  ({optimizer.upper()}, "
            f"n_restarts={n_restarts}, n_hops={n_hops}) ..."
        )
        # Rank restarts by the reach of H(R), not by Omega. Omega is
        # translation-invariant per Wannier function, so two gauges whose WFs
        # differ by lattice vectors score identically while interpolating
        # completely differently -- on bcc Nb that let argmin(Omega) pick a
        # model 52x worse at E_F. Selection only: every candidate is already a
        # converged minimum of Omega, and building H(R) per restart is one FFT.
        def _hr_score(res):
            Wc = torch.bmm(V, res.U_final) if V is not None else res.U_final
            return hopping_range(
                compute_hr(Wc, wdata.eig, wdata.kpts, mp_grid, real_lattice),
                real_lattice)

        spread_result = global_minimize_spread(
            U_init, Mmn_opt, wdata.wb, wdata.bvecs, wdata.kb_idx,
            score        = _hr_score if n_restarts > 1 else None,
            n_restarts   = n_restarts,
            n_hops       = n_hops,
            hop_strength = hop_strength,
            optimizer    = optimizer,
            lr           = lr,
            n_iter       = n_iter,
            conv_tol     = conv_tol,
            conv_window  = conv_window,
            seed         = seed,
            n_workers    = n_workers,
            guiding_centres = guiding_centres,
            guide_refresh   = guide_refresh,
            use_ss_functional = use_ss_functional,
        )

    U      = spread_result.U_final
    status = "converged" if spread_result.converged else "not converged"
    if slwf_num is not None:
        log(
            f"  Omega = {spread_result.Omega:.6f} Bohr²  "
            f"(IOD={spread_result.Omega_IOD:.4f}, D={spread_result.Omega_D:.4f}, "
            f"nu={spread_result.Omega_nu:.4f})  [{status}]"
        )
    else:
        log(
            f"  Omega = {spread_result.Omega:.6f} Bohr²  "
            f"(I={spread_result.Omega_I:.4f}, D={spread_result.Omega_D:.4f}, "
            f"OD={spread_result.Omega_OD:.4f})  [{status}]"
        )

    # Per-WF spreads from the final rotated overlaps.
    # Recompute centres from U_final so they are consistent with M_tilde;
    # SpreadResult.centres is from the penultimate U (before the last retract).
    # The centres formula is identical for SLWF and plain MLWF -- only
    # omega_final_val needs branching to report the quantity actually minimized.
    with torch.no_grad():
        M_tilde = rotate_overlaps(U, Mmn_opt, wdata.kb_idx)
        _, _, _, _, centres_final = compute_spread_from_M_tilde(
            M_tilde, wdata.wb, wdata.bvecs, spread_result.rguide
        )
        if slwf_num is not None:
            omega_final_val = compute_slwf_spread_from_M_tilde(
                M_tilde, wdata.wb, wdata.bvecs, slwf_num, slwf_constrain,
                slwf_target_centres, slwf_lambda, spread_result.rguide,
            )[0]
        elif use_ss_functional:
            # Report the functional actually minimized -- the ordinary MV
            # Omega evaluated at an SS-converged U is a different quantity.
            omega_final_val = compute_ss_spread_from_M_tilde(
                M_tilde, wdata.wb, wdata.bvecs
            )[0]
        else:
            omega_final_val = compute_spread_from_M_tilde(
                M_tilde, wdata.wb, wdata.bvecs, spread_result.rguide
            )[0]
        omega_final_val = float(omega_final_val.item())
    spreads_bohr2 = _per_wf_spreads(
        M_tilde, wdata.wb, wdata.bvecs, centres_final, spread_result.rguide,
        use_ss_functional=use_ss_functional and slwf_num is None,
    )

    # ------------------------------------------------------------------
    # 4. Real-space Hamiltonian (Hartree)
    # ------------------------------------------------------------------
    # Full gauge W = V @ U for entangled; W = U for isolated
    if V is not None:
        W = torch.bmm(V, U)    # (nk, nb, nw)
    else:
        W = U                  # (nk, nw, nw)

    hr = compute_hr(W, wdata.eig, wdata.kpts, mp_grid, real_lattice)
    # centres make the interpolation centre-aware (use_ws_distance) by default
    hr.centres = centres_final.detach().cpu().numpy()
    log(f"H(R) built: nR={hr.H_R.shape[0]}  mp_grid={mp_grid}")

    return WannierResult(
        spread        = spread_result,
        hr            = hr,
        dis           = dis_result,
        wdata         = wdata,
        spreads_bohr2 = spreads_bohr2,
        omega_final   = omega_final_val,
        centres_bohr  = centres_final.detach().cpu().numpy(),
        m_tilde       = M_tilde,
    )
