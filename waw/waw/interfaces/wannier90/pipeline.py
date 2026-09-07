"""
Wannier90 file interface to the core Wannierization driver.

This is the eV/Angstrom + Wannier90-file boundary around
``waw.core.pipeline.wannierize`` (which is atomic-units-only):

  1. Load  — read .win/.nnkp/.eig/.mmn/.amn into a WannierData (eV -> Hartree,
             Angstrom -> Bohr done in the loader)
  2. Run   — convert eV disentanglement windows to Hartree, call the core driver
  3. Write — convert the atomic-unit result back and emit _hr.dat (eV),
             _centres.xyz (Angstrom), and .chk.fmt (Angstrom)

Entry point:
    result = wannierize("path/to/silicon", nw=4)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from ...core.pipeline import wannierize as _core_wannierize, WannierResult
from ...core.sitesym  import site_symmetry_from_dmn
from ...units         import BOHR_TO_ANG, HARTREE_TO_EV, EV_TO_HARTREE
from .loader          import load, parse_real_lattice, parse_recip_lattice
from .io              import read_win, read_dmn, write_hr, write_centres, write_chk_fmt

BOHR_TO_ANG2 = BOHR_TO_ANG ** 2


def _parse_exclude_bands(params: dict) -> np.ndarray:
    """
    1-based excluded band indices from the .win `exclude_bands` keyword,
    e.g. "1,2,5-7" -> [1, 2, 5, 6, 7].  Empty if the keyword is absent.
    """
    raw = params.get("exclude_bands")
    if raw is None:
        return np.array([], dtype=np.int64)
    indices: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return np.array(indices, dtype=np.int64)


def _parse_select_projections(params: dict) -> np.ndarray | None:
    """
    0-based selected-projection column indices from the .win
    `select_projections` keyword, e.g. "1 2 3 4" or "5-12" -> picks those
    columns out of the .amn's trial-projection axis (whose width is the
    number of declared `projections`, which can exceed num_wann -- W90
    uses `select_projections` to pick the num_wann of them to seed the SVD
    initial guess with). None if the keyword is absent (use Amn as-is).
    """
    raw = params.get("select_projections")
    if raw is None:
        return None
    indices: list[int] = []
    for part in re.split(r"[,\s]+", str(raw).strip()):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return np.array(indices, dtype=np.int64) - 1   # 1-based (.win) -> 0-based


def _eV_window_to_hartree(window):
    """Convert a user (E_min, E_max) window in eV to Hartree; None stays None."""
    if window is None:
        return None
    return tuple(EV_TO_HARTREE * e for e in window)


def wannierize(
    seedname:      str | Path,
    nw:            int | None                   = None,
    *,
    outer_window:  tuple[float, float] | None   = None,
    frozen_window: tuple[float, float] | None   = None,
    proj_min:      float | None                 = None,
    proj_max:      float | None                 = None,
    dis_spheres:   list[tuple[float, float, float, float]] | None = None,
    dis_spheres_first_wann: int                 = 0,
    use_sitesym:   bool                         = False,
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
    device:        str | torch.device           = "cpu",
    write_outputs: bool                         = True,
    verbose:       bool                         = True,
) -> WannierResult:
    """
    Run the complete Wannierization pipeline from raw Wannier90 input files.

    Reads the standard Wannier90 input files (.win, .nnkp, .eig, .mmn, .amn),
    runs the core driver (disentanglement if nb > nw, spread minimization, and
    the real-space Hamiltonian H(R)), and optionally writes _hr.dat,
    _centres.xyz and .chk.fmt output files.

    If the .win declares more `projections` than `num_wann` (a wider trial
    set than the target manifold, picked via W90's `select_projections`
    keyword), only the selected .amn columns are used to seed the SVD
    initial guess -- parsed and applied automatically, no extra argument
    needed here.

    Parameters
    ----------
    seedname : str or Path
        Path prefix shared by all input files, e.g. "tests/data/gaas/gaas".
        Reads <seedname>.win, .nnkp, .eig, .mmn, .amn.
    nw : int, optional
        Number of Wannier functions.  If None, read from ``num_wann`` in
        the .win file.
    outer_window : (E_min, E_max) in eV, optional
        Outer energy window for disentanglement (matching the .win
        ``dis_win_max``/``dis_win_min`` convention).  Only bands inside are
        used.  None means all bands (no outer window restriction).
    frozen_window : (E_min, E_max) in eV, optional
        Frozen energy window (matching ``dis_froz_max``/``dis_froz_min``).
        Bands inside are always included in the disentangled subspace.  None
        means no frozen bands.
    proj_min, proj_max : float, optional
        Wannier90's `dis_proj_min`/`dis_proj_max` (projectability-based
        disentanglement).  Dimensionless (no eV conversion); combines
        with (does not replace) outer_window/frozen_window.  See
        ``core.disentangle.disentangle``.
    dis_spheres, dis_spheres_first_wann :
        Wannier90's `dis_spheres`/`dis_spheres_first_wann` (k-space-
        localized disentanglement).  `dis_spheres` centres are
        fractional k-points, radii in Bohr^-1 (core convention, no eV/Ang
        conversion needed).  See ``core.disentangle.disentangle``.
    use_sitesym : bool
        Wannier90's `site_symmetry = .true.`.  Reads `<seedname>.dmn`
        (`interfaces.wannier90.io.read_dmn`, needs
        `pw2wannier90 write_dmn=.true.`) and runs disentanglement + spread
        minimization over the irreducible k-wedge only (`core.pipeline.
        wannierize`'s `sitesym`).  `n_restarts`/`n_hops` are forced off
        (global restarts/basin-hopping not implemented for the
        symmetrized optimizers); `guiding_centres`/`guide_refresh` are
        supported and forwarded as usual.
    dis_n_iter, dis_conv_tol, n_iter, conv_tol, conv_window, n_restarts,
    n_hops, hop_strength, optimizer, lr, seed, n_workers :
        Optimizer controls, forwarded to the core driver.
    guiding_centres, guide_refresh : bool, int
        Enable Wannier90-style guiding centres (prevents the MLWF "runaway
        centre" branch-cut pathology on systems prone to it -- small/few-k
        periodic metals). Off by default; not auto-parsed from the .win's
        own `guiding_centres` keyword. Pass `guiding_centres=True`
        explicitly if a run's Omega_D looks anomalously large or centres
        drift outside the cell.
    device : str or torch.device
        PyTorch compute device ("cpu" or "cuda") for the loaded tensors.
    write_outputs : bool
        If True (default), write <seedname>_hr.dat, <seedname>_centres.xyz,
        and <seedname>.chk.fmt (the latter is a formatted Wannier90
        checkpoint; run `w90chk2chk.x -import <seedname>` to convert it to
        the binary .chk that consumers such as EPW expect).
    verbose : bool
        If True (default), print progress to stdout.

    Returns
    -------
    WannierResult
        Atomic-unit result from the core driver (energies Hartree, lengths Bohr).
    """
    seed_path = Path(seedname)

    def log(msg: str) -> None:
        if verbose:
            print(f"[waw] {msg}")

    # ------------------------------------------------------------------
    # 1. Load input files (eV -> Hartree, Angstrom -> Bohr in the loader)
    # ------------------------------------------------------------------
    log(f"Loading: {seed_path}")
    params        = read_win(seed_path.with_suffix(".win"))
    recip_lattice = parse_recip_lattice(params)
    real_lattice  = parse_real_lattice(params)
    wdata         = load(seed_path, recip_lattice, device=device, params=params,
                        select_projections=_parse_select_projections(params))

    if nw is None:
        nw = int(params.get("num_wann", wdata.nw))

    mp_str = params.get("mp_grid", None)
    if mp_str is None:
        raise ValueError(
            "mp_grid not found in .win file; cannot build the real-space Hamiltonian."
        )
    mp_grid = tuple(int(x) for x in str(mp_str).split())

    sitesym = None
    if use_sitesym:
        dmn = read_dmn(seed_path.with_suffix(".dmn"), num_wann=nw)
        sitesym = site_symmetry_from_dmn(dmn)
        log(f"site_symmetry: nkptirr={sitesym.nkptirr} of {sitesym.num_kpts}, "
            f"nsymmetry={sitesym.nsymmetry}")

    # ------------------------------------------------------------------
    # 2. Run the core driver (atomic units; eV windows -> Hartree here)
    # ------------------------------------------------------------------
    result = _core_wannierize(
        wdata, nw, mp_grid, real_lattice,
        outer_window  = _eV_window_to_hartree(outer_window),
        frozen_window = _eV_window_to_hartree(frozen_window),
        proj_min      = proj_min,
        proj_max      = proj_max,
        dis_spheres   = dis_spheres,
        dis_spheres_first_wann = dis_spheres_first_wann,
        sitesym       = sitesym,
        dis_n_iter    = dis_n_iter,
        dis_conv_tol  = dis_conv_tol,
        dis_mix_ratio = dis_mix_ratio,
        n_iter        = n_iter,
        conv_tol      = conv_tol,
        conv_window   = conv_window,
        n_restarts    = n_restarts,
        n_hops        = n_hops,
        hop_strength  = hop_strength,
        optimizer     = optimizer,
        lr            = lr,
        guiding_centres = guiding_centres,
        guide_refresh   = guide_refresh,
        seed          = seed,
        n_workers     = n_workers,
        verbose       = verbose,
    )

    # ------------------------------------------------------------------
    # 3. Write Wannier90 output files (atomic units -> eV / Angstrom)
    # ------------------------------------------------------------------
    if write_outputs:
        hr_path  = seed_path.parent / (seed_path.name + "_hr.dat")
        xyz_path = seed_path.parent / (seed_path.name + "_centres.xyz")
        chk_path = seed_path.parent / (seed_path.name + ".chk.fmt")

        # hr.H_R is in Hartree (core units); Wannier90 _hr.dat is in eV.
        H_R_np  = result.hr.H_R.detach().cpu().numpy() * HARTREE_TO_EV
        write_hr(hr_path, H_R_np, result.hr.R_vectors, result.hr.degen, nw,
                 seedname=seed_path.name)

        centres_ang  = result.centres_bohr * BOHR_TO_ANG
        spreads_ang2 = result.spreads_bohr2 * BOHR_TO_ANG2
        write_centres(xyz_path, centres_ang, spreads_ang2)

        dis_result = result.dis
        write_chk_fmt(
            chk_path,
            num_bands        = wdata.nb,
            exclude_bands    = _parse_exclude_bands(params),
            real_lattice     = real_lattice * BOHR_TO_ANG,     # .chk.fmt is in Angstrom
            recip_lattice    = recip_lattice / BOHR_TO_ANG,    # .chk.fmt is in Angstrom^-1
            mp_grid          = mp_grid,
            kpt_latt         = wdata.kpts.detach().cpu().numpy(),
            nntot            = wdata.nnb,
            num_wann         = nw,
            have_disentangled = dis_result is not None,
            omega_invariant  = dis_result.omega_i * BOHR_TO_ANG2 if dis_result is not None else 0.0,
            lwindow          = dis_result.lwindow.detach().cpu().numpy() if dis_result is not None else None,
            ndimwin          = dis_result.ndimwin.detach().cpu().numpy() if dis_result is not None else None,
            u_matrix_opt     = dis_result.V.detach().cpu().numpy() if dis_result is not None else None,
            u_matrix         = result.spread.U_final.detach().cpu().numpy(),
            m_matrix         = result.m_tilde.detach().cpu().numpy(),
            wannier_centres  = centres_ang,
            wannier_spreads  = spreads_ang2,
            checkpoint       = "postwann",
            header           = "written by waw",
        )

        log(f"Wrote {hr_path.name}, {xyz_path.name}, and {chk_path.name}")

    return result
