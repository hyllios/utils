"""
ASE / numpy driver for the atomic-unit core.

Builds a core ``WannierData`` from plain numpy overlap arrays and drives
the core wannierize engine from an ``ase.Atoms`` structure. This is the
eV/Angstrom <-> atomic-unit boundary for the ASE interface: eigenvalues
arrive in eV and lattices in Angstrom, converted to Hartree/Bohr here.

Overlap topology
----------------
Building the finite-difference b-vectors needs the k-point neighbour table
(``nnkpts``) and folding G-vectors (``g_vectors``) — the information a Wannier90
``.nnkp`` file carries.  Supply them as numpy arrays, or reconstruct them from a
``.mmn`` k-pair map with ``nnkp_from_kpb_map`` when no ``.nnkp`` is available.

numpy interchange
-----------------
``save_npz`` / ``load_npz`` round-trip the whole problem (overlaps + topology +
lattice + mesh) through a single ``.npz`` file — no Wannier90 text files needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...core.types    import WannierData
from ...core.kmesh    import _compute_bvecs_and_weights, generate_nnkp
from ...core.pipeline import wannierize as _core_wannierize, WannierResult
from ...core.sitesym  import SiteSymmetry
from ...units         import EV_TO_HARTREE, ANG_TO_BOHR
from . import structure


# ---------------------------------------------------------------------------
# Neighbour-topology reconstruction (when no .nnkp is available)
# ---------------------------------------------------------------------------

def nnkp_from_kpb_map(kpb_map, nk: int, nnb: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct (nnkpts, g_vectors) from a Wannier90 ``.mmn`` k-pair map.

    ``kpb_map`` is the list of ``(ik, ik2, g1, g2, g3)`` block headers returned
    by ``waw.interfaces.wannier90.io.read_mmn`` (0-based k-indices).

    Returns
    -------
    nnkpts    : (nk, nnb) int64   neighbour k-index table
    g_vectors : (nk, nnb, 3) int64  folding G-vectors
    """
    nnkpts    = np.zeros((nk, nnb), dtype=np.int64)
    g_vectors = np.zeros((nk, nnb, 3), dtype=np.int64)
    # Track neighbours filled per k-point so assignment is order-independent.
    ib_counter: dict[int, int] = {}
    for (ik, ik2, g1, g2, g3) in kpb_map:
        ib = ib_counter.get(ik, 0)
        nnkpts[ik, ib]    = ik2
        g_vectors[ik, ib] = (g1, g2, g3)
        ib_counter[ik]    = ib + 1
    return nnkpts, g_vectors


# ---------------------------------------------------------------------------
# Build a core WannierData from arrays
# ---------------------------------------------------------------------------

def build_wannier_data(
    recip_lattice: np.ndarray,
    kpts:          np.ndarray,
    mmn:           np.ndarray,
    amn:           np.ndarray,
    eig:           np.ndarray,
    nnkpts:        np.ndarray,
    g_vectors:     np.ndarray,
    *,
    device:        str | torch.device = "cpu",
    dtype_real:    torch.dtype = torch.float64,
    dtype_complex: torch.dtype = torch.complex128,
) -> WannierData:
    """
    Assemble a core ``WannierData`` (atomic units) from numpy arrays.

    Parameters
    ----------
    recip_lattice : (3, 3) float64
        Reciprocal lattice rows in Bohr^-1 (2π convention); e.g. from
        ``structure.recip_lattice(atoms)``.
    kpts : (nk, 3) float64
        k-points in crystal coordinates.
    mmn : (nk, nnb, nb, nb) complex
        Overlap matrices M^(k,b).
    amn : (nk, nb, nw) complex
        Trial-projection matrices A^(k).
    eig : (nk, nb) float64
        Band eigenvalues **in eV** (converted to Hartree here).
    nnkpts : (nk, nnb) int
        Neighbour k-index table.
    g_vectors : (nk, nnb, 3) int
        Folding G-vectors.

    Returns
    -------
    WannierData
        eig in Hartree, b-vectors in Bohr^-1 — ready for the core engine.
    """
    kpts      = np.asarray(kpts, dtype=np.float64)
    nnkpts    = np.asarray(nnkpts, dtype=np.int64)
    g_vectors = np.asarray(g_vectors, dtype=np.int64)

    bvecs_np, wb_np = _compute_bvecs_and_weights(
        kpts          = kpts,
        nnkpts        = nnkpts,
        g_vectors     = g_vectors,
        recip_lattice = np.asarray(recip_lattice, dtype=np.float64),
    )

    def to_real(arr):
        return torch.tensor(np.asarray(arr), dtype=dtype_real, device=device)

    def to_cplx(arr):
        return torch.tensor(np.asarray(arr), dtype=dtype_complex, device=device)

    return WannierData(
        Mmn    = to_cplx(mmn),
        Amn    = to_cplx(amn),
        # .eig arrives in eV; the core works in Hartree.
        eig    = to_real(np.asarray(eig, dtype=np.float64) * EV_TO_HARTREE),
        kpts   = to_real(kpts),
        bvecs  = to_real(bvecs_np),
        wb     = to_real(wb_np),
        kb_idx = torch.tensor(nnkpts, dtype=torch.long, device=device),
        params = {},
    )


def wannier_data_from_atoms(
    atoms,
    kpts:      np.ndarray,
    mmn:       np.ndarray,
    amn:       np.ndarray,
    eig:       np.ndarray,
    nnkpts:    np.ndarray,
    g_vectors: np.ndarray,
    **kwargs,
) -> WannierData:
    """
    ``build_wannier_data`` with the reciprocal lattice taken from an ase.Atoms.
    """
    return build_wannier_data(
        structure.recip_lattice(atoms), kpts, mmn, amn, eig, nnkpts, g_vectors,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# numpy .npz interchange
# ---------------------------------------------------------------------------

def save_npz(
    path,
    *,
    kpts:         np.ndarray,
    mmn:          np.ndarray,
    amn:          np.ndarray,
    eig:          np.ndarray,
    nnkpts:       np.ndarray,
    g_vectors:    np.ndarray,
    real_lattice: np.ndarray,
    mp_grid:      tuple[int, int, int],
) -> Path:
    """
    Serialize a full Wannierization problem to a single ``.npz`` file.

    ``real_lattice`` is stored in Bohr and ``eig`` in eV (the on-disk
    convention of this interface); ``load_npz`` returns them unchanged.
    """
    path = Path(path)
    np.savez(
        path,
        kpts=np.asarray(kpts), mmn=np.asarray(mmn), amn=np.asarray(amn),
        eig=np.asarray(eig), nnkpts=np.asarray(nnkpts),
        g_vectors=np.asarray(g_vectors),
        real_lattice=np.asarray(real_lattice),
        mp_grid=np.asarray(mp_grid, dtype=np.int64),
    )
    # np.savez appends .npz if absent; return the actual path written.
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def load_npz(path) -> dict:
    """
    Load a problem saved by ``save_npz`` into a plain dict of numpy arrays
    (keys: kpts, mmn, amn, eig, nnkpts, g_vectors, real_lattice, mp_grid).
    """
    with np.load(Path(path)) as data:
        out = {k: data[k] for k in data.files}
    out["mp_grid"] = tuple(int(x) for x in out["mp_grid"])
    return out


# ---------------------------------------------------------------------------
# ASE-native wannierize entry
# ---------------------------------------------------------------------------

def _warn_on_two_point_directions(mp_grid) -> None:
    """
    Warn when a Monkhorst-Pack direction has only 2 points.

    With N_i = 2 the mesh samples that direction at k_i = 0 and 1/2 only, so
    H(R) keeps a single harmonic (R_i in {-1, 0, +1}, and +1 is the same
    Born-von-Karman image as -1). The interpolation is then EXACT at the two
    endpoints and completely unconstrained between them: at k_i = 1/4 it
    returns the spectrum of (H(0) + H(1/2)) / 2, which is not the mean of the
    two spectra -- level repulsion inside the averaged Hamiltonian shows up as
    a spurious mid-segment excursion. Measured on NiI2's (6,6,2) model
    (c-doubled magnetic cell): 436 meV of deviation from VASP along Gamma-A,
    where the true bands are flat to 43 meV, versus 6-14 meV on every in-plane
    segment.

    N_i = 1 is exempt: that direction is then dispersionless by construction
    (a slab or chain), so there is nothing to interpolate and no artifact.
    """
    bad = [i for i, n in enumerate(np.asarray(mp_grid).ravel()) if int(n) == 2]
    if not bad:
        return
    import warnings
    axes = ", ".join("abc"[i] for i in bad)
    warnings.warn(
        f"mp_grid has only 2 k-points along {axes}: the interpolated bands are "
        f"exact at k=0 and k=1/2 along that direction but unconstrained in "
        f"between (a single Fourier harmonic), and can show a spurious "
        f"mid-segment excursion of several hundred meV. Cross-check against a "
        f"DFT band path before trusting off-mesh quantities along it, or use 4 "
        f"points. See waw.interfaces.ase.driver._warn_on_two_point_directions.",
        stacklevel=2,
    )


def wannierize(
    atoms,
    mp_grid:       tuple[int, int, int],
    kpts:          np.ndarray,
    *,
    mmn:           np.ndarray,
    amn:           np.ndarray,
    eig:           np.ndarray,
    nnkpts:        np.ndarray | None           = None,
    g_vectors:     np.ndarray | None           = None,
    nw:            int | None                 = None,
    outer_window:  tuple[float, float] | None = None,
    frozen_window: tuple[float, float] | None = None,
    proj_min:      float | None                = None,
    proj_max:      float | None                = None,
    dis_spheres:   list[tuple[float, float, float, float]] | None = None,
    dis_spheres_first_wann: int                = 0,
    sitesym:       SiteSymmetry | None         = None,
    slwf_num:       int | None                 = None,
    slwf_constrain: bool                        = False,
    slwf_target_centres: np.ndarray | None      = None,
    slwf_lambda:    float                       = 1.0,
    device:        str | torch.device         = "cpu",
    verbose:       bool                        = True,
    **optim_kwargs,
) -> WannierResult:
    """
    Wannierize from an ``ase.Atoms`` plus numpy overlap arrays.

    ASE-native counterpart of the Wannier90 wrapper: the structure supplies
    the lattice (Angstrom -> Bohr), ``eig`` and the disentanglement windows
    are in eV (converted to atomic units here) before the core driver runs.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure; its cell gives the real/reciprocal lattice.
    mp_grid : (N1, N2, N3)
        Monkhorst-Pack grid (needed to build H(R)).
    kpts : (nk, 3) float64
        k-points in crystal coordinates.
    mmn, amn, eig :
        Overlap arrays (see ``build_wannier_data``).
    nnkpts, g_vectors : optional
        Neighbour topology. If omitted, generated from the mesh via
        ``core.kmesh.generate_nnkp`` -- the ``mmn`` blocks must then be in
        that generated b-vector order.
    nw : int, optional
        Number of Wannier functions. Defaults to ``amn.shape[-1]``.
    outer_window, frozen_window : (E_min, E_max) in eV, optional
        Disentanglement windows. Converted to Hartree before the core runs.
    proj_min, proj_max : float, optional
        Wannier90's `dis_proj_min`/`dis_proj_max` (projectability-based
        disentanglement). Dimensionless; combines with (does not replace)
        outer_window/frozen_window. See ``core.disentangle.disentangle``.
    dis_spheres, dis_spheres_first_wann :
        Wannier90's `dis_spheres`/`dis_spheres_first_wann` (k-space-
        localized disentanglement). `dis_spheres` centres are fractional
        k-points, radii in Bohr^-1. See ``core.disentangle.disentangle``.
    sitesym : core.sitesym.SiteSymmetry, optional (`site_symmetry = .true.`).
        Build with `core.sitesym.site_symmetry_from_dmn(interfaces.
        wannier90.io.read_dmn(...))` (this driver doesn't read files
        itself). `n_restarts`/`n_hops` are forced off together with
        `sitesym`; `guiding_centres`/`guide_refresh` are still forwarded.
        `optimizer` defaults to `'cg'`, the recommended choice for site
        symmetry -- see `core.optim.minimize_spread_symmetrized`.
    slwf_num : int, optional.
        Selectively localized Wannier functions: only the first `slwf_num`
        of `nw` are localized -- see `core.pipeline.wannierize`'s own
        `slwf_num` docstring. Mutually exclusive with `sitesym`.
    slwf_constrain, slwf_target_centres, slwf_lambda :
        Wannier90's `slwf_constrain`/`slwf_centres`/`slwf_lambda`.
        `slwf_target_centres` is given in Angstrom here (unlike
        `core.pipeline.wannierize`'s Bohr), shape `(slwf_num, 3)`,
        converted to Bohr before the core driver runs.
    **optim_kwargs :
        Forwarded to the core driver (n_iter, n_restarts, conv_tol, …).

    Returns
    -------
    WannierResult
        Atomic-unit result (energies Hartree, lengths Bohr).
    """
    _warn_on_two_point_directions(mp_grid)
    recip = structure.recip_lattice(atoms)
    if nnkpts is None or g_vectors is None:
        nnkp = generate_nnkp(np.asarray(kpts, dtype=np.float64), recip, mp_grid)
        nnkpts, g_vectors = nnkp["nnkpts"], nnkp["g_vectors"]
    wdata = build_wannier_data(
        recip, kpts, mmn, amn, eig, nnkpts, g_vectors, device=device,
    )
    if nw is None:
        nw = int(np.asarray(amn).shape[-1])

    def to_ha(window):
        return None if window is None else tuple(EV_TO_HARTREE * e for e in window)

    slwf_target_centres_bohr = (
        None if slwf_target_centres is None
        else torch.tensor(np.asarray(slwf_target_centres, dtype=np.float64) * ANG_TO_BOHR, dtype=torch.float64)
    )

    return _core_wannierize(
        wdata, nw, mp_grid, structure.real_lattice(atoms),
        outer_window  = to_ha(outer_window),
        frozen_window = to_ha(frozen_window),
        proj_min      = proj_min,
        proj_max      = proj_max,
        dis_spheres   = dis_spheres,
        dis_spheres_first_wann = dis_spheres_first_wann,
        sitesym       = sitesym,
        slwf_num       = slwf_num,
        slwf_constrain = slwf_constrain,
        slwf_target_centres = slwf_target_centres_bohr,
        slwf_lambda    = slwf_lambda,
        verbose       = verbose,
        **optim_kwargs,
    )
