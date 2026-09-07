"""
Wannier90 loader: read .win/.nnkp/.eig/.mmn/.amn into a core WannierData.

Boundary between the Wannier90 file interface and the atomic-unit core:
reads raw numpy arrays via .io, converts units (Angstrom -> Bohr, eV ->
Hartree), computes k-mesh geometry via core.kmesh, and packs everything
into a core.types.WannierData.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...core.types import WannierData
from ...core.kmesh import _compute_bvecs_and_weights, _nnkp_from_mmn
from ...units import ANG_TO_BOHR, EV_TO_HARTREE
from .io import read_win, read_nnkp, read_eig, read_mmn, read_amn, resolve_input_path


def load(
    seedname: str | Path,
    recip_lattice: np.ndarray,
    device: str | torch.device = "cpu",
    dtype_real: torch.dtype = torch.float64,
    dtype_complex: torch.dtype = torch.complex128,
    params: dict | None = None,
    select_projections: np.ndarray | None = None,
) -> WannierData:
    """
    Load all Wannier90 input files for a given seedname and return a
    WannierData instance with everything as PyTorch tensors.

    Args:
      seedname     : path prefix (e.g. "silicon" reads silicon.win, etc.)
      recip_lattice: (3, 3) reciprocal lattice vectors as rows, in Bohr^-1.
                     Required to convert b-vectors from crystal to Cartesian.
      device       : PyTorch device ("cpu" or "cuda")
      dtype_real   : floating-point dtype for real tensors
      dtype_complex: complex dtype for overlap/projection matrices
      params       : pre-parsed .win parameters (from read_win).  When provided,
                     the .win file is not re-read; pass this from the caller that
                     already has params to avoid parsing the file twice.
      select_projections: 0-based column indices into the .amn's trial-
                     projection axis, when the .amn was written with more
                     projections declared than num_wann (W90's
                     `select_projections` keyword picks a subset of them to
                     seed the SVD initial guess). None keeps Amn as-is (the
                     usual case: num_proj == num_wann already).

    The reciprocal lattice is not stored in any of the standard input files
    read here; it must be extracted from .win's unit_cell_cart block by the
    caller.  See parse_recip_lattice() below.

    If .nnkp is absent, k-neighbor information is reconstructed from the .mmn
    file header.  If .eig is absent (isolated bands, nb == nw), eigenvalues
    default to zero (only H(R) accuracy is affected; spread minimization is not).

    Eigenvalues are converted from eV to Hartree.
    """
    seed = Path(seedname)

    # ---- read raw numpy arrays from disk ----------------------------------
    if params is None:
        params = read_win(seed.with_suffix(".win"))
    Mmn_np, kpb_map = read_mmn(seed.with_suffix(".mmn"))
    Amn_np          = read_amn(seed.with_suffix(".amn"))
    if select_projections is not None:
        Amn_np = Amn_np[:, :, select_projections]

    nk, nnb, nb, _ = Mmn_np.shape

    # .nnkp: use file if present (plain or .gz), else reconstruct from .mmn + .win
    nnkp_path = seed.with_suffix(".nnkp")
    if resolve_input_path(nnkp_path).exists():
        nnkp = read_nnkp(nnkp_path)
    else:
        nnkp = _nnkp_from_mmn(kpb_map, params, nk, nnb)

    # .eig: optional for isolated bands (nb == nw); default to zeros.
    eig_path = seed.with_suffix(".eig")
    if resolve_input_path(eig_path).exists():
        eig_np = read_eig(eig_path) * EV_TO_HARTREE
    else:
        eig_np = np.zeros((nk, nb), dtype=np.float64)

    # ---- compute b-vectors and shell weights ------------------------------
    bvecs_np, wb_np = _compute_bvecs_and_weights(
        kpts          = nnkp["kpoints"],
        nnkpts        = nnkp["nnkpts"],
        g_vectors     = nnkp["g_vectors"],
        recip_lattice = recip_lattice,
    )

    # ---- pack into PyTorch tensors ----------------------------------------
    def to_real(arr):
        return torch.tensor(arr, dtype=dtype_real,    device=device)

    def to_cplx(arr):
        return torch.tensor(arr, dtype=dtype_complex, device=device)

    return WannierData(
        Mmn    = to_cplx(Mmn_np),
        Amn    = to_cplx(Amn_np),
        eig    = to_real(eig_np),
        kpts   = to_real(nnkp["kpoints"]),
        bvecs  = to_real(bvecs_np),
        wb     = to_real(wb_np),
        kb_idx = torch.tensor(nnkp["nnkpts"], dtype=torch.long, device=device),
        params = params,
    )


# ---------------------------------------------------------------------------
# Lattice parsers (convenience, used before calling load())
# ---------------------------------------------------------------------------

def _parse_lattice_block(win_params: dict) -> np.ndarray:
    """
    Parse unit_cell_cart block -> real-space lattice rows in Bohr.

    An optional first line names the units ('ang'/'angstrom' or 'bohr').
    When absent, the default is Angstrom, matching Wannier90's own
    convention for a bare block.
    """
    cell_lines = win_params.get("unit_cell_cart", [])
    if not cell_lines:
        raise ValueError("unit_cell_cart block missing from .win file")
    unit = "ang"
    data_lines = cell_lines
    first = cell_lines[0].strip().lower()
    if first in ("ang", "angstrom"):
        unit       = "ang"
        data_lines = cell_lines[1:]
    elif first == "bohr":
        unit       = "bohr"
        data_lines = cell_lines[1:]
    if len(data_lines) < 3:
        raise ValueError("unit_cell_cart block must contain 3 lattice vectors")
    lattice = np.array(
        [[float(x) for x in line.split()[:3]] for line in data_lines[:3]],
        dtype=np.float64,
    )
    if unit == "ang":
        lattice *= ANG_TO_BOHR
    return lattice


def parse_real_lattice(win_params: dict) -> np.ndarray:
    """
    Extract the real-space lattice matrix (rows = a1, a2, a3) in Bohr
    from the parsed .win parameter dict.
    """
    return _parse_lattice_block(win_params)


def parse_recip_lattice(win_params: dict) -> np.ndarray:
    """
    Extract the reciprocal lattice matrix (rows = b1, b2, b3) in Bohr^-1
    from the parsed .win parameter dict.
    """
    lattice = _parse_lattice_block(win_params)
    return 2 * np.pi * np.linalg.inv(lattice).T


def parse_atoms(win_params: dict) -> tuple[list[str], np.ndarray]:
    """
    Extract atomic symbols and Cartesian positions (Bohr) from the .win
    file's atoms_frac or atoms_cart block.

    atoms_frac positions are converted to Cartesian via the unit_cell_cart
    block (also required in that case).  atoms_cart may carry an optional
    leading 'bohr'/'ang' unit line; when absent the default is Angstrom,
    same convention as unit_cell_cart (see _parse_lattice_block).
    """
    if "atoms_frac" in win_params:
        lattice = parse_real_lattice(win_params)
        symbols, frac = [], []
        for line in win_params["atoms_frac"]:
            parts = line.split()
            symbols.append(parts[0])
            frac.append([float(x) for x in parts[1:4]])
        cart = np.array(frac, dtype=np.float64) @ lattice
        return symbols, cart

    if "atoms_cart" in win_params:
        lines = win_params["atoms_cart"]
        unit = "ang"
        data_lines = lines
        if lines and lines[0].strip().lower() in ("ang", "angstrom", "bohr"):
            unit = "bohr" if lines[0].strip().lower() == "bohr" else "ang"
            data_lines = lines[1:]
        symbols, cart = [], []
        for line in data_lines:
            parts = line.split()
            symbols.append(parts[0])
            cart.append([float(x) for x in parts[1:4]])
        cart = np.array(cart, dtype=np.float64)
        if unit == "ang":
            cart *= ANG_TO_BOHR
        return symbols, cart

    raise ValueError("Neither atoms_frac nor atoms_cart block found in .win file")
