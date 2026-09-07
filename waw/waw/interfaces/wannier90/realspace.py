"""
Real-space Wannier functions from UNK files.

UNK files hold the periodic Bloch parts u_nk(r) on the real-space FFT
grid, written by pw2wannier90 with write_unk=.true..  This module
reproduces Wannier90's plot_wannier:

    W_n0(r) = (1/N_k) * sum_k  e^{i k.r} u_nk(r)

where u_nk(r) is first rotated by the disentanglement matrix V (if
entangled) and then by the spread-minimization unitary U -- the same
U/V used to build H(R) in hamiltonian.py -- before the Bloch sum.

After the Bloch sum, the global phase is fixed so each WF is real and
positive at its point of maximum modulus (Wannier90's own convention,
needed since the MLWF gauge leaves each WF undetermined up to an
overall phase).

Limitations relative to Wannier90's plot_wannier:
  - both binary and formatted UNK files (auto-detected by io.read_unk)
  - no spinor / noncollinear support
  - Gaussian cube output (`write_cube`) only implements the "molecule
    mode" full-supercell branch, not the default radius-cropped
    "crystal mode"

Amplitude scale: pw2wannier90's raw UNK values are not L2-normalized
(integral |u_nk|^2 dV over one cell equals the cell volume, not 1).
Wannier90's own plot_wannier does not correct for this either, so the
output here matches its real-space amplitude scale; the returned WF is
not unit-normalized in the continuum sense.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_unk, unk_filename
from ...units import BOHR_TO_ANG


@dataclass
class RealSpaceWF:
    """Real part of one or more Wannier functions on the plotting grid."""
    wf:               np.ndarray   # (n_wf, ngx*ngs[0], ngy*ngs[1], ngz*ngs[2]) float64
    max_im_re_ratio:  np.ndarray   # (n_wf,) float64 -- reality check, see build_wannier_functions
    grid:             tuple[int, int, int]       # (ngx, ngy, ngz)
    supercell:        tuple[int, int, int]
    wann_list:        list[int]                  # 0-based WF indices included


def build_wannier_functions(
    unk_dir: str | Path,
    kpts: np.ndarray,
    U: np.ndarray,
    V: np.ndarray | None = None,
    wann_list: list[int] | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    spin_channel: int = 1,
) -> RealSpaceWF:
    """
    Build real-space MLWFs on the (super)cell FFT grid from UNK files.

    Parameters
    ----------
    unk_dir : directory containing UNK00001.1, UNK00002.1, ... (1-based,
        in the same order as `kpts`).
    kpts    : (nk, 3) k-points in crystal/fractional coordinates
        (e.g. result.wdata.kpts).
    U       : (nk, nw, nw) complex  final spread-minimization unitary
        (result.spread.U_final), U[k][b, w] = coefficient of manifold
        band b in Wannier function w.
    V       : (nk, nb, nw) complex, optional  disentanglement mixing
        matrix (result.dis.V); None for isolated bands (nb == nw).
        V is zero-padded outside the outer window by construction, so
        no separate lwindow mask is needed here.
    wann_list : which WF indices (0-based) to build; None = all num_wann.
    supercell : number of home-cell repeats to tile in each direction
        (Wannier90's wannier_plot_supercell).

    Returns
    -------
    RealSpaceWF
    """
    unk_dir = Path(unk_dir)
    nk = kpts.shape[0]
    nw = U.shape[-1]
    if wann_list is None:
        wann_list = list(range(nw))
    n_wf = len(wann_list)

    grid = None
    wf_accum = None
    ngs = supercell

    for ik in range(nk):
        unk = read_unk(unk_dir / unk_filename(ik + 1, spin_channel))
        ngx, ngy, ngz = unk["ngx"], unk["ngy"], unk["ngz"]
        u_nk = unk["u_nk"]   # (nbnd, ngx, ngy, ngz)

        if grid is None:
            grid = (ngx, ngy, ngz)
            nxx_lo = -(ngs[0] // 2) * ngx
            nxx_hi = ((ngs[0] + 1) // 2) * ngx - 1
            nyy_lo = -(ngs[1] // 2) * ngy
            nyy_hi = ((ngs[1] + 1) // 2) * ngy - 1
            nzz_lo = -(ngs[2] // 2) * ngz
            nzz_hi = ((ngs[2] + 1) // 2) * ngz - 1
            nxx = np.arange(nxx_lo, nxx_hi + 1)
            nyy = np.arange(nyy_lo, nyy_hi + 1)
            nzz = np.arange(nzz_lo, nzz_hi + 1)
            ix = (nxx - 1) % ngx    # 0-based index into the stored UNK grid
            iy = (nyy - 1) % ngy
            iz = (nzz - 1) % ngz
            wf_accum = np.zeros((n_wf, nxx.size, nyy.size, nzz.size), dtype=np.complex128)
        elif (ngx, ngy, ngz) != grid:
            raise ValueError(
                f"UNK grid mismatch at k-point {ik + 1}: expected {grid}, "
                f"got {(ngx, ngy, ngz)}"
            )

        if V is not None:
            if u_nk.shape[0] != V.shape[1]:
                raise ValueError(
                    f"UNK band count {u_nk.shape[0]} != V's num_bands "
                    f"{V.shape[1]} at k-point {ik + 1}"
                )
            r_wvfn = np.tensordot(V[ik], u_nk, axes=(0, 0))       # (nw, ngx,ngy,ngz)
        else:
            if u_nk.shape[0] != nw:
                raise ValueError(
                    f"UNK band count {u_nk.shape[0]} != num_wann {nw} "
                    f"(isolated bands) at k-point {ik + 1}"
                )
            r_wvfn = u_nk

        c_wvfn = np.tensordot(U[ik][:, wann_list], r_wvfn, axes=(0, 0))  # (n_wf, ngx,ngy,ngz)
        u_grid = c_wvfn[:, ix][:, :, iy][:, :, :, iz]                    # (n_wf, |nxx|,|nyy|,|nzz|)

        phase = np.exp(2j * np.pi * (
            kpts[ik, 0] * nxx[:, None, None] / ngx +
            kpts[ik, 1] * nyy[None, :, None] / ngy +
            kpts[ik, 2] * nzz[None, None, :] / ngz -
            kpts[ik, 0] / ngx - kpts[ik, 1] / ngy - kpts[ik, 2] / ngz
        ))
        wf_accum += u_grid * phase[None, :, :, :]

    wf_accum /= nk

    wf_real = np.empty_like(wf_accum, dtype=np.float64)
    max_im_re_ratio = np.empty(n_wf, dtype=np.float64)
    for i in range(n_wf):
        f = wf_accum[i]
        imax = np.argmax(np.abs(f))
        phase0 = f.flat[imax] / abs(f.flat[imax])
        f = f / phase0
        wf_real[i] = f.real
        mask_re = np.abs(f.real) >= 0.01
        max_im_re_ratio[i] = (
            np.max(np.abs(f.imag[mask_re]) / np.abs(f.real[mask_re])) if np.any(mask_re) else 0.0
        )

    return RealSpaceWF(
        wf=wf_real, max_im_re_ratio=max_im_re_ratio,
        grid=grid, supercell=supercell, wann_list=wann_list,
    )


def write_xsf(
    path: str | Path,
    wf: np.ndarray,
    real_lattice_bohr: np.ndarray,
    atom_symbols: list[str],
    atom_pos_cart_bohr: np.ndarray,
    grid: tuple[int, int, int],
    supercell: tuple[int, int, int] = (1, 1, 1),
    header: str = "written by waw",
) -> None:
    """
    Write one real-space Wannier function in XCrySDen .xsf format,
    matching Wannier90's internal_xsf_format (lengths/positions in
    Angstrom, as XSF requires).

    Parameters
    ----------
    wf : (nxx, nyy, nzz) float64  real part of the WF on the plotting
        grid, in the exact nxx/nyy/nzz order produced by
        build_wannier_functions (one entry of RealSpaceWF.wf).
    real_lattice_bohr : (3, 3) real-space lattice rows, Bohr.
    atom_symbols, atom_pos_cart_bohr : atomic symbols and Cartesian
        positions (Bohr) in the home unit cell.
    grid : (ngx, ngy, ngz) FFT grid dimensions (RealSpaceWF.grid).
    supercell : must match what build_wannier_functions used.
    """
    lattice = np.asarray(real_lattice_bohr, dtype=np.float64) * BOHR_TO_ANG
    pos = np.asarray(atom_pos_cart_bohr, dtype=np.float64) * BOHR_TO_ANG
    ngx, ngy, ngz = grid
    ngs = supercell

    x0 = -((ngs[0] // 2) * ngx + 1) / ngx * lattice[0, 0] \
         - ((ngs[1] // 2) * ngy + 1) / ngy * lattice[1, 0] \
         - ((ngs[2] // 2) * ngz + 1) / ngz * lattice[2, 0]
    y0 = -((ngs[0] // 2) * ngx + 1) / ngx * lattice[0, 1] \
         - ((ngs[1] // 2) * ngy + 1) / ngy * lattice[1, 1] \
         - ((ngs[2] // 2) * ngz + 1) / ngz * lattice[2, 1]
    z0 = -((ngs[0] // 2) * ngx + 1) / ngx * lattice[0, 2] \
         - ((ngs[1] // 2) * ngy + 1) / ngy * lattice[1, 2] \
         - ((ngs[2] // 2) * ngz + 1) / ngz * lattice[2, 2]

    fxcry = np.array([
        (ngs[0] * ngx - 1) / ngx,
        (ngs[1] * ngy - 1) / ngy,
        (ngs[2] * ngz - 1) / ngz,
    ])
    dirl = fxcry[:, None] * lattice   # dirl[i, j] = fxcry[i] * lattice[i, j]

    lines = [
        "      #",
        f"      # Generated by waw ({header})",
        "      #",
        "CRYSTAL",
        "PRIMVEC",
        *(f"{v[0]:12.7f}{v[1]:12.7f}{v[2]:12.7f}" for v in lattice),
        "CONVVEC",
        *(f"{v[0]:12.7f}{v[1]:12.7f}{v[2]:12.7f}" for v in lattice),
        "PRIMCOORD",
        f"{len(atom_symbols):6d}  1",
        *(f"{s:2s}   {p[0]:12.7f}{p[1]:12.7f}{p[2]:12.7f}"
          for s, p in zip(atom_symbols, pos)),
        "",
        "BEGIN_BLOCK_DATAGRID_3D",
        "3D_field",
        "BEGIN_DATAGRID_3D_UNKNOWN",
        f"{ngs[0] * ngx:6d}{ngs[1] * ngy:6d}{ngs[2] * ngz:6d}",
        f"{x0:12.6f}{y0:12.6f}{z0:12.6f}",
        f"{dirl[0, 0]:12.7f}{dirl[0, 1]:12.7f}{dirl[0, 2]:12.7f}",
        f"{dirl[1, 0]:12.7f}{dirl[1, 1]:12.7f}{dirl[1, 2]:12.7f}",
        f"{dirl[2, 0]:12.7f}{dirl[2, 1]:12.7f}{dirl[2, 2]:12.7f}",
    ]

    # Fortran's implied-do ((val, nx=..), ny=..), nz=..) varies nx fastest,
    # then ny, then nz -- i.e. flatten wf (nxx,nyy,nzz) in Fortran order.
    flat = np.asarray(wf, dtype=np.float64).flatten(order="F")
    for start in range(0, flat.size, 6):
        chunk = flat[start:start + 6]
        lines.append("".join(f"{v:13.5E}" for v in chunk))

    lines += ["END_DATAGRID_3D", "END_BLOCK_DATAGRID_3D"]

    Path(path).write_text("\n".join(lines) + "\n")


def write_cube(
    path: str | Path,
    wf: np.ndarray,
    real_lattice_bohr: np.ndarray,
    atom_symbols: list[str],
    atom_pos_cart_bohr: np.ndarray,
    grid: tuple[int, int, int],
    supercell: tuple[int, int, int] = (1, 1, 1),
    header: str = "written by waw",
) -> None:
    """
    Write one real-space Wannier function in Gaussian .cube format
    (Wannier90 `wannier_plot_format=cube`), for viewers that don't read
    XSF (e.g. VESTA). Implements only the "molecule mode"
    (`wannier_plot_mode=molecule`) branch: the full supercell grid
    `write_xsf` already builds is written as one cube spanning the whole
    tiled supercell, atoms listed in their home-cell positions as-is.

    Not implemented: the default "crystal mode" per-Wannier-centre
    radius-cropped cube (`wannier_plot_radius`/`_scale`). The
    full-supercell cube written here is a strict superset of that data;
    any cube viewer opens it identically.

    Parameters
    ----------
    wf : (nxx, nyy, nzz) float64  real part of the WF on the plotting
        grid, in the exact nxx/nyy/nzz order produced by
        build_wannier_functions (one entry of RealSpaceWF.wf) -- same
        array `write_xsf` takes, just a different output format.
    real_lattice_bohr : (3, 3) real-space lattice rows, Bohr.
    atom_symbols, atom_pos_cart_bohr : atomic symbols and Cartesian
        positions (Bohr) in the home unit cell.
    grid : (ngx, ngy, ngz) FFT grid dimensions (RealSpaceWF.grid).
    supercell : must match what build_wannier_functions used.
    """
    lattice = np.asarray(real_lattice_bohr, dtype=np.float64)
    pos = np.asarray(atom_pos_cart_bohr, dtype=np.float64)
    ngx, ngy, ngz = grid
    ngs = supercell

    # Same supercell-centred origin formula as write_xsf, but in Bohr (cube
    # format's own native unit -- no BOHR_TO_ANG here).
    origin = np.zeros(3)
    for i in range(3):
        origin[i] = (
            -((ngs[0] // 2) * ngx + 1) / ngx * lattice[0, i]
            - ((ngs[1] // 2) * ngy + 1) / ngy * lattice[1, i]
            - ((ngs[2] // 2) * ngz + 1) / ngz * lattice[2, i]
        )

    atomic_z = [_ELEMENT_Z[s.lower()] for s in atom_symbols]

    lines = [
        "Generated by waw (write_cube)",
        header,
        f"{len(atom_symbols):5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}",
        f"{ngs[0] * ngx:5d}{lattice[0, 0] / ngx:12.6f}{lattice[0, 1] / ngx:12.6f}{lattice[0, 2] / ngx:12.6f}",
        f"{ngs[1] * ngy:5d}{lattice[1, 0] / ngy:12.6f}{lattice[1, 1] / ngy:12.6f}{lattice[1, 2] / ngy:12.6f}",
        f"{ngs[2] * ngz:5d}{lattice[2, 0] / ngz:12.6f}{lattice[2, 1] / ngz:12.6f}{lattice[2, 2] / ngz:12.6f}",
    ]
    for z, p in zip(atomic_z, pos):
        lines.append(f"{z:5d}{1.0:12.6f}{p[0]:12.6f}{p[1]:12.6f}{p[2]:12.6f}")

    # Cube's own nested loop order (nxx outer, nyy middle, nzz fastest) is
    # exactly numpy's default C order for a (nxx,nyy,nzz)-shaped array --
    # unlike write_xsf, no Fortran-order flatten needed here.
    flat = np.asarray(wf, dtype=np.float64).flatten(order="C")
    for start in range(0, flat.size, 6):
        chunk = flat[start:start + 6]
        lines.append("".join(f"{v:13.5E}" for v in chunk))

    Path(path).write_text("\n".join(lines) + "\n")


_ELEMENT_Z = {
    "h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8, "f": 9, "ne": 10,
    "na": 11, "mg": 12, "al": 13, "si": 14, "p": 15, "s": 16, "cl": 17, "ar": 18,
    "k": 19, "ca": 20, "sc": 21, "ti": 22, "v": 23, "cr": 24, "mn": 25, "fe": 26,
    "co": 27, "ni": 28, "cu": 29, "zn": 30, "ga": 31, "ge": 32, "as": 33, "se": 34,
    "br": 35, "kr": 36, "rb": 37, "sr": 38, "y": 39, "zr": 40, "nb": 41, "mo": 42,
    "tc": 43, "ru": 44, "rh": 45, "pd": 46, "ag": 47, "cd": 48, "in": 49, "sn": 50,
    "sb": 51, "te": 52, "i": 53, "xe": 54, "cs": 55, "ba": 56, "la": 57,
    "hf": 72, "ta": 73, "w": 74, "re": 75, "os": 76, "ir": 77, "pt": 78, "au": 79,
    "hg": 80, "tl": 81, "pb": 82, "bi": 83,
}


def plot_wannier_functions(
    seedname: str | Path,
    result,
    unk_dir: str | Path | None = None,
    wann_list: list[int] | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    spin_channel: int = 1,
    out_dir: str | Path | None = None,
    format: str = "xsf",
) -> list[Path]:
    """
    Convenience wrapper: read UNK files + <seedname>.win, build the
    requested real-space MLWFs, and write one file per WF, named like
    Wannier90's own wannier_plot output (<seedname>_00001.xsf/.cube, ...
    1-based WF index).

    Parameters
    ----------
    seedname : path prefix, used to find <seedname>.win (for the
        lattice/atoms) and, if unk_dir is None, as the default
        directory to look for UNK files in.
    result   : the WannierResult from wannierize() for this seedname.
    unk_dir  : directory containing the UNK files, if different from
        seedname's parent directory.
    wann_list : which WF indices (0-based) to plot; None = all.
    out_dir  : directory to write the files to; None = same
        directory as seedname.
    format   : 'xsf' (XCrySDen, default) or 'cube' (Gaussian cube; see
        `write_cube`'s docstring for the "molecule mode only"
        simplification).

    Returns the list of file paths written.
    """
    from .io import read_win
    from .loader import parse_real_lattice, parse_atoms

    if format not in ("xsf", "cube"):
        raise ValueError(f"format must be 'xsf' or 'cube', got {format!r}")

    seed_path = Path(seedname)
    win_path = seed_path.with_suffix(".win")
    win = read_win(win_path)
    lattice = parse_real_lattice(win)
    symbols, pos = parse_atoms(win)

    kpts = result.wdata.kpts.detach().cpu().numpy()
    U = result.spread.U_final.detach().cpu().numpy()
    V = result.dis.V.detach().cpu().numpy() if result.dis is not None else None

    rs = build_wannier_functions(
        unk_dir if unk_dir is not None else seed_path.parent,
        kpts, U, V=V, wann_list=wann_list,
        supercell=supercell, spin_channel=spin_channel,
    )

    out_dir = Path(out_dir) if out_dir is not None else seed_path.parent
    writer = write_cube if format == "cube" else write_xsf
    paths = []
    for i, w in enumerate(rs.wann_list):
        path = out_dir / f"{seed_path.name}_{w + 1:05d}.{format}"
        writer(path, rs.wf[i], lattice, symbols, pos, rs.grid, supercell=supercell)
        paths.append(path)
    return paths
