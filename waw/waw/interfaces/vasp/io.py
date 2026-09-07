"""
Direct-input VASP driver: low-level VASP I/O.

Writes POSCAR / INCAR / KPOINTS / POTCAR (+ a ``wannier90.win`` projection
block) from an ``ase.Atoms`` plus explicit parameter dicts, runs VASP under
MPI, and reads back the k-points, Fermi energy, band eigenvalues, and the
Wannier90 overlaps VASP produces via its VASP2WANNIER90 interface
(``LWANNIER90``). The overlap files (``wannier90.mmn/.amn/.eig``) are the
code-agnostic hand-off to ``waw.core``/``waw.interfaces.wannier90`` -- waw does
its own disentanglement + MLWF, exactly as in the Quantum ESPRESSO path.

This VASP (6.6.1) is built against OpenMPI (Intel MPI fails ``MPI_Init`` on the
target node), so it launches like the QE OpenMPI runs:
``mpirun --mca pml ob1 --bind-to none -np N vasp_ncl``. The caller is expected
to have the build's runtime modules loaded (openmpi + MKL), same convention as
the QE interface expecting its module loaded.

Noncollinear/SOC uses ``vasp_ncl``; magnetic sublattices are set by per-atom
``MAGMOM`` (one POTCAR species per element -- unlike the QE path's split
``Ni1``/``Ni2`` species). Wannier neighbour topology (``nnkpts``/``g_vectors``)
is taken directly from the ``.mmn`` block headers, and the k-points from
``IBZKPT`` -- both in VASP's own ordering (which differs from
``monkhorst_pack``'s), so the overlap k/b indexing stays self-consistent.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from ase.io import write as ase_write

from ..wannier90.io import read_mmn, read_amn, read_eig

# The OpenMPI-built VASP 6.6.1 with the VASP2WANNIER90v2 interface.
DEFAULT_VASP_BIN_DIR = Path.home() / "software/source/vasp.6.6.1/bin"
DEFAULT_POTCAR_DIR = Path.home() / "MHM_master/pseudos/matpes/potpaw_PBE"


def mpirun_prefix(ncores: int) -> list[str]:
    return ["mpirun", "--mca", "pml", "ob1", "--bind-to", "none", "-np", str(ncores)]


def _vasp_env() -> dict:
    """Environment for the VASP subprocess: pure MPI, ONE thread per rank.

    Forces the OpenMP/BLAS thread envs to 1 so VASP's ``-D_OPENMP`` threads do
    NOT multiply with the ``-np N`` MPI ranks (an inherited
    ``waw.set_num_threads`` would otherwise give N*N-way oversubscription).
    """
    return {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "OMP_STACKSIZE": "512m"}


# ---------------------------------------------------------------------------
# Input writers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, bool):
        return ".TRUE." if v else ".FALSE."
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(_fmt(x) for x in v)
    return str(v)


def write_poscar(atoms, path) -> tuple[list[str], list[int], list[int]]:
    """Write a VASP5 POSCAR (species grouped/sorted). Returns (species_order,
    counts, sort_index) where sort_index maps POSCAR atom order -> the input
    ``atoms`` index (so MAGMOM etc. can be built in POSCAR order)."""
    path = Path(path)
    ase_write(path, atoms, format="vasp", sort=True, vasp5=True, direct=True)
    lines = path.read_text().splitlines()
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
    return species, counts, list(order)


def write_potcar(species: list[str], potcar_dir, path) -> Path:
    """Concatenate per-species ``POTCAR``s (in POSCAR species order)."""
    potcar_dir, path = Path(potcar_dir), Path(path)
    blob = b""
    for s in species:
        blob += (potcar_dir / s / "POTCAR").read_bytes()
    path.write_bytes(blob)
    return path


def write_kpoints(mp_grid: tuple[int, int, int], path, shift=(0, 0, 0)) -> Path:
    path = Path(path)
    path.write_text(
        "auto mesh\n0\nGamma\n"
        f"{mp_grid[0]} {mp_grid[1]} {mp_grid[2]}\n"
        f"{shift[0]} {shift[1]} {shift[2]}\n"
    )
    return path


def write_kpoints_explicit(kpts, path, weights=None) -> Path:
    """Explicit reciprocal k-list (for a non-self-consistent bands run along a
    path; ICHARG=11). ``kpts`` is (nk, 3) fractional. Weights default to 1 --
    VASP rejects an all-zero-weight KPOINTS ('sum of weights is zero'); with a
    fixed charge density (ICHARG=11) the weights don't affect the eigenvalues."""
    kpts = np.asarray(kpts, float)
    w = weights if weights is not None else [1] * len(kpts)
    path = Path(path)
    lines = ["explicit k-path", str(len(kpts)), "Reciprocal"]
    lines += [f" {k[0]:.10f} {k[1]:.10f} {k[2]:.10f} {wi}" for k, wi in zip(kpts, w)]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_incar(incar: dict, path) -> Path:
    path = Path(path)
    text = "".join(f"{k} = {_fmt(v)}\n" for k, v in incar.items())
    path.write_text(text)
    return path


def write_wannier_win(projections: list[str], num_wann: int, path,
                      spinors: bool = True) -> Path:
    """Pre-write ``wannier90.win`` with the projection block + num_wann, so VASP
    projects the ``.amn`` onto our orbitals (not an auto template). ``num_iter``
    /``dis_num_iter`` are 0: VASP only writes overlaps, waw minimizes."""
    path = Path(path)
    text = f"num_wann = {num_wann}\n"
    if spinors:
        text += "spinors = .true.\n"
    text += "num_iter = 0\ndis_num_iter = 0\n\nbegin projections\n"
    text += "\n".join(projections) + "\nend projections\n"
    path.write_text(text)
    return path


def noncollinear_magmom(atoms, moments: dict, sort_index: list[int],
                        axis=(0.0, 0.0, 1.0)) -> list[float]:
    """Build the noncollinear MAGMOM list (3 numbers per atom) in POSCAR order.

    ``moments`` maps an ASE tag (int) or element symbol to a signed moment
    magnitude along ``axis``; unspecified atoms get 0. ``sort_index`` is
    ``write_poscar``'s POSCAR->atoms map.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    syms = atoms.get_chemical_symbols()
    tags = atoms.get_tags()
    out = []
    for i in sort_index:
        m = moments.get(int(tags[i]), moments.get(syms[i], 0.0))
        out.extend((m * axis).tolist())
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_vasp(workdir, *, binary=None, noncollinear=True, ncores: int = 16) -> Path:
    """Run VASP under MPI in ``workdir``. ``binary`` defaults to ``vasp_ncl``
    (noncollinear) or ``vasp_std`` from ``DEFAULT_VASP_BIN_DIR``. Assumes the
    build's runtime modules (openmpi + MKL) are loaded in the environment."""
    workdir = Path(workdir)
    if binary is None:
        binary = DEFAULT_VASP_BIN_DIR / ("vasp_ncl" if noncollinear else "vasp_std")
    out = workdir / "vasp.stdout"
    with open(out, "w") as fh:
        subprocess.run(mpirun_prefix(ncores) + [str(binary)],
                       cwd=workdir, stdout=fh, stderr=subprocess.STDOUT,
                       env=_vasp_env(), check=True)
    from waw.utils.runs import autostamp
    autostamp(workdir, code="vasp", settings={"ncores": ncores,
                                              "binary": Path(binary).name,
                                              "noncollinear": noncollinear})
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_fermi_energy(workdir) -> float:
    """Fermi energy (eV) from OUTCAR (last 'E-fermi :' line)."""
    outcar = Path(workdir) / "OUTCAR"
    ef = None
    for line in outcar.read_text().splitlines():
        if "E-fermi" in line:
            ef = float(line.split("E-fermi")[1].split()[1])
    if ef is None:
        raise ValueError(f"no E-fermi in {outcar}")
    return ef


def read_kpoints(workdir) -> np.ndarray:
    """(nk, 3) k-points in fractional (reciprocal) coords, in VASP's ordering
    (matching the .mmn/.eig k-index), read from IBZKPT."""
    lines = (Path(workdir) / "IBZKPT").read_text().splitlines()
    nk = int(lines[1].split()[0])
    kpts = np.array([[float(x) for x in lines[3 + i].split()[:3]] for i in range(nk)])
    return kpts


def read_overlaps(workdir) -> dict:
    """Load VASP's Wannier90 overlaps into the dict `wannierize` consumes.

    ``nnkpts``/``g_vectors`` come straight from the ``.mmn`` block headers
    (VASP/wannier90's own b-ordering), so no separate ``.nnkp`` is needed.
    """
    workdir = Path(workdir)
    mmn, kpb = read_mmn(workdir / "wannier90.mmn")   # kpb: (ik, ik2, g1,g2,g3) per (k,b)
    nk, nnb = mmn.shape[0], mmn.shape[1]
    kpb = np.asarray(kpb, dtype=np.int64)
    nnkpts = kpb[:, 1].reshape(nk, nnb)
    g_vectors = kpb[:, 2:5].reshape(nk, nnb, 3)
    return {
        "mmn": mmn,
        "amn": read_amn(workdir / "wannier90.amn"),
        "eig": read_eig(workdir / "wannier90.eig"),
        "nnkpts": nnkpts,
        "g_vectors": g_vectors,
    }


def read_bands(workdir, nbnd: int | None = None) -> np.ndarray:
    """Band eigenvalues (eV), (nk, nbnd), from EIGENVAL -- for the DFT-vs-Wannier
    overlay after a bands run (ICHARG=11 along an explicit k-path). Noncollinear
    EIGENVAL has one energy column."""
    lines = (Path(workdir) / "EIGENVAL").read_text().splitlines()
    nk, nb = int(lines[5].split()[1]), int(lines[5].split()[2])
    bands = np.zeros((nk, nb))
    i = 6
    for ik in range(nk):
        while lines[i].strip() == "":
            i += 1
        i += 1  # k-point coordinate line
        for ib in range(nb):
            bands[ik, ib] = float(lines[i].split()[1])
            i += 1
    return bands[:, :nbnd] if nbnd else bands
