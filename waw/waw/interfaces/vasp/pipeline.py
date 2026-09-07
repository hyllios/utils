"""
VASP -> Wannier90 overlap generation, mirroring
``waw.interfaces.quantum_espresso.pipeline.generate_overlaps``.

Runs a single VASP SCF with ``LWANNIER90`` (VASP2WANNIER90v2) to write the
``wannier90.mmn/.amn/.eig`` overlaps, then returns them (plus k-points and the
Fermi energy) in the exact dict shape ``waw.interfaces.ase.driver.wannierize``
consumes -- so a notebook can swap ``qe.generate_overlaps`` for
``vasp.generate_overlaps`` and leave everything downstream unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io as vio


def _ldau_arrays(species, ldau):
    """LDAUL/LDAUU/LDAUJ lists in POSCAR species order. ``ldau`` maps an element
    to ``{"L":.., "U":.., "J":..}``; others get L=-1 (no +U)."""
    L, U, J = [], [], []
    for s in species:
        d = ldau.get(s)
        if d:
            L.append(int(d["L"])); U.append(float(d.get("U", 0.0))); J.append(float(d.get("J", 0.0)))
        else:
            L.append(-1); U.append(0.0); J.append(0.0)
    return L, U, J


def _finished(workdir) -> bool:
    outcar = Path(workdir) / "OUTCAR"
    if not outcar.exists():
        return False
    tail = outcar.read_text()[-4000:]
    done = ("General timing and accounting" in tail) or ("reached required accuracy" in tail)
    files = all((Path(workdir) / f).exists()
                for f in ("wannier90.mmn", "wannier90.amn", "wannier90.eig", "IBZKPT"))
    return done and files


def generate_overlaps(
    atoms, mp_grid, workdir, *,
    projections,
    num_wann: int,
    encut: float = 520.0,
    nbands: int | None = None,
    moments: dict | None = None,
    ldau: dict | None = None,
    ivdw: int | None = 11,
    noncollinear: bool = True,
    lsorbit: bool = True,
    spinors: bool | None = None,
    ncores: int = 16,
    potcar_dir=vio.DEFAULT_POTCAR_DIR,
    vasp_bin=None,
    incar_extra: dict | None = None,
    rerun: bool = True,
) -> dict:
    """Run VASP (LWANNIER90) and return the overlaps for ``wannierize``.

    Returns a dict with keys ``kpts, mmn, amn, eig, nnkpts, g_vectors,
    fermi_energy`` -- same as ``qe.generate_overlaps``. ``moments`` (for
    noncollinear MAGMOM) and ``ldau`` (per-element ``{"L","U","J"}``,
    Liechtenstein LDAUTYPE=1) map the QE notebook's magnetism/DFT+U to VASP
    tags. ``rerun=False`` reuses a finished run in ``workdir``.
    """
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    moments = moments or {}
    ldau = ldau or {}
    if spinors is None:
        spinors = noncollinear or lsorbit

    if rerun or not _finished(workdir):
        species, counts, sort_index = vio.write_poscar(atoms, workdir / "POSCAR")
        vio.write_potcar(species, potcar_dir, workdir / "POTCAR")
        vio.write_kpoints(tuple(mp_grid), workdir / "KPOINTS")
        vio.write_wannier_win(projections, num_wann, workdir / "wannier90.win", spinors=spinors)

        incar = {
            "SYSTEM": "waw vasp overlaps",
            "ISTART": 0, "ENCUT": encut, "PREC": "Accurate",
            "EDIFF": 1e-6, "NELM": 200,
            "ISMEAR": 0, "SIGMA": 0.05,
            "ISYM": -1,                         # full mesh for Wannier
            "LWANNIER90": True, "LWRITE_MMN_AMN": True, "NUM_WANN": num_wann,
        }
        if nbands:
            incar["NBANDS"] = nbands
        if noncollinear:
            incar["LNONCOLLINEAR"] = True
        if lsorbit:
            incar["LSORBIT"] = True
        if moments:
            incar["MAGMOM"] = vio.noncollinear_magmom(atoms, moments, sort_index)
        if ldau:
            L, U, J = _ldau_arrays(species, ldau)
            incar.update({"LDAU": True, "LDAUTYPE": 1, "LDAUL": L, "LDAUU": U,
                          "LDAUJ": J, "LMAXMIX": 4})
        if ivdw:
            incar["IVDW"] = ivdw
        incar.update(incar_extra or {})
        vio.write_incar(incar, workdir / "INCAR")

        vio.run_vasp(workdir, binary=vasp_bin, noncollinear=noncollinear, ncores=ncores)

    ov = vio.read_overlaps(workdir)
    ov["kpts"] = vio.read_kpoints(workdir)
    ov["fermi_energy"] = vio.read_fermi_energy(workdir)
    return ov


def bands(atoms, scf_workdir, kpts_frac, *,
          encut: float = 520.0, nbands: int | None = None,
          moments: dict | None = None, ldau: dict | None = None,
          ivdw: int | None = 11, noncollinear: bool = True, lsorbit: bool = True,
          ncores: int = 16, potcar_dir=vio.DEFAULT_POTCAR_DIR, vasp_bin=None,
          incar_extra: dict | None = None, rerun: bool = True) -> np.ndarray:
    """Non-self-consistent band eigenvalues (eV), (nk, nbnd), along an explicit
    ``kpts_frac`` path -- for the DFT-vs-Wannier overlay. Reads the SCF charge
    density (``ICHARG=11``) from a completed ``generate_overlaps`` run in
    ``scf_workdir``. Runs in ``scf_workdir/bands/``.
    """
    scf_workdir = Path(scf_workdir)
    bdir = scf_workdir / "bands"; bdir.mkdir(parents=True, exist_ok=True)
    done = (bdir / "EIGENVAL").exists() and \
        "General timing" in ((bdir / "OUTCAR").read_text()[-3000:] if (bdir / "OUTCAR").exists() else "")
    if rerun or not done:
        moments = moments or {}; ldau = ldau or {}
        species, _, sort_index = vio.write_poscar(atoms, bdir / "POSCAR")
        vio.write_potcar(species, potcar_dir, bdir / "POTCAR")
        vio.write_kpoints_explicit(kpts_frac, bdir / "KPOINTS")
        # reuse the SCF charge density
        chg = scf_workdir / "CHGCAR"
        if chg.exists():
            (bdir / "CHGCAR").unlink(missing_ok=True)
            (bdir / "CHGCAR").symlink_to(chg.resolve())
        incar = {"SYSTEM": "waw vasp bands", "ISTART": 1, "ICHARG": 11,
                 "ENCUT": encut, "PREC": "Accurate", "ISMEAR": 0, "SIGMA": 0.05,
                 "ISYM": -1, "LORBIT": 11}
        if nbands:
            incar["NBANDS"] = nbands
        if noncollinear:
            incar["LNONCOLLINEAR"] = True
        if lsorbit:
            incar["LSORBIT"] = True
        if moments:
            incar["MAGMOM"] = vio.noncollinear_magmom(atoms, moments, sort_index)
        if ldau:
            L, U, J = _ldau_arrays(species, ldau)
            incar.update({"LDAU": True, "LDAUTYPE": 1, "LDAUL": L, "LDAUU": U,
                          "LDAUJ": J, "LMAXMIX": 4})
        if ivdw:
            incar["IVDW"] = ivdw
        incar.update(incar_extra or {})
        vio.write_incar(incar, bdir / "INCAR")
        vio.run_vasp(bdir, binary=vasp_bin, noncollinear=noncollinear, ncores=ncores)
    return vio.read_bands(bdir, nbnd=nbands)
