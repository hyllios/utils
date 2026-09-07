"""
SIESTA driver: fdf input writer + runner + output parsing.

SIESTA's numerical-atomic-orbital (NAO) basis gives a genuinely localized
real-space Hamiltonian directly -- no Wannierization step. This module
runs SIESTA (module siesta/5.4.1 on this cluster) and `loader.py` turns
the resulting H(R)/S(R) into a waw `HamiltonianR` via Löwdin
orthogonalization. Pseudopotentials: PSML (PseudoDojo), one `<El>.psml`
per species in `pseudo_dir` (repo copies in `workflows/pseudos/psml/`).
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import warnings
from pathlib import Path

import numpy as np


def write_fdf(
    path,
    atoms,
    *,
    label: str,
    pseudo_dir,
    kgrid: tuple[int, int, int],
    spin: str = "polarized",
    basis: str = "DZP",
    mesh_cutoff_ry: float = 400.0,
    xc: str = "PBE",
    initial_moments: dict | None = None,
    electronic_temperature_ev: float = 0.025,
    extra: dict | None = None,
) -> Path:
    """
    Write a SIESTA .fdf input for `atoms` (ASE).

    Args:
      label           : SystemLabel (output files are <label>.*)
      pseudo_dir      : directory holding <El>.psml files
      kgrid           : Monkhorst-Pack grid for the scf
      spin            : 'non-polarized' | 'polarized' | 'spin-orbit'
      basis           : PAO.BasisSize (SZ/DZ/SZP/DZP/TZP)
      initial_moments : {atom_index: moment} initial spins (DM.InitSpin);
                        by ASE tag name is NOT supported -- plain indices
      extra           : verbatim extra fdf lines, {key: value}

    Always sets `SaveHS T` so the H/S file (HSX with SIESTA 5.x; H and S in the NAO basis)
    is written for `loader.load_hamiltonian`.
    """
    path = Path(path)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols), key=symbols.index)
    from ase.data import atomic_numbers

    lines = [
        f"SystemName   {label}",
        f"SystemLabel  {label}",
        "",
        f"NumberOfAtoms    {len(atoms)}",
        f"NumberOfSpecies  {len(species)}",
        "",
        "%block ChemicalSpeciesLabel",
    ]
    for i, s in enumerate(species):
        lines.append(f"  {i + 1}  {atomic_numbers[s]}  {s}")
    lines += ["%endblock ChemicalSpeciesLabel", ""]

    cell = atoms.get_cell()[:]
    lines += ["LatticeConstant 1.0 Ang", "%block LatticeVectors"]
    for v in cell:
        lines.append(f"  {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}")
    lines += ["%endblock LatticeVectors", ""]

    lines += ["AtomicCoordinatesFormat Fractional",
              "%block AtomicCoordinatesAndAtomicSpecies"]
    for pos, s in zip(atoms.get_scaled_positions(), symbols):
        lines.append(f"  {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}  {species.index(s) + 1}")
    lines += ["%endblock AtomicCoordinatesAndAtomicSpecies", ""]

    lines += [
        f"PAO.BasisSize {basis}",
        f"XC.Functional GGA",
        f"XC.Authors {xc}",
        f"MeshCutoff {mesh_cutoff_ry:.1f} Ry",
        f"Spin {spin}",
        f"ElectronicTemperature {electronic_temperature_ev:.4f} eV",
        "",
        f"%block kgrid.MonkhorstPack",
        f"  {kgrid[0]} 0 0  0.0",
        f"  0 {kgrid[1]} 0  0.0",
        f"  0 0 {kgrid[2]}  0.0",
        "%endblock kgrid.MonkhorstPack",
        "",
        "DM.MixingWeight 0.15",
        "DM.NumberPulay 6",
        "MaxSCFIterations 200",
        "SCF.DM.Tolerance 1.0e-5",
        "",
        "SaveHS T",
        "CDF.Save T",
        "WriteEigenvalues T",
        "",
    ]
    if initial_moments:
        lines.append("%block DM.InitSpin")
        for idx, mom in initial_moments.items():
            lines.append(f"  {idx + 1}  {mom:+.2f}")
        lines += ["%endblock DM.InitSpin", ""]
    # `extra` must OVERRIDE the defaults above, not sit behind them. In fdf the
    # FIRST occurrence of a label wins, and labels ignore case and the
    # separators '.', '-', '_' -- so a caller writing "Max.SCF.Iterations 400"
    # was silently shadowed by the built-in "MaxSCFIterations 200" and got 200.
    # Drop any default whose normalized label the caller has set.
    def _norm(label):
        return label.replace(".", "").replace("-", "").replace("_", "").lower()

    overridden = {_norm(k) for k in (extra or {})}
    kept, in_block = [], False
    for ln in lines:
        low = ln.strip().lower()
        if low.startswith("%block"):
            in_block = True
        if not in_block and ln.strip() and _norm(ln.split()[0]) in overridden:
            continue                      # the caller sets this one
        kept.append(ln)
        if low.startswith("%endblock"):
            in_block = False
    lines = kept
    for k, v in (extra or {}).items():
        lines.append(f"{k} {v}")

    path.write_text("\n".join(lines) + "\n")

    # SIESTA looks for pseudos in the working directory
    for s in species:
        src = Path(pseudo_dir) / f"{s}.psml"
        dst = path.parent / f"{s}.psml"
        if not dst.exists():
            dst.symlink_to(src.resolve())
    return path


SIESTA_LAUNCHER_ENV = "WAW_SIESTA_LAUNCHER"


def siesta_launcher(ncores: int, launcher=None) -> list[str]:
    """
    The argv prefix that starts SIESTA, e.g. ``["mpirun", "-np", "4"]``.

    HOW SIESTA IS LAUNCHED IS THE CALLER'S BUSINESS, NOT THIS LIBRARY'S. MPI
    flags are site- and node-specific -- what a login node wants (``--bind-to
    none`` to stop ranks piling onto one core) is not what a batch node wants,
    and a scheduler may prefer ``srun`` over ``mpirun`` entirely. Baking a fixed
    incantation in here means every change of cluster, queue or MPI version
    requires editing the library and reinstalling it, which is backwards.

    Resolution order:
      1. the `launcher` argument -- a list, or a string that is shell-split;
      2. the ``WAW_SIESTA_LAUNCHER`` environment variable, so a submission
         script can set it once for a whole job without touching any code;
      3. ``["mpirun", "-np", str(ncores)]``.

    When 1 or 2 supplies the prefix it is used VERBATIM, including any rank
    count, and `ncores` then only informs the >= 8 ranks warning.
    """
    if launcher is None:
        launcher = os.environ.get(SIESTA_LAUNCHER_ENV) or None
    if launcher is None:
        return ["mpirun", "-np", str(ncores)]
    if isinstance(launcher, str):
        return shlex.split(launcher)
    return [str(x) for x in launcher]


def run_siesta(fdf_path, out_path=None, ncores: int = 4,
               consistency_ev: float | None = 0.01, launcher=None) -> Path:
    """
    Run `siesta < label.fdf > label.out`, launched by `siesta_launcher(ncores,
    launcher)` -- by default plain ``mpirun -np ncores``, overridable per call
    or through ``WAW_SIESTA_LAUNCHER`` (see that function; MPI flags are not
    this library's to choose). Load `module load
    siesta/5.4.1-gcc-13.2.0-dfgltdg` in the caller's environment first. Skips
    the run if out_path already ends in 'Job completed'.

    *** ncores DEFAULTS TO 4 BECAUSE >= 8 RANKS CAN SILENTLY CORRUPT THIS
    SIESTA BUILD. *** On siesta/5.4.1-gcc-13.2.0 here, some combinations of
    (n_orbitals, ncores, parallel BlockSize) return a Hamiltonian that is
    simply wrong: the SCF still reports convergence, but the final summary
    energy sits hundreds of eV from the converged E_KS, forces reach
    hundreds of eV/Ang, the total energy falls BELOW the variational bound,
    and the residual charge piles onto the last atom of the coordinate block.
    Measured on rocksalt Ni4O4 (128 orbitals) with byte-identical input:
    np = 1, 2, 4, 5, 6, 7 all give -20050.2421 eV with a zero consistency
    gap; np = 8 gives -21259.09 eV, a 2309 eV gap and 175 eV/Ang forces.
    Whether a given np >= 8 run is affected depends on the block size SIESTA
    picks (np8: BlockSize 17 broken, 15 and 16 clean at 128 orbitals, 15
    broken at 118) and is not predictable from outside the code -- so ncores
    <= 7, which is clean in every test, is the default. Forcing an explicit
    even `BlockSize` via `extra` is a workaround for larger runs, but it is
    unproven in general; the guarantee is the check below, not the workaround.

    After the run, `check_scf_consistency` is applied with threshold
    ``consistency_ev`` (pass None to skip): at self-consistency the final
    summary `siesta: Total` must equal the SCF's converged E_KS. Every
    corrupted run found so far fails it by 830-2310 eV while every healthy
    one gives exactly 0.00, so this raises rather than warns.
    """
    fdf_path = Path(fdf_path)
    out_path = Path(out_path) if out_path else fdf_path.with_suffix(".out")
    if out_path.exists() and "Job completed" in out_path.read_text()[-2000:]:
        if consistency_ev is not None:
            check_scf_consistency(out_path, tol_ev=consistency_ev)
        return out_path
    if ncores >= 8:
        warnings.warn(
            f"ncores = {ncores}: this SIESTA build can silently return a wrong "
            f"Hamiltonian at >= 8 MPI ranks (see run_siesta's docstring). "
            f"check_scf_consistency will catch it, but 4 ranks is the tested-"
            f"safe setting.", RuntimeWarning, stacklevel=2)
    argv = siesta_launcher(ncores, launcher) + ["siesta"]
    with open(fdf_path) as fin, open(out_path, "w") as fout:
        subprocess.run(
            argv,
            stdin=fin, stdout=fout, stderr=subprocess.STDOUT,
            cwd=fdf_path.parent, check=True,
        )
    if "Job completed" not in out_path.read_text()[-2000:]:
        raise RuntimeError(f"SIESTA did not complete -- see {out_path}")
    if consistency_ev is not None:
        check_scf_consistency(out_path, tol_ev=consistency_ev)
    from waw.utils.runs import autostamp
    autostamp(fdf_path.parent, code="siesta",
              settings={"ncores": ncores, "fdf": fdf_path.name})
    return out_path


def check_scf_consistency(out_path, tol_ev: float = 0.01) -> float:
    """
    |Eharris - Etot| from the FINAL energy summary, in eV, raising past tol.

    The two are equal at self-consistency by construction, so a nonzero value
    means the run is not what it claims to be. On this cluster's SIESTA 5.4.1
    they differ by 830-2310 eV whenever the parallel orbital distribution hits
    the >= 8-rank bug (see `run_siesta`), while every healthy run gives exactly
    0.00 -- and the SCF trace itself looks converged either way, which is why
    this check is the only cheap tell. The pathology also shows up as huge
    forces and pressures, but the energy pair alone identifies it.

    BOTH NUMBERS COME FROM THE SAME SUMMARY BLOCK, deliberately. An earlier
    version compared the last `scf:` line against the summary `Total` and gave
    a false positive of 0.891 eV on a healthy run: SIESTA output files can
    carry INTERLEAVED lines from two processes (a killed run whose file
    descriptor still held an offset when the file was truncated by a relaunch),
    so the textually last `scf:` line is not necessarily the last iteration of
    the run that wrote the summary. Reading one adjacent pair from the tail of
    the file is immune to that, and to multi-geometry runs with many SCF loops.
    """
    out_path = Path(out_path)
    text = out_path.read_text(errors="replace")
    if "End of run" not in text and "Job completed" not in text:
        raise RuntimeError(
            f"{out_path} does not reach the end of a run -- an unfinished "
            f"output cannot be certified. (Its early Eharris/Etot block "
            f"differs by ~1 keV in ANY healthy run: those two agree only at "
            f"self-consistency, so reading them mid-SCF proves nothing.)")
    harris = list(re.finditer(r"^siesta: +Eharris\S* +=\s*(-?\d+\.\d+)", text, re.M))
    if not harris:
        raise RuntimeError(f"no final energy summary in {out_path}")
    last = harris[-1]
    etot = re.search(r"^siesta: +Etot\S* +=\s*(-?\d+\.\d+)", text[last.end():], re.M)
    if etot is None:
        raise RuntimeError(f"summary in {out_path} has Eharris but no Etot")
    e_h, e_t = float(last.group(1)), float(etot.group(1))
    gap = abs(e_h - e_t)
    if gap > tol_ev:
        # Two very different faults land here, and saying the wrong one sends
        # the reader chasing a parallel bug that isn't there. An unconverged
        # SCF leaves Eharris and Etot a fraction of an eV apart (measured:
        # 0.50 eV on two Ag2NdCd fixed-spin points that ran out of iterations);
        # the parallel corruption leaves them 830-2310 eV apart AND still
        # prints a convergence marker.
        converged = "SCF Convergence" in text
        if not converged:
            raise RuntimeError(
                f"{out_path}: the SCF did NOT converge (no convergence marker) "
                f"and Eharris {e_h} differs from Etot {e_t} by {gap:.3f} eV. "
                f"This is ordinary non-convergence, not the parallel bug -- "
                f"give it more iterations or gentler mixing.")
        raise RuntimeError(
            f"{out_path}: the SCF reported convergence, yet the final summary "
            f"has Eharris {e_h} eV against Etot {e_t} eV ({gap:.1f} eV apart) "
            f"-- they are equal at self-consistency by construction, so this "
            f"run is NOT self-consistent whatever it printed. On this SIESTA "
            f"build that is the >= 8-rank parallel corruption (see "
            f"run_siesta); rerun with ncores <= 7.")
    return gap


def read_fermi_level(workdir, label: str) -> float:
    """Fermi level in eV from <label>.EIG (its first line).

    NOTE: this is the .EIG file's reference. The saved Hamiltonian that
    `loader.lowdin_hamiltonian` consumes is stored as H - E_F S, so the
    loaded model's Fermi level is 0, not this value."""
    eig = Path(workdir) / f"{label}.EIG"
    return float(eig.read_text().split()[0])
