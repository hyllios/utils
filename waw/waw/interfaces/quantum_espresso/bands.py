"""
Ab-initio band energies along a k-path, for DFT-vs-Wannier comparison plots.

The single most effective way to catch a bad Wannier model is to overlay the
interpolated bands on the ab-initio ones with the disentanglement windows
drawn in (`waw.vis.plot_wannierization_windows`). This module supplies the
missing half of that picture: a `calculation='bands'` run on an explicit
k-path, reusing the run's own NSCF input so nothing about the system is
re-specified (and nothing can silently drift out of sync with the model).

Two traps this encapsulates, both found the hard way:

1. **`verbosity = 'high'` is mandatory.** Without it pw.x prints no
   ``bands (ev)`` blocks at all and `read_bands_eigenvalues` raises.
2. **Cards after `K_POINTS` must survive.** A naive
   ``re.sub(r'K_POINTS.*', klist, src, flags=S)`` deletes everything to EOF --
   which for a DFT+U run silently drops the `HUBBARD` card, turning the
   "reference" bands into plain PBE (Ni-3d off by 1.3 eV in notebook 17
   before this was caught). The replacement here stops at the next card.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .io import read_bands_eigenvalues, run_pw

__all__ = ["bands_along_path"]

#: Cards that may legally follow K_POINTS in a pw.x input and must be kept.
_TRAILING_CARDS = ("HUBBARD", "OCCUPATIONS", "CONSTRAINTS", "ATOMIC_FORCES",
                   "ATOMIC_VELOCITIES", "SOLVENTS", "TOTAL_CHARGE")


def _cached_kpoints(path: Path) -> np.ndarray | None:
    """The explicit K_POINTS crystal list from a pw.x input, or None."""
    if not path.exists():
        return None
    m = re.search(r"K_POINTS\s+crystal\s*\n\s*(\d+)\s*\n(.*)", path.read_text(),
                  re.S | re.I)
    if not m:
        return None
    rows = []
    for line in m.group(2).splitlines()[: int(m.group(1))]:
        parts = line.split()
        if len(parts) < 3:
            break
        rows.append([float(x) for x in parts[:3]])
    return np.asarray(rows, dtype=np.float64) if rows else None


def _cache_matches(workdir: Path, seedname: str, kpts: np.ndarray) -> bool:
    """
    True only if a finished bands run exists AND was computed on these k-points.

    Checking "JOB DONE" alone is not enough: a run left over from a DIFFERENT
    k-path is reused silently and the caller gets ab-initio energies belonging
    to other k-points. That surfaced in notebook 3, whose cached output held a
    path of a different length -- and had the lengths happened to agree it would
    have produced a plausible, wrong reference curve instead of an error.
    """
    out = workdir / f"{seedname}.bands.out"
    if not (out.exists() and "JOB DONE" in out.read_text()[-2000:]):
        return False
    cached = _cached_kpoints(workdir / f"{seedname}.bands.in")
    return (cached is not None and cached.shape == kpts.shape
            and np.allclose(cached, kpts, atol=1e-8))


def bands_along_path(
    workdir,
    seedname: str,
    kpts,
    *,
    nbnd: int | None = None,
    ncores: int = 8,
    nspin: int = 1,
    rerun: bool = False,
) -> np.ndarray:
    """
    Ab-initio eigenvalues (eV) at `kpts`, via `calculation='bands'`.

    Reuses ``<workdir>/<seedname>.nscf.in`` as the template, so the cell,
    pseudopotentials, cutoffs, SOC/magnetic settings and any DFT+U card are
    exactly those the Wannier model was built from. Needs the run's charge
    density (``<workdir>/out/<seedname>.save/``) to still be present.

    Args:
      workdir, seedname : the run directory and prefix
      kpts              : (nk, 3) k-points in crystal coordinates, e.g.
                          ``interfaces.ase.structure.band_path(atoms).kpts``
      nbnd              : number of bands (default: whatever the NSCF used)
      ncores            : MPI ranks
      nspin             : 2 for a collinear spin-polarized run -- pw.x then
                          prints the spin-up blocks followed by spin-down,
                          and the return shape becomes (2, nk, nbnd)
      rerun             : recompute even if a completed run is cached

    Returns
    -------
    (nk, nbnd) float64 eV, or (2, nk, nbnd) when ``nspin=2``.
    """
    workdir = Path(workdir)
    kpts = np.asarray(kpts, dtype=np.float64)
    nk = len(kpts)
    out = workdir / f"{seedname}.bands.out"

    if rerun or not _cache_matches(workdir, seedname, kpts):
        template = workdir / f"{seedname}.nscf.in"
        if not template.exists():
            raise FileNotFoundError(
                f"{template} is required as the template for the bands run "
                f"(it carries the cell, pseudos and any HUBBARD card)."
            )
        save = workdir / "out" / f"{seedname}.save"
        if not save.is_dir():
            raise FileNotFoundError(
                f"{save} not found: a bands run needs the charge density from "
                f"the original scf. Re-run the scf/nscf, or fall back to the "
                f"on-mesh fidelity check (analysis.bands.mesh_fidelity)."
            )
        src = template.read_text()
        src = re.sub(r"calculation\s*=\s*'[^']*'", "calculation = 'bands'", src, count=1)
        if "verbosity" not in src:      # without this pw.x prints no eigenvalues
            src = re.sub(r"(&control|&CONTROL)", r"\1\n  verbosity = 'high'", src, count=1)
        if nbnd is not None:
            if re.search(r"nbnd\s*=", src):
                src = re.sub(r"nbnd\s*=\s*\d+", f"nbnd = {nbnd}", src, count=1)
            else:
                src = re.sub(r"(&system|&SYSTEM)", rf"\1\n  nbnd = {nbnd}", src, count=1)

        klist = (f"K_POINTS crystal\n{nk}\n"
                 + "".join(f"  {k[0]:.10f}  {k[1]:.10f}  {k[2]:.10f}  1.0\n" for k in kpts))
        # replace the K_POINTS card only, keeping any card that follows it
        stop = "|".join(_TRAILING_CARDS)
        src = re.sub(rf"K_POINTS.*?(?=^\s*(?:{stop})|\Z)", klist,
                     src, count=1, flags=re.S | re.M)

        (workdir / f"{seedname}.bands.in").write_text(src)
        run_pw(workdir / f"{seedname}.bands.in", out, ncores=ncores)

    if nbnd is None:      # take it from the run's own header
        m = re.search(r"number of Kohn-Sham states\s*=\s*(\d+)", out.read_text())
        if not m:
            raise ValueError(f"{out}: cannot determine nbnd; pass it explicitly.")
        nbnd = int(m.group(1))
    eig = np.asarray(read_bands_eigenvalues(out, nbnd=nbnd))
    if nspin == 2:
        if len(eig) != 2 * nk:
            raise ValueError(
                f"{out}: expected {2 * nk} eigenvalue blocks for nspin=2, got {len(eig)}"
            )
        return np.stack([eig[:nk], eig[nk:]])
    if len(eig) != nk:
        raise ValueError(
            f"{out}: expected {nk} eigenvalue blocks, got {len(eig)} -- if this is a "
            f"collinear spin-polarized run, pass nspin=2."
        )
    return eig
