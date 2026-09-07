"""
``projwfc.x``: atomic-orbital projectability of the DFT bands.

Used to pick an SCDM entanglement function (``scdm_entanglement='erfc'``)
by fitting the erfc form to the (energy, total atomic projectability)
curve from a converged NSCF, instead of guessing ``scdm_mu``/``scdm_sigma``.

``projwfc.x`` is MPI-parallel on this cluster (unlike the serial-only
``wannier90.x``/``postw90.x`` builds here) -- always launch under ``mpirun``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .io import mpirun_prefix, _namelist, _mpi_env


def run_projwfc(workdir, seedname: str, *, ncores: int = 16,
                projwfc: str = "projwfc.x") -> Path:
    """
    Run ``projwfc.x`` on a converged ``<seedname>`` SCF/NSCF in ``workdir/out``.

    Writes ``<seedname>.proj.in`` (the ``&projwfc`` namelist) and runs it
    under ``mpirun``, redirecting stdout+stderr to ``<workdir>/proj.out``.

    ``ncores`` is capped at 4: ``projwfc.x``'s subspace diagonalization runs
    over the small ``natomwfc`` atomic-orbital basis, and QE's ScaLAPACK
    path aborts once the MPI rank count exceeds that matrix dimension.

    Returns the path to ``proj.out`` (consumed by `read_projectability`).
    """
    workdir = Path(workdir)
    inp = {"prefix": seedname, "outdir": "./out", "filproj": f"{seedname}-proj"}
    (workdir / f"{seedname}.proj.in").write_text(_namelist("projwfc", inp))
    out_path = workdir / "proj.out"
    with open(out_path, "w") as fout:
        subprocess.run(
            mpirun_prefix(min(ncores, 4)) + [projwfc, "-in", f"{seedname}.proj.in"],
            cwd=workdir, stdout=fout, stderr=subprocess.STDOUT, env=_mpi_env(),
            check=True,
        )
    return out_path


def read_projectability(proj_out_path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse ``projwfc.x``'s ``proj.out`` into (energy, total-projectability) pairs.

    One pair per (k-point, band), i.e. ``nk * nbnd`` entries, not sorted or
    deduplicated by band index. Line formats::

        ==== e(   1) =    -5.73167 eV ====
             psi = 0.498*[#   1]+0.498*[#   5]
            |psi|^2 = 0.996

    Energy is whitespace-split token 4 of the ``==== e(...)`` line;
    projectability (dimensionless, <= 1) is token 2 of the ``|psi|^2`` line.

    Returns
    -------
    energies_eV, projectability : (n,) float64 arrays, n = nk * nbnd.
    """
    text = Path(proj_out_path).read_text()
    energies = []
    proj = []
    for line in text.splitlines():
        if "==== e(" in line:
            energies.append(float(line.split()[4]))
        elif "|psi|^2" in line:
            proj.append(float(line.split()[2]))
    if len(energies) != len(proj):
        raise ValueError(
            f"{proj_out_path}: found {len(energies)} energy lines but "
            f"{len(proj)} |psi|^2 lines -- mismatched/truncated proj.out"
        )
    return np.array(energies, dtype=np.float64), np.array(proj, dtype=np.float64)
