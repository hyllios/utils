"""
Direct-input Yambo driver: run p2y/yambo/ypp for a G0W0 quasiparticle
correction (Wannier90 tutorial 23, Silicon).

MPI subprocess wrapper writing a ``.out`` log per step, mirroring
``waw.interfaces.quantum_espresso.io`` -- p2y/yambo/ypp abort in
``MPI_Init`` if run bare, like ``pw.x``/``pw2wannier90.x``.

Requires ``module load quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k`` (not
this project's usual 7.5 -- ``p2y`` is sensitive to the exact QE
XML/wavefunction format) plus ``module load intel/oneapi/mkl/2024.2`` and
``~/software/yambo/bin`` on ``PATH``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from waw.interfaces.quantum_espresso.io import mpirun_prefix, _mpi_env


def run_p2y(qe_save_dir: Path, *, ncores: int = 1) -> Path:
    """
    Convert a QE ``<prefix>.save/`` directory (from a ``wf_collect=.true.``
    nscf) into a Yambo ``SAVE/`` database, written inside ``qe_save_dir``.
    """
    qe_save_dir = Path(qe_save_dir)
    out_path = qe_save_dir / "p2y.out"
    with open(out_path, "w") as out:
        subprocess.run(mpirun_prefix(ncores) + ["p2y"], cwd=qe_save_dir,
                       stdout=out, stderr=subprocess.STDOUT, env=_mpi_env(), check=True)
    return qe_save_dir / "SAVE"


_YAMBO_TEMPLATE = """\
em1d
gw0
ppa
HF_and_locXC
EXXRLvcs= 60000        RL
Chimod= "hartree"
% BndsRnXp
   1 | {nbnd_gw} |
%
NGsBlkXp= 1            RL
% LongDrXp
 1.000000 | 0.000000 | 0.000000 |
%
PPAPntXp= 27.21138     eV
% GbndRnge
   1 | {nbnd_gw} |
%
GDamping=  0.10000     eV
dScStep=  0.10000      eV
GTermKind= "none"
DysSolver= "n"
%QPkrange
{qpkrange}
%
"""


def write_yambo_input(path, *, qpkrange: str, nbnd_gw: int = 100) -> Path:
    """
    Write a ``yambo.in`` for the em1d+gw0+ppa+HF_and_locXC G0W0 recipe.

    ``EXXRLvcs`` is set generously large -- Yambo clips it to the max RL
    vectors available in the density FFT grid and reports the clipped
    value. ``qpkrange`` is the pre-formatted ``"k0|k1|b0|b1|"`` string (see
    ``pipeline.run_gw_correction``); ``k0``/``k1`` need Yambo's own IBZ
    k-point count (``parse_ibz_kpoint_count``), not the Wannier mesh count.
    """
    path = Path(path)
    path.write_text(_YAMBO_TEMPLATE.format(qpkrange=qpkrange, nbnd_gw=nbnd_gw))
    return path


def run_yambo(rundir, input_path, *, jobname: str = "gw", ncores: int = 1) -> Path:
    """
    Run yambo (needs ``rundir/SAVE`` already present, e.g. via ``run_p2y``).

    Yambo writes its human-readable report to a job-specific
    ``r-<jobname>_<runlevels>`` file, not stdout (per-rank logs go to
    ``LOG/``) -- returns the path to the newest such report file.
    """
    rundir, input_path = Path(rundir), Path(input_path)
    out_path = rundir / f"{jobname}.out"
    with open(out_path, "w") as out:
        subprocess.run(mpirun_prefix(ncores) +
                       ["yambo", "-F", input_path.name, "-J", jobname],
                       cwd=rundir, stdout=out, stderr=subprocess.STDOUT,
                       env=_mpi_env(), check=True)
    reports = sorted(rundir.glob(f"r-{jobname}_*"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"yambo produced no r-{jobname}_* report in {rundir}")
    return reports[-1]


def parse_ibz_kpoint_count(report_text: str) -> int:
    """
    Read the number of inequivalent (IBZ) k-points Yambo reduced the input
    mesh to, from the report's system-setup section (the line labeled
    ``K-points``, distinct from ``IBZ Q-points``/``BZ Q-points`` which
    describe the momentum-transfer mesh, not the electronic k-mesh). Needed
    to size ``%QPkrange`` so every Wannier-mesh k-point has a QP correction
    for ``ypp -wannier`` to unfold.
    """
    for line in report_text.splitlines():
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        if label.strip() == "K-points":
            return int(value.split()[0])
    raise ValueError("could not find a 'K-points' line in the Yambo report")


_YPP_TEMPLATE = """\
wannier
Seed= "{seedname}"
WriteAMU= ""
"""


def write_ypp_input(path, *, seedname: str) -> Path:
    path = Path(path)
    path.write_text(_YPP_TEMPLATE.format(seedname=seedname))
    return path


def run_ypp(rundir, input_path, *, jobname: str = "gw", ncores: int = 1) -> Path:
    """
    Run ypp (`wannier` runlevel): reads `<seedname>.nnkp` plus the yambo QP
    database from `rundir/<jobname>/ndb.QP` -- `jobname` MUST match the
    real `run_yambo` GW pass's, or ypp silently writes nothing.

    Writes `<seedname>.gw.unsorted.eig`: QP energy corrections
    (`eig_gw - eig_dft`) per (band, k-point), same format as
    `wannier90.io.read_eig` but not yet absolute/sorted -- see
    `pipeline.run_gw_correction`.
    """
    rundir, input_path = Path(rundir), Path(input_path)
    out_path = rundir / "ypp.out"
    with open(out_path, "w") as out:
        subprocess.run(mpirun_prefix(ncores) +
                       ["ypp", "-F", input_path.name, "-J", jobname],
                       cwd=rundir, stdout=out, stderr=subprocess.STDOUT,
                       env=_mpi_env(), check=True)
    return out_path
