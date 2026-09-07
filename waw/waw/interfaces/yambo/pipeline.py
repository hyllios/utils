"""
End-to-end G0W0 quasiparticle correction (Wannier90 tutorial 23): a denser
QE nscf -> p2y -> yambo (em1d+gw0+ppa+HF_and_locXC) -> ypp (`wannier` mode)
pipeline that turns a converged DFT calculation into a GW-corrected
``.eig``-format eigenvalue array, ready to feed back into
``waw.interfaces.ase.driver.wannierize`` like
``quantum_espresso.generate_overlaps``'s own ``out["eig"]``.

``ypp -wannier`` produces a Wannier90-standard ``.eig``-format file of QP
corrections (``eig_gw - eig_dft`` per band/k, not absolute eigenvalues or
band-sorted), parsed by the existing
``waw.interfaces.wannier90.io.read_eig``; this module adds the correction
to the DFT eigenvalues and re-sorts each k-point's bands ascending (QP
corrections can reorder near-degenerate bands).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from waw.interfaces.wannier90.io import read_eig
from waw.interfaces.quantum_espresso.io import write_pw_input, run_pw

from .io import (
    run_p2y, write_yambo_input, run_yambo, parse_ibz_kpoint_count,
    write_ypp_input, run_ypp,
)


def run_gw_correction(
    atoms, gw_mesh, workdir, seedname, dft_eig, *,
    ecutwfc: float,
    pseudopotentials: dict,
    pseudo_dir,
    nbnd_gw: int = 100,
    qpkrange_bands: tuple[int, int] = (1, 14),
    ncores: int = 1,
):
    """
    Run the GW correction and return the corrected, band-sorted ``eig``
    array (nk, nb) in eV -- ``dft_eig`` (nk, nb) is the DFT ``.eig`` array
    from a prior ``quantum_espresso.generate_overlaps`` call with the SAME
    ``workdir``/``seedname`` (this function reuses that call's converged
    SCF charge density and ``.nnkp``, only recomputing wavefunctions on the
    denser ``gw_mesh`` -- run the plain DFT wannierisation first). ``nk``
    ordering matches ``dft_eig``'s own (both come from the same ``.nnkp``
    k-point list).

    ``gw_mesh``: the denser automatic mesh for the GW self-energy/screening
    itself (every Wannier k-point must be a subset of this mesh).
    ``qpkrange_bands``: the 1-based band range to QP-correct (only the
    bands feeding the Wannier ``.eig`` need correcting, not all
    ``nbnd_gw`` bands used for screening).
    """
    workdir = Path(workdir)
    qe_save_dir = workdir / "out" / f"{seedname}.save"
    nnkp_path = workdir / f"{seedname}.nnkp"
    if not qe_save_dir.exists() or not nnkp_path.exists():
        raise FileNotFoundError(
            f"expected a converged SCF + .nnkp for {seedname!r} in {workdir} "
            "already (run quantum_espresso.generate_overlaps first)")

    # 1. Denser nscf for the GW self-energy/screening, reusing the existing
    #    converged charge density (same prefix/outdir, calculation=nscf).
    write_pw_input(
        workdir / f"{seedname}.gw.nscf.in", atoms,
        control={"calculation": "nscf", "prefix": seedname,
                 "outdir": "./out", "pseudo_dir": str(pseudo_dir),
                 "wf_collect": True},
        system={"ecutwfc": ecutwfc, "nbnd": nbnd_gw,
                "force_symmorphic": True},
        electrons={"conv_thr": 1.0e-8, "diago_thr_init": 1.0e-4},
        pseudopotentials=pseudopotentials,
        kpoints=("automatic", tuple(gw_mesh), (0, 0, 0)),
    )
    run_pw(workdir / f"{seedname}.gw.nscf.in", workdir / f"{seedname}.gw.nscf.out",
           ncores=ncores)

    # 2. p2y: QE .save -> Yambo SAVE/ (must run with cwd = the .save dir).
    run_p2y(qe_save_dir, ncores=ncores)

    # 3. Separate, portable Yambo run directory (SAVE/ + .nnkp copied in).
    rundir = workdir / "gw"
    rundir.mkdir(parents=True, exist_ok=True)
    if (rundir / "SAVE").exists():
        shutil.rmtree(rundir / "SAVE")
    shutil.copytree(qe_save_dir / "SAVE", rundir / "SAVE")
    shutil.copy(nnkp_path, rundir / f"{seedname}.nnkp")

    # 4. First pass with a placeholder %QPkrange, just to read Yambo's own
    #    IBZ k-point count off its report.
    b0, b1 = qpkrange_bands
    write_yambo_input(rundir / "yambo_setup.in", qpkrange=f"1|1|{b0}|{b1}|",
                       nbnd_gw=nbnd_gw)
    setup_report = run_yambo(rundir, rundir / "yambo_setup.in", jobname="setup",
                             ncores=ncores)
    n_ibz = parse_ibz_kpoint_count(setup_report.read_text())

    # 5. Real G0W0 run with the full %QPkrange.
    write_yambo_input(rundir / "yambo.in", qpkrange=f"1|{n_ibz}|{b0}|{b1}|",
                       nbnd_gw=nbnd_gw)
    run_yambo(rundir, rundir / "yambo.in", jobname="gw", ncores=ncores)

    # 6. ypp (`wannier` mode): Yambo QP database -> per-(band,k) QP shift.
    write_ypp_input(rundir / "ypp.in", seedname=seedname)
    run_ypp(rundir, rundir / "ypp.in", jobname="gw", ncores=ncores)
    delta_qp = read_eig(rundir / f"{seedname}.gw.unsorted.eig")

    # 7. Add to the DFT eigenvalues (same band index, before reordering)
    #    and re-sort ascending per k-point (QP corrections can reorder
    #    near-degenerate bands).
    dft_eig = np.asarray(dft_eig, dtype=np.float64)
    return np.sort(dft_eig + delta_qp, axis=1)
