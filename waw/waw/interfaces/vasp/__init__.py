"""
Direct-input VASP interface for waw.

``from waw.interfaces import vasp as vasp`` gives ``vasp.generate_overlaps(...)``
(the VASP analogue of ``qe.generate_overlaps``): run a VASP SCF with the
VASP2WANNIER90 interface and hand the ``wannier90.mmn/.amn/.eig`` overlaps to
``waw.interfaces.ase.driver.wannierize``. Also exposes the low-level input
writers / runner / readers in ``io``.

``Wavecar`` reads the plane-wave coefficients themselves, which band unfolding
(``waw.analysis.unfolding``) needs and the .mmn/.amn/.eig do not carry.
"""

from .io import (
    write_poscar, write_potcar, write_kpoints, write_kpoints_explicit,
    write_incar, write_wannier_win, noncollinear_magmom, run_vasp,
    read_fermi_energy, read_kpoints, read_overlaps, read_bands, mpirun_prefix,
    DEFAULT_VASP_BIN_DIR, DEFAULT_POTCAR_DIR,
)
from .pipeline import generate_overlaps, bands
from .wavecar import Wavecar, WavecarHeader

__all__ = [
    "generate_overlaps", "bands", "Wavecar", "WavecarHeader",
    "write_poscar", "write_potcar", "write_kpoints", "write_kpoints_explicit",
    "write_incar", "write_wannier_win", "noncollinear_magmom", "run_vasp",
    "read_fermi_energy", "read_kpoints", "read_overlaps", "read_bands",
    "mpirun_prefix", "DEFAULT_VASP_BIN_DIR", "DEFAULT_POTCAR_DIR",
]
