"""
Direct-input Quantum ESPRESSO interface.

``from waw.interfaces import quantum_espresso as qe`` gives
``qe.generate_overlaps(...)``, ``qe.write_pw_input(...)``, etc.
"""

from .io import (
    mpirun_prefix, write_pw_input, run_pw, read_fermi_energy,
    read_bands_eigenvalues, gamma_only_half_shell_nnkp, run_pw2wannier90,
    write_ph_input, write_ph_input_explicit_q, write_q2r_input,
    magnetization_keys,
)
from .pipeline import generate_overlaps
from .bands import bands_along_path
from .projwfc import run_projwfc, read_projectability
from .phonon_io import read_force_constants, read_ph_frequencies

__all__ = [
    "mpirun_prefix", "write_pw_input", "run_pw", "read_fermi_energy",
    "read_bands_eigenvalues",
    "bands_along_path",
    "gamma_only_half_shell_nnkp", "run_pw2wannier90", "generate_overlaps",
    "magnetization_keys",
    "run_projwfc", "read_projectability",
    "write_ph_input", "write_ph_input_explicit_q", "write_q2r_input",
    "read_force_constants", "read_ph_frequencies",
]
