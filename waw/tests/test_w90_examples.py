"""
Validation against Wannier90 reference outputs.

Runs waw on precomputed .mmn/.amn/.eig files from the W90 test suite
and compares the final spread components (Omega_I, Omega_D, Omega_OD, Omega)
against W90's benchmark.out reference values.

Fixtures
--------
  tests/data/gaas/ — GaAs, 4 WFs, isolated valence bands (no disentanglement)

The reference files contain Wannier90 v3.x output; tolerances are set to
allow for minor numerical differences in the spread decomposition.

(Copper — the entangled 7-WF case — has moved to tests/test_tutorial04.py,
which validates it against a full real wannier90.x 3.1.0 run.)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw import wannierize

GAAS_DIR   = Path(__file__).parent / "data" / "gaas"

HAS_GAAS   = (GAAS_DIR   / "gaas.mmn").exists()

from waw.units import BOHR_TO_ANG, ANG_TO_BOHR
ANG2_TO_BOHR2 = ANG_TO_BOHR ** 2   # ≈ 3.5711 Bohr²/Ang²


# ---------------------------------------------------------------------------
# Parse reference values from benchmark.out
# ---------------------------------------------------------------------------

def _parse_benchmark(path: Path) -> dict[str, float]:
    """
    Extract final spread components from a Wannier90 benchmark.out file.

    Parses lines of the form (W90 ≥ 3.x output):
        Final Omega_I  3.956862958 (Ang^2)
        Omega I   = 3.956862958
        Omega D   = 0.008030049
        Omega OD  = 0.501987969
        Final Spread (Ang^2)  Omega Total = 4.466880976
    """
    text = path.read_text()
    result: dict[str, float] = {}

    # "Final Omega_I   X.XXXXX (Ang^2)" — disentanglement output
    m = re.search(r"Final\s+Omega_I\s+([\d.]+)\s+\(Ang", text)
    if m:
        result["omega_i_dis"] = float(m.group(1))

    # "Omega I      =   X.XXXXXXX"
    m = re.search(r"Omega\s+I\s*=\s*([\d.]+)", text)
    if m:
        result["omega_i"] = float(m.group(1))

    # "Omega D      =   X.XXXXXXX"
    m = re.search(r"Omega\s+D\s*=\s*([\d.]+)", text)
    if m:
        result["omega_d"] = float(m.group(1))

    # "Omega OD     =   X.XXXXXXX"
    m = re.search(r"Omega\s+OD\s*=\s*([\d.]+)", text)
    if m:
        result["omega_od"] = float(m.group(1))

    # "Final Spread (Ang^2)  Omega Total  =  X.XXXXXXX"
    m = re.search(r"Final\s+Spread.*?Omega\s+Total\s*=\s*([\d.]+)", text)
    if m:
        result["omega_total"] = float(m.group(1))

    return result


BOHR_TO_ANG2 = BOHR_TO_ANG ** 2


# ---------------------------------------------------------------------------
# GaAs — isolated valence bands, 4 sp3 WFs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_GAAS, reason="GaAs example files not found")
class TestGaAs:
    """
    GaAs: 4 valence bands, 4 WFs, isolated (nb == nw).

    W90 reference (benchmark.out):
        Omega I   = 3.956862958 Ang²
        Omega D   = 0.008030049 Ang²
        Omega OD  = 0.501987969 Ang²
        Omega     = 4.466880976 Ang²
    """

    @pytest.fixture(scope="class")
    @classmethod
    def ref(cls):
        return _parse_benchmark(GAAS_DIR / "benchmark.out")

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        return wannierize(
            GAAS_DIR / "gaas",
            n_iter       = 500,
            conv_tol     = 1e-10,
            write_outputs = False,
            verbose       = False,
        )

    def test_omega_total(self, result, ref):
        got = result.spread.Omega * BOHR_TO_ANG2
        exp = ref["omega_total"]
        assert abs(got - exp) < 0.01, (
            f"Omega_total: got {got:.6f}, expected {exp:.6f} Ang²"
        )

    def test_omega_i(self, result, ref):
        got = result.spread.Omega_I * BOHR_TO_ANG2
        exp = ref["omega_i"]
        assert abs(got - exp) < 0.01, (
            f"Omega_I: got {got:.6f}, expected {exp:.6f} Ang²"
        )

    def test_omega_d(self, result, ref):
        got = result.spread.Omega_D * BOHR_TO_ANG2
        exp = ref["omega_d"]
        assert abs(got - exp) < 0.005, (
            f"Omega_D: got {got:.6f}, expected {exp:.6f} Ang²"
        )

    def test_omega_od(self, result, ref):
        got = result.spread.Omega_OD * BOHR_TO_ANG2
        exp = ref["omega_od"]
        assert abs(got - exp) < 0.01, (
            f"Omega_OD: got {got:.6f}, expected {exp:.6f} Ang²"
        )

    def test_spreads_sum_consistent(self, result):
        """Per-WF spreads must sum to Omega recomputed at U_final."""
        total = result.spreads_bohr2.sum() * BOHR_TO_ANG2
        omega = result.omega_final * BOHR_TO_ANG2
        assert abs(total - omega) < 1e-8, (
            f"Sum of per-WF spreads {total:.10f} ≠ omega_final {omega:.10f}"
        )
