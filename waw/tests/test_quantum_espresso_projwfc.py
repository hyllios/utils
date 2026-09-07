"""
Tests for waw/interfaces/quantum_espresso/projwfc.py.

`read_projectability` is tested against a REAL, genuinely captured
`projwfc.x` v7.5 output sample (`tests/data/proj_si_sample.out` -- the first
4 k-points of a real run on the tutorial27 silicon system, `nbnd=4`), not a
synthesized string -- this project's usual standard for file-format parsers.
"""

from pathlib import Path

import numpy as np
import pytest

from waw.interfaces.quantum_espresso.projwfc import read_projectability

DATA = Path(__file__).parent / "data" / "proj_si_sample.out"

# Hand-verified against the raw fixture text (4 k-points x 4 bands = 16 pairs).
EXPECTED_E = [
    -5.73167, 6.23496, 6.23505, 6.23507,
    -4.93213, 2.30364, 5.47333, 5.47345,
    -3.40061, -0.73979, 5.03392, 5.03395,
    -4.93196, 2.30355, 5.47332, 5.47397,
]
EXPECTED_P = [
    0.996, 0.962, 0.962, 0.962,
    0.996, 0.987, 0.981, 0.981,
    0.997, 0.987, 0.989, 0.989,
    0.996, 0.987, 0.981, 0.982,
]


def test_read_projectability_matches_real_sample():
    energies, proj = read_projectability(DATA)
    assert energies.shape == proj.shape == (16,)
    np.testing.assert_allclose(energies, EXPECTED_E)
    np.testing.assert_allclose(proj, EXPECTED_P)


def test_read_projectability_energy_and_proj_ranges():
    energies, proj = read_projectability(DATA)
    # projectability is a normalized weight, physically in (0, 1]
    assert np.all(proj > 0) and np.all(proj <= 1.0)
    # this sample's valence band sits well below its conduction bands
    assert energies.min() < -3.0
    assert energies.max() > 5.0


def test_read_projectability_mismatched_file_raises(tmp_path):
    bad = tmp_path / "bad.out"
    bad.write_text("==== e(   1) =    -5.0 eV ====\n    |psi|^2 = 0.9\n"
                    "==== e(   2) =    -4.0 eV ====\n")   # missing 2nd |psi|^2
    with pytest.raises(ValueError, match="mismatched"):
        read_projectability(bad)


def test_read_projectability_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_projectability(Path("/nonexistent/proj.out"))
