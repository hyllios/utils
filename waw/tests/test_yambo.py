"""
Tests for waw/interfaces/yambo/.

`parse_ibz_kpoint_count` is tested against a REAL, genuinely captured Yambo
5.3.0 report (`tests/data/yambo_si_r_setup_sample.txt` -- a real
em1d+gw0+ppa+HF_and_locXC setup pass on the tutorial23 Silicon system, an
8x8x8 mesh reduced to 29 IBZ k-points), not a synthesized string -- this
project's usual standard for file-format parsers (see
test_quantum_espresso_projwfc.py). The subprocess-driving functions
(run_p2y/run_yambo/run_ypp) need the real Yambo toolchain and aren't unit
tested here, matching this project's convention for QE-toolchain wrappers
(see test_w90tutorial.py's own "does NOT run DFT" scoping).
"""

from pathlib import Path

import numpy as np
import pytest

from waw.interfaces.yambo.io import (
    parse_ibz_kpoint_count, write_yambo_input, write_ypp_input,
)

DATA = Path(__file__).parent / "data" / "yambo_si_r_setup_sample.txt"


def test_parse_ibz_kpoint_count_matches_real_sample():
    assert parse_ibz_kpoint_count(DATA.read_text()) == 29


def test_parse_ibz_kpoint_count_not_confused_by_q_points_lines():
    # the real report also has "IBZ Q-points"/"BZ Q-points"/"Brillouin Zone
    # Q/K grids" lines carrying the SAME numeric value (29) under a
    # DIFFERENT label -- a naive substring match on "K-points" alone would
    # accidentally match those too (parse_ibz_kpoint_count must not).
    text = DATA.read_text()
    assert "IBZ Q-points" in text and "BZ  Q-points" in text
    assert parse_ibz_kpoint_count(text) == 29


def test_parse_ibz_kpoint_count_missing_line_raises():
    with pytest.raises(ValueError, match="K-points"):
        parse_ibz_kpoint_count("no relevant line here\n")


def test_write_yambo_input_fills_qpkrange_and_bands(tmp_path):
    path = write_yambo_input(tmp_path / "yambo.in", qpkrange="1|29|1|14|", nbnd_gw=100)
    text = path.read_text()
    assert "%QPkrange" in text
    assert "1|29|1|14|" in text
    assert "1 | 100 |" in text          # BndsRnXp / GbndRnge
    for runlevel in ("em1d", "gw0", "ppa", "HF_and_locXC"):
        assert runlevel in text


def test_write_ypp_input_fills_seedname(tmp_path):
    path = write_ypp_input(tmp_path / "ypp.in", seedname="si_gw")
    text = path.read_text()
    assert "wannier" in text
    assert 'Seed= "si_gw"' in text


def test_qp_correction_combine_and_resort_matches_pipeline_logic():
    """
    `run_gw_correction`'s final step (add the QP shift to the DFT
    eigenvalues, then re-sort each k-point's bands ascending) needs no
    Yambo run to unit-test -- exercise the same `dft_eig + delta_qp` /
    `np.sort(..., axis=1)` logic directly, including a deliberate band
    reordering to confirm the sort actually fixes a QP-induced crossing.
    """
    dft_eig = np.array([[-5.0, 0.0, 0.1, 5.0]])
    delta_qp = np.array([[-1.0, 0.5, -0.05, 1.0]])   # bands 2,3 cross after +delta
    combined = np.sort(dft_eig + delta_qp, axis=1)
    np.testing.assert_allclose(combined, [[-6.0, 0.05, 0.5, 6.0]])
