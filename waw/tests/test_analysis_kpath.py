"""
Tests for waw/analysis/kpath.py.

  1. parse_kpoint_path splits each line into (label1, kpt1, label2, kpt2).
  2. build_kpath produces n_points per segment, merging shared endpoints
     of continuous segments (no duplicate k-points).
  3. Cumulative distance is measured in Cartesian (reciprocal) units and
     is monotonically non-decreasing.
  4. Discontinuous jumps (segment i's end != segment i+1's start) are not
     merged, and their tick labels are combined as "A|B".
"""

from pathlib import Path
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.analysis.kpath import parse_kpoint_path, build_kpath


CUBIC_RECIP = 2 * np.pi * np.eye(3)   # simple cubic, a=1 Bohr


def test_parse_kpoint_path():
    lines = [
        "G 0.00  0.00  0.00    X 0.50  0.50  0.00",
        "X 0.50  0.50  0.00    W 0.50  0.75  0.25",
    ]
    segments = parse_kpoint_path(lines)
    assert len(segments) == 2

    label1, k1, label2, k2 = segments[0]
    assert label1 == "G" and label2 == "X"
    np.testing.assert_allclose(k1, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(k2, [0.5, 0.5, 0.0])


def test_parse_kpoint_path_malformed():
    with pytest.raises(ValueError):
        parse_kpoint_path(["G 0.0 0.0 0.0 X 0.5 0.5"])


def test_build_kpath_continuous_no_duplicates():
    lines = [
        "G 0.00 0.00 0.00   X 0.50 0.00 0.00",
        "X 0.50 0.00 0.00   G 0.00 0.00 0.00",
    ]
    segments = parse_kpoint_path(lines)
    n_points = 10
    kpath = build_kpath(segments, CUBIC_RECIP, n_points=n_points)

    # 2 segments * 10 points, minus 1 shared point at the G-X/X-G junction
    assert kpath.kpts.shape == (2 * n_points - 1, 3)
    assert kpath.dists.shape == (2 * n_points - 1,)

    # No duplicated k-points at the junction
    junction = kpath.kpts[n_points - 1]
    np.testing.assert_allclose(junction, [0.5, 0.0, 0.0])
    assert not np.allclose(kpath.kpts[n_points - 2], kpath.kpts[n_points - 1])

    # Vertices: G, X, G — continuous path, no combined labels
    assert kpath.tick_labels == ["G", "X", "G"]
    assert kpath.tick_dists[0] == 0.0
    assert np.all(np.diff(kpath.tick_dists) > 0)


def test_build_kpath_distance_is_cartesian():
    # G -> X spans half of the reciprocal x-axis: |0.5 * 2pi| = pi
    lines = ["G 0.00 0.00 0.00   X 0.50 0.00 0.00"]
    segments = parse_kpoint_path(lines)
    kpath = build_kpath(segments, CUBIC_RECIP, n_points=5)
    np.testing.assert_allclose(kpath.dists[-1], np.pi)
    np.testing.assert_allclose(kpath.tick_dists, [0.0, np.pi])


def test_build_kpath_discontinuous_jump_combines_labels():
    # W -> L, then a jump to G -> K (L != G, so no shared point)
    lines = [
        "W 0.50 0.75 0.25   L 0.00 0.50 0.00",
        "G 0.00 0.00 0.00   K 0.00 0.50 -0.50",
    ]
    segments = parse_kpoint_path(lines)
    n_points = 5
    kpath = build_kpath(segments, CUBIC_RECIP, n_points=n_points)

    # No point dropped across the jump: full n_points per segment
    assert kpath.kpts.shape == (2 * n_points, 3)
    assert kpath.tick_labels == ["W", "L|G", "K"]


def test_build_kpath_monotonic_distance():
    lines = [
        "G 0.00 0.00 0.00    X 0.50 0.50 0.00",
        "X 0.50 0.50 0.00    W 0.50 0.75 0.25",
        "W 0.50 0.75 0.25    L 0.00 0.50 0.00",
    ]
    segments = parse_kpoint_path(lines)
    kpath = build_kpath(segments, CUBIC_RECIP, n_points=20)
    assert np.all(np.diff(kpath.dists) >= 0)


def test_build_kpath_empty_raises():
    with pytest.raises(ValueError):
        build_kpath([], CUBIC_RECIP)


def test_total_points_gives_uniform_cartesian_k_density():
    """On an anisotropic cell, a fixed count per segment distorts the plot:
    the short segment gets the same points as the long one, so its dispersion
    is drawn across a sliver of the x-axis (the NiI2 Gamma-A artifact)."""
    import numpy as np
    from waw.analysis.kpath import build_kpath

    # a = 3.9, c = 39 Ang -> |c*| is 10x shorter than |a*|
    recip = np.diag([1.0, 1.0, 0.1])
    segs = [("G", np.array([0., 0., 0.]), "X", np.array([0.5, 0., 0.])),
            ("X", np.array([0.5, 0., 0.]), "Z", np.array([0.5, 0., 0.5]))]

    fixed = build_kpath(segs, recip, n_points=60)
    n_gx = int(np.sum(fixed.dists <= fixed.tick_dists[1] + 1e-12))
    n_xz = len(fixed.dists) - n_gx
    assert abs(n_gx - n_xz) <= 1                      # equal counts...
    len_gx = fixed.tick_dists[1] - fixed.tick_dists[0]
    len_xz = fixed.tick_dists[2] - fixed.tick_dists[1]
    assert len_gx == pytest.approx(10 * len_xz)       # ...on very unequal lengths

    prop = build_kpath(segs, recip, total_points=110)
    n_gx = int(np.sum(prop.dists <= prop.tick_dists[1] + 1e-12))
    n_xz = len(prop.dists) - n_gx
    # points now follow Cartesian length: 10:1, so the k-density is uniform
    d_gx = len_gx / (n_gx - 1)
    d_xz = len_xz / max(n_xz - 1, 1)
    assert d_gx == pytest.approx(d_xz, rel=0.25)
    assert prop.tick_labels == fixed.tick_labels
    assert np.allclose(prop.tick_dists, fixed.tick_dists)


def test_total_points_keeps_at_least_two_points_on_a_tiny_segment():
    import numpy as np
    from waw.analysis.kpath import build_kpath

    recip = np.eye(3)
    segs = [("G", np.array([0., 0., 0.]), "X", np.array([0.5, 0., 0.])),
            ("X", np.array([0.5, 0., 0.]), "Y", np.array([0.5, 0.001, 0.]))]
    kp = build_kpath(segs, recip, total_points=50)
    assert len(kp.kpts) >= 50           # the tiny segment is floored at 2, not 0
    assert np.allclose(kp.kpts[-1], [0.5, 0.001, 0.])
