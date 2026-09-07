"""
`bands_along_path`'s cache must key on the k-points, not just on completion.

A finished run left over from a DIFFERENT k-path was being reused silently, so
the caller received ab-initio energies belonging to other k-points. Notebook 3
happened to catch it only because the two paths differed in LENGTH, which raised;
with matching lengths it would have returned a plausible, wrong reference curve.
"""

import numpy as np
import pytest

from waw.interfaces.quantum_espresso.bands import _cache_matches, _cached_kpoints


def _write_run(tmp_path, kpts, done=True):
    seed = "sys"
    lines = [f"K_POINTS crystal\n{len(kpts)}"]
    lines += [f"  {k[0]:.10f}  {k[1]:.10f}  {k[2]:.10f}  1.0" for k in kpts]
    (tmp_path / f"{seed}.bands.in").write_text("\n".join(lines) + "\n")
    (tmp_path / f"{seed}.bands.out").write_text(
        "     bands (ev):\n" + ("JOB DONE.\n" if done else "crashed\n"))
    return seed


def test_cache_is_reused_only_for_the_same_kpoints(tmp_path):
    kpts = np.linspace(0, 0.5, 12)[:, None] * np.array([1.0, 0.0, 0.0])
    seed = _write_run(tmp_path, kpts)
    assert _cache_matches(tmp_path, seed, kpts)

    shifted = kpts.copy()
    shifted[3, 1] += 1e-3
    assert not _cache_matches(tmp_path, seed, shifted)     # same length, moved
    assert not _cache_matches(tmp_path, seed, kpts[:-1])   # different length


def test_unfinished_run_is_not_reused(tmp_path):
    kpts = np.zeros((4, 3))
    seed = _write_run(tmp_path, kpts, done=False)
    assert not _cache_matches(tmp_path, seed, kpts)


def test_missing_input_defeats_the_cache(tmp_path):
    kpts = np.zeros((4, 3))
    seed = _write_run(tmp_path, kpts)
    (tmp_path / f"{seed}.bands.in").unlink()
    assert not _cache_matches(tmp_path, seed, kpts)        # cannot verify -> rerun


def test_kpoint_parser_reads_the_declared_count_only(tmp_path):
    (tmp_path / "x.bands.in").write_text(
        "&control\n/\nK_POINTS crystal\n2\n"
        " 0.1 0.2 0.3 1.0\n 0.4 0.5 0.6 1.0\nHUBBARD ortho-atomic\n")
    k = _cached_kpoints(tmp_path / "x.bands.in")
    assert k.shape == (2, 3)
    assert k[1] == pytest.approx([0.4, 0.5, 0.6])
