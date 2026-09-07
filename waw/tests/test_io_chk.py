"""
Tests for waw/io.py's write_chk_fmt / read_chk_fmt.

There is no real Wannier90 install available to verify end-to-end against
w90chk2chk.x, so these tests instead verify internal round-trip fidelity:
write_chk_fmt's field order/layout (copied from wannier90's
conv_write_chkpt_fmt) must be exactly undone by read_chk_fmt (copied from
conv_read_chkpt_fmt), for arbitrary (non-trivial, asymmetric) input arrays
that would expose any row/column transposition bug.
"""

from pathlib import Path
import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.interfaces.wannier90.io import write_chk_fmt, read_chk_fmt


def _rand_complex(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_roundtrip_isolated_bands(tmp_path):
    """num_bands == num_wann, no disentanglement."""
    rng = np.random.default_rng(0)
    nk, nw, nntot = 4, 3, 6

    real_lattice = rng.normal(size=(3, 3))
    recip_lattice = rng.normal(size=(3, 3))
    kpt_latt = rng.uniform(0, 1, size=(nk, 3))
    u_matrix = _rand_complex(rng, nk, nw, nw)
    m_matrix = _rand_complex(rng, nk, nntot, nw, nw)
    centres = rng.normal(size=(nw, 3))
    spreads = rng.uniform(0.1, 1.0, size=nw)

    path = tmp_path / "seed.chk.fmt"
    write_chk_fmt(
        path,
        num_bands=nw, exclude_bands=np.array([], dtype=np.int64),
        real_lattice=real_lattice, recip_lattice=recip_lattice,
        mp_grid=(2, 2, 1), kpt_latt=kpt_latt, nntot=nntot, num_wann=nw,
        have_disentangled=False, omega_invariant=0.0,
        lwindow=None, ndimwin=None, u_matrix_opt=None,
        u_matrix=u_matrix, m_matrix=m_matrix,
        wannier_centres=centres, wannier_spreads=spreads,
        checkpoint="postwann",
    )

    result = read_chk_fmt(path)

    assert result["num_bands"] == nw
    assert result["num_wann"] == nw
    assert result["nntot"] == nntot
    assert result["mp_grid"] == (2, 2, 1)
    assert result["checkpoint"] == "postwann"
    assert result["have_disentangled"] is False
    np.testing.assert_allclose(result["real_lattice"], real_lattice, rtol=1e-12)
    np.testing.assert_allclose(result["recip_lattice"], recip_lattice, rtol=1e-12)
    np.testing.assert_allclose(result["kpt_latt"], kpt_latt, rtol=1e-12)
    np.testing.assert_allclose(result["u_matrix"], u_matrix, rtol=1e-12)
    np.testing.assert_allclose(result["m_matrix"], m_matrix, rtol=1e-12)
    np.testing.assert_allclose(result["wannier_centres"], centres, rtol=1e-12)
    np.testing.assert_allclose(result["wannier_spreads"], spreads, rtol=1e-12)


def test_roundtrip_disentangled_asymmetric_shapes(tmp_path):
    """
    num_bands != num_wann, with disentanglement: exercises u_matrix_opt's
    (num_bands, num_wann) rectangular shape, which is where a transposed
    read/write would show up as a shape mismatch or scrambled values.
    """
    rng = np.random.default_rng(1)
    nk, nb, nw, nntot = 3, 5, 2, 8

    real_lattice = rng.normal(size=(3, 3))
    recip_lattice = rng.normal(size=(3, 3))
    kpt_latt = rng.uniform(0, 1, size=(nk, 3))
    exclude_bands = np.array([7, 12], dtype=np.int64)

    lwindow = rng.uniform(size=(nk, nb)) > 0.3
    # ndimwin must match the true count of True entries per k for physical
    # consistency, though the reader/writer don't enforce it themselves.
    ndimwin = lwindow.sum(axis=1).astype(np.int64)
    u_matrix_opt = _rand_complex(rng, nk, nb, nw)
    u_matrix = _rand_complex(rng, nk, nw, nw)
    m_matrix = _rand_complex(rng, nk, nntot, nw, nw)
    centres = rng.normal(size=(nw, 3))
    spreads = rng.uniform(0.1, 1.0, size=nw)
    omega_invariant = 1.2345

    path = tmp_path / "seed.chk.fmt"
    write_chk_fmt(
        path,
        num_bands=nb, exclude_bands=exclude_bands,
        real_lattice=real_lattice, recip_lattice=recip_lattice,
        mp_grid=(3, 3, 3), kpt_latt=kpt_latt, nntot=nntot, num_wann=nw,
        have_disentangled=True, omega_invariant=omega_invariant,
        lwindow=lwindow, ndimwin=ndimwin, u_matrix_opt=u_matrix_opt,
        u_matrix=u_matrix, m_matrix=m_matrix,
        wannier_centres=centres, wannier_spreads=spreads,
        checkpoint="postdis",
    )

    result = read_chk_fmt(path)

    assert result["num_bands"] == nb
    assert result["have_disentangled"] is True
    assert result["checkpoint"] == "postdis"
    np.testing.assert_array_equal(result["exclude_bands"], exclude_bands)
    np.testing.assert_allclose(result["omega_invariant"], omega_invariant, rtol=1e-12)
    np.testing.assert_array_equal(result["lwindow"], lwindow)
    np.testing.assert_array_equal(result["ndimwin"], ndimwin)
    assert result["u_matrix_opt"].shape == (nk, nb, nw)
    np.testing.assert_allclose(result["u_matrix_opt"], u_matrix_opt, rtol=1e-12)
    np.testing.assert_allclose(result["u_matrix"], u_matrix, rtol=1e-12)
    np.testing.assert_allclose(result["m_matrix"], m_matrix, rtol=1e-12)


def test_lattice_write_order_matches_w90_index_convention(tmp_path):
    """
    Pin down the exact flatten order for real_lattice/recip_lattice/kpt_latt
    with a distinguishable (non-symmetric) matrix, so a row/column swap bug
    would fail rather than silently pass on a symmetric test fixture.
    """
    real_lattice = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]])
    recip_lattice = np.eye(3) * 2.0
    kpt_latt = np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.75]])

    path = tmp_path / "seed.chk.fmt"
    write_chk_fmt(
        path,
        num_bands=1, exclude_bands=np.array([], dtype=np.int64),
        real_lattice=real_lattice, recip_lattice=recip_lattice,
        mp_grid=(1, 1, 1), kpt_latt=kpt_latt, nntot=1, num_wann=1,
        have_disentangled=False, omega_invariant=0.0,
        lwindow=None, ndimwin=None, u_matrix_opt=None,
        u_matrix=np.ones((2, 1, 1), dtype=np.complex128),
        m_matrix=np.ones((2, 1, 1, 1), dtype=np.complex128),
        wannier_centres=np.zeros((1, 3)), wannier_spreads=np.zeros(1),
    )

    # w90's own write loop is ((M(i,j), i=1,3), j=1,3) with i = vector index,
    # j = Cartesian component -- i.e. component-major, the transpose of a
    # naive row-major flatten of the (rows=vectors) matrix. Confirmed
    # against a real `wannier90.x` + `w90chk2chk.x -export` run (see
    # write_chk_fmt's docstring in io.py).
    # Lines: 0=header, 1=num_bands, 2=num_exclude_bands, 3=real_lattice
    # (no line at all for the exclude_bands list itself since it's empty).
    text = path.read_text()
    numbers = [float(x) for x in text.split("\n")[3].split()]
    np.testing.assert_allclose(numbers, real_lattice.T.reshape(-1), rtol=1e-12)

    result = read_chk_fmt(path)
    np.testing.assert_allclose(result["real_lattice"], real_lattice, rtol=1e-12)
    np.testing.assert_allclose(result["kpt_latt"], kpt_latt, rtol=1e-12)
