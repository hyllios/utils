"""
Tests for waw.analysis.spin_texture (spin-orbit / noncollinear spin
texture from a .spn file), and the .spn reader
(interfaces.wannier90.io.read_spn) and its supporting core machinery
(core.hamiltonian.compute_operator_r).

Validated against an exactly-solvable synthetic model: a two-level system
H = (Delta/2) sigma_z + t sigma_x has eigenvalues +-sqrt((Delta/2)^2+t^2)
and eigenstates whose <sigma_z> expectation value is the standard
closed-form -+ (Delta/2)/sqrt((Delta/2)^2+t^2) -- i.e. the ground state's
spin points opposite the effective field h=(t,0,Delta/2), <sigma>=-h/|h|.
No external wannier90/postw90 reference is used in this file (that
cross-validation is in test_tutorial17.py, against a real noncollinear
+SOC QE/wannier90.x run).
"""

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import compute_hr
from waw.analysis import spin_operator_r, interpolate_spin, spin_colored_bands
from waw.interfaces.wannier90.io import read_spn


def _two_level_setup(delta: float, t: float):
    """H = (delta/2) sigma_z + t sigma_x, in the up/down (Bloch) basis;
    returns (hr, SS_R, kpts) with everything expressed in the ab-initio
    BAND-INDEX basis (matching how a real .spn file is indexed -- see
    read_spn's docstring), i.e. spn is sigma_z rotated into H's eigenbasis."""
    nk, nw = 1, 2
    real_lattice = np.eye(3) * 10.0
    kpts = torch.zeros((nk, 3), dtype=torch.float64)
    mp_grid = (1, 1, 1)

    H0 = np.array([[delta / 2, t], [t, -delta / 2]])
    eigvals, eigvecs = np.linalg.eigh(H0)
    eig = torch.tensor(eigvals[None, :], dtype=torch.float64)
    W = torch.tensor(eigvecs, dtype=torch.complex128).unsqueeze(0)
    hr = compute_hr(W, eig, kpts, mp_grid, real_lattice)

    sigma_z_updown = np.array([[1, 0], [0, -1]])
    sigma_z_bandbasis = eigvecs.conj().T @ sigma_z_updown @ eigvecs
    spn_bloch = np.zeros((nk, nw, nw, 3), dtype=complex)
    spn_bloch[0, :, :, 2] = sigma_z_bandbasis

    SS_R = spin_operator_r(W, spn_bloch, kpts, mp_grid, real_lattice)
    return hr, SS_R, np.array([[0.0, 0.0, 0.0]])


def test_spin_z_matches_two_level_closed_form():
    delta, t = 0.06, 0.02
    hr, SS_R, kpts = _two_level_setup(delta, t)
    spin_z = interpolate_spin(hr, SS_R, kpts, axis=(0, 0, 1))[0]

    expected = delta / 2 / np.sqrt((delta / 2) ** 2 + t ** 2)
    assert spin_z[0] == pytest.approx(-expected, abs=1e-10)   # lower band
    assert spin_z[1] == pytest.approx(expected, abs=1e-10)    # upper band


def test_spin_bounded_by_pauli_eigenvalues():
    """|<S>| <= 1 always (Pauli operator, any axis/state)."""
    for delta, t in [(0.5, 0.001), (0.001, 0.5), (0.1, 0.1)]:
        hr, SS_R, kpts = _two_level_setup(delta, t)
        spin_z = interpolate_spin(hr, SS_R, kpts, axis=(0, 0, 1))
        assert np.all(np.abs(spin_z) <= 1.0 + 1e-10)


def test_pure_zeeman_gives_saturated_spin():
    """t=0: H is already diagonal in sigma_z -- eigenstates are pure spin
    up/down, so <Sz> = +-1 exactly (no mixing to reduce the polarization)."""
    hr, SS_R, kpts = _two_level_setup(delta=0.08, t=0.0)
    spin_z = interpolate_spin(hr, SS_R, kpts, axis=(0, 0, 1))[0]
    assert sorted(np.round(spin_z, 8)) == [-1.0, 1.0]


def test_axis_projection_x_vs_z():
    """At t=0 (pure sigma_z Hamiltonian), eigenstates are sigma_z
    eigenstates -- <Sx> must vanish exactly for both bands (orthogonal
    quantization axis of a pure state)."""
    hr, SS_R, kpts = _two_level_setup(delta=0.08, t=0.0)
    spin_x = interpolate_spin(hr, SS_R, kpts, axis=(1, 0, 0))[0]
    assert spin_x == pytest.approx([0.0, 0.0], abs=1e-10)


def _write_spn_formatted(path, spn, header="test"):
    nk, nb, _, _ = spn.shape
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write(f"{nb} {nk}\n")
        for ik in range(nk):
            for m in range(nb):
                for n in range(m + 1):
                    for c in range(3):
                        v = spn[ik, n, m, c]
                        f.write(f"{v.real:.12f} {v.imag:.12f}\n")


def _write_spn_unformatted(path, spn, header="test"):
    nk, nb, _, _ = spn.shape
    ntri = nb * (nb + 1) // 2

    def rec(f, data_bytes):
        n = len(data_bytes)
        f.write(struct.pack('<i', n))
        f.write(data_bytes)
        f.write(struct.pack('<i', n))

    with open(path, "wb") as f:
        rec(f, header.encode().ljust(60))
        rec(f, struct.pack('<ii', nb, nk))
        for ik in range(nk):
            buf = np.empty((ntri, 3), dtype=complex)
            counter = 0
            for m in range(nb):
                for n in range(m + 1):
                    buf[counter] = spn[ik, n, m, :]
                    counter += 1
            rec(f, buf.astype('<c16').tobytes())


def test_read_spn_formatted_and_unformatted_agree(tmp_path):
    rng = np.random.default_rng(0)
    nb, nk = 4, 3
    spn = np.zeros((nk, nb, nb, 3), dtype=complex)
    for ik in range(nk):
        for c in range(3):
            A = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
            spn[ik, :, :, c] = (A + A.conj().T) / 2   # Hermitian, like a real Pauli operator

    fmt_path = tmp_path / "test_fmt.spn"
    unf_path = tmp_path / "test_unf.spn"
    _write_spn_formatted(fmt_path, spn)
    _write_spn_unformatted(unf_path, spn)

    d_fmt = read_spn(fmt_path)
    d_unf = read_spn(unf_path)
    assert d_fmt["num_bands"] == nb and d_fmt["num_kpts"] == nk
    assert np.allclose(d_fmt["spn"], spn)
    assert np.allclose(d_unf["spn"], spn)


def test_read_spn_hermitian(tmp_path):
    rng = np.random.default_rng(1)
    nb, nk = 3, 1
    spn = np.zeros((nk, nb, nb, 3), dtype=complex)
    for c in range(3):
        A = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        spn[0, :, :, c] = (A + A.conj().T) / 2

    path = tmp_path / "herm.spn"
    _write_spn_formatted(path, spn)
    d = read_spn(path)
    for c in range(3):
        assert d["spn"][0, :, :, c] == pytest.approx(d["spn"][0, :, :, c].conj().T)
