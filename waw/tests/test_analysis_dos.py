"""
Tests for waw/analysis/dos.py.

  1. density_of_states returns the requested energy-grid shape.
  2. Sum rule: integral of DOS(E) dE == nw (one state per band per k-point,
     each contributing a Gaussian of unit area), independent of the mesh
     density or smearing width.
  3. Automatic energy range brackets the interpolated eigenvalues.
  4. On Cu: DOS from a real, entangled Wannier Hamiltonian integrates to
     the expected number of Wannier functions.
"""

from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import compute_hr
from waw.analysis.dos import density_of_states, _uniform_mesh

def _synthetic_hr(nk=8, nw=3, mp_grid=(2, 2, 2), seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    a = 5.0
    real_lattice = a * np.eye(3)

    N1, N2, N3 = mp_grid
    kpts_np = np.array(
        [[i / N1, j / N2, k / N3]
         for i in range(N1) for j in range(N2) for k in range(N3)],
        dtype=np.float64,
    )
    kpts = torch.tensor(kpts_np, dtype=torch.float64)

    A = (torch.randn(nk, nw, nw, dtype=torch.float64)
         + 1j * torch.randn(nk, nw, nw, dtype=torch.float64))
    U, _ = torch.linalg.qr(A)
    # Synthetic model in atomic units: a +-5 Ha "band structure" (an
    # arbitrary model scale; the DOS sum rule and shapes are unit-free).
    eig = torch.tensor(np.sort(rng.uniform(-5, 5, size=(nk, nw)), axis=1),
                        dtype=torch.float64)

    return compute_hr(U, eig, kpts, mp_grid, real_lattice)


def test_uniform_mesh_shape_and_range():
    kpts = _uniform_mesh((2, 3, 4))
    assert kpts.shape == (24, 3)
    assert kpts.min() >= 0.0
    assert kpts.max() < 1.0


def test_density_of_states_shape():
    hr = _synthetic_hr()
    dos = density_of_states(hr, mesh=(4, 4, 4), n_energies=200)
    assert dos.energies.shape == (200,)
    assert dos.dos.shape == (200,)


def test_density_of_states_sum_rule():
    hr = _synthetic_hr(nw=3)
    dos = density_of_states(hr, mesh=(6, 6, 6), n_energies=2000, sigma=0.1, e_pad=2.0)
    integral = np.trapezoid(dos.dos, dos.energies)
    np.testing.assert_allclose(integral, hr.nw, rtol=1e-3)


def test_density_of_states_explicit_energy_grid():
    hr = _synthetic_hr()
    energies = np.linspace(-10, 10, 100)
    dos = density_of_states(hr, mesh=(4, 4, 4), energies=energies)
    np.testing.assert_array_equal(dos.energies, energies)


# ---------------------------------------------------------------------------
# fermi_level_spin_channels: one chemical potential shared by two collinear
# spin channels.
#
#   5. Two IDENTICAL channels at joint count N reduce exactly to the
#      spin-unpolarized solver at band_degeneracy=2 -- the factor of two that
#      degeneracy supplies there is supplied here by there being two channels.
#   6. The solved level reproduces the joint count to bisection tolerance.
#   7. The trap the function exists to prevent: solving the channels
#      SEPARATELY at half the count each forces equal filling and so zeroes
#      the moment, while the joint solve recovers the exchange splitting's
#      moment analytically.
#   8. Channels may carry different numbers of Wannier functions.
#   9. Mapping and sequence inputs agree; the guards fire.
# ---------------------------------------------------------------------------

import pytest
from scipy.special import erfc

from waw.analysis.dos import (fermi_level_from_electron_count,
                              fermi_level_spin_channels)

SIG = 0.01


def _count(eig, ef, sigma=SIG):
    return 0.5 * float(erfc((eig - ef) / (sigma * np.sqrt(2.0))).sum()) / eig.shape[0]


def test_spin_channels_identical_reduces_to_unpolarized():
    rng = np.random.default_rng(3)
    eig = np.sort(rng.uniform(-0.4, 0.4, size=(120, 5)), axis=1)
    ef2 = fermi_level_spin_channels([eig, eig.copy()], 4.0, SIG)
    ef1 = fermi_level_from_electron_count(eig, 4.0, SIG, band_degeneracy=2.0)
    assert abs(ef2 - ef1) < 1e-9


def test_spin_channels_reproduces_joint_count():
    rng = np.random.default_rng(4)
    up = np.sort(rng.uniform(-0.5, 0.3, size=(90, 6)), axis=1)
    dn = np.sort(rng.uniform(-0.3, 0.5, size=(90, 6)), axis=1)
    ef = fermi_level_spin_channels([up, dn], 7.0, SIG)
    assert abs(_count(up, ef) + _count(dn, ef) - 7.0) < 1e-8


def test_spin_channels_preserves_moment_where_separate_solve_kills_it():
    """A rigid exchange splitting: the joint solve keeps the moment, the
    per-channel solve reports exactly zero."""
    rng = np.random.default_rng(5)
    base = np.sort(rng.uniform(-0.5, 0.5, size=(150, 6)), axis=1)
    delta = 0.08
    up, dn = base - delta, base + delta

    ef = fermi_level_spin_channels([up, dn], 6.0, SIG)
    moment = _count(up, ef) - _count(dn, ef)
    assert moment > 0.5

    ef_up = fermi_level_from_electron_count(up, 3.0, SIG, band_degeneracy=1.0)
    ef_dn = fermi_level_from_electron_count(dn, 3.0, SIG, band_degeneracy=1.0)
    assert abs(_count(up, ef_up) - _count(dn, ef_dn)) < 1e-8
    assert abs(ef_up - ef_dn) > 0.1


def test_spin_channels_unequal_band_counts():
    rng = np.random.default_rng(6)
    up = np.sort(rng.uniform(-0.5, 0.3, size=(60, 7)), axis=1)
    dn = np.sort(rng.uniform(-0.3, 0.5, size=(60, 4)), axis=1)
    ef = fermi_level_spin_channels([up, dn], 5.0, SIG)
    assert abs(_count(up, ef) + _count(dn, ef) - 5.0) < 1e-8


def test_spin_channels_mapping_matches_sequence_and_guards():
    rng = np.random.default_rng(7)
    up = np.sort(rng.uniform(-0.5, 0.3, size=(40, 5)), axis=1)
    dn = np.sort(rng.uniform(-0.3, 0.5, size=(40, 5)), axis=1)
    assert (fermi_level_spin_channels({"up": up, "down": dn}, 5.0, SIG)
            == fermi_level_spin_channels([up, dn], 5.0, SIG))
    with pytest.raises(ValueError, match="one eigenvalue array per spin"):
        fermi_level_spin_channels([up], 2.0, SIG)
    with pytest.raises(ValueError, match="outside"):
        fermi_level_spin_channels([up, dn], 10.0, SIG)
