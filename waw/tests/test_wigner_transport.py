"""
Tests for waw.analysis.wigner_transport (CRTA ab initio Wigner
Transport Equation interband/Zener-tunneling conductivity correction).
The formula implemented is re-derived from Phoebe's actual C++ source
(`wigner_electron.cpp`), NOT the literal printed paper equation -- see
the module docstring for the full derivation and why it differs.

There is no external reference run for this module (no electron-phonon
linewidth pipeline or Bi2Se3/SOC fixture exists here), so these tests
check internal physical invariants only:

  1. Delta_sigma is real (proven by the code re-derivation in the module
     docstring, holds regardless of the CRTA approximation used here).
  2. Delta_sigma's DIAGONAL is provably >= 0 for any Hamiltonian (see
     module docstring's positivity proof) -- checked on a synthetic toy
     model (the same kind of system that falsified the two earlier
     (paper-literal and naively-sign-flipped) formula attempts).
  3. Delta_sigma vanishes as inter-band separation grows far past the
     CRTA linewidth Gamma = 1/relax_time (recovers aiBTE).
  4. Delta_sigma stays finite and smooth through the near-degenerate
     limit (the L'Hopital branch of `_interband_ratio`).
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis import interband_conductivity_correction, velocity_matrix

K_B_HARTREE = 3.166811563e-6
AU_TIME_PER_FS = 1.0 / 2.4188843265857e-2


def _two_band_hr(gap: float, Vc: complex = 0.01, t1: float = 0.02, t2: float = -0.03,
                 a: float = 5.0) -> tuple:
    """1D two-orbital chain with tunable gap and a genuine off-diagonal
    (interband) velocity, since t1 != t2 (see test module docstring)."""
    real_lattice = np.diag([a, 20.0, 20.0])
    E1, E2 = gap / 2, -gap / 2
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    H_R = np.zeros((3, 2, 2), dtype=complex)
    H_R[0] = [[E1, Vc], [np.conj(Vc), E2]]
    H_R[1] = np.diag([t1, t2])
    H_R[2] = np.diag([t1, t2])
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(3), nw=2)
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    return hr, real_lattice, recip_lattice


def test_velocity_matrix_is_hermitian():
    hr, real_lattice, recip_lattice = _two_band_hr(gap=0.05)
    eig, vmat = velocity_matrix(hr, np.array([[0.17, 0.0, 0.0]]), recip_lattice)
    for a in range(3):
        assert vmat[0, :, :, a] == pytest.approx(vmat[0, :, :, a].conj().T, abs=1e-12)


def test_offdiagonal_velocity_nonzero_when_hoppings_differ():
    """t1 != t2 is what makes the eigenbasis rotation pick up off-diagonal
    velocity elements -- the whole point of this synthetic model."""
    hr, real_lattice, recip_lattice = _two_band_hr(gap=0.05)
    eig, vmat = velocity_matrix(hr, np.array([[0.17, 0.0, 0.0]]), recip_lattice)
    assert abs(vmat[0, 0, 1, 0]) > 1e-6


def test_delta_sigma_is_real():
    """Reality is a model-independent Hermiticity identity (see module
    docstring) -- interband_conductivity_correction asserts this
    internally, so simply not raising is the test."""
    hr, real_lattice, recip_lattice = _two_band_hr(gap=0.02, Vc=0.01 * np.exp(1j * 1.3))
    ds = interband_conductivity_correction(
        hr, real_lattice, recip_lattice, (40, 1, 1),
        mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS,
        num_elec_per_state=1,
    )
    assert np.isrealobj(ds)


def test_delta_sigma_diagonal_is_nonnegative_toy_model():
    """The diagonal of Delta_sigma is provably >= 0 (see module docstring)
    -- sweep gap and mu on a genuinely complex-velocity toy model."""
    for gap in [0.001, 0.01, 0.1, 0.5]:
        for mu in [-0.3, 0.0, 0.3]:
            hr, real_lattice, recip_lattice = _two_band_hr(gap=gap, Vc=0.01 * np.exp(1j * 0.7))
            ds = interband_conductivity_correction(
                hr, real_lattice, recip_lattice, (40, 1, 1),
                mu=mu, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS,
                num_elec_per_state=1,
            )
            assert ds[0, 0] >= -1e-10, f"gap={gap}, mu={mu}: {ds[0, 0]}"


def test_delta_sigma_vanishes_for_well_separated_bands():
    hr, real_lattice, recip_lattice = _two_band_hr(gap=2.0)   # Gamma ~ 0.0024 Ha << gap
    ds = interband_conductivity_correction(
        hr, real_lattice, recip_lattice, (40, 1, 1),
        mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS,
        num_elec_per_state=1,
    )
    assert abs(ds[0, 0]) < 1e-6


def test_delta_sigma_grows_as_gap_shrinks_towards_linewidth():
    hr_wide, real_lattice, recip_lattice = _two_band_hr(gap=0.5)
    hr_narrow, _, _ = _two_band_hr(gap=0.005)   # comparable to Gamma ~ 0.0024 Ha
    kwargs = dict(mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS,
                  num_elec_per_state=1)
    ds_wide = interband_conductivity_correction(hr_wide, real_lattice, recip_lattice, (60, 1, 1), **kwargs)
    ds_narrow = interband_conductivity_correction(hr_narrow, real_lattice, recip_lattice, (60, 1, 1), **kwargs)
    assert abs(ds_narrow[0, 0]) > 100 * abs(ds_wide[0, 0])


def test_delta_sigma_continuous_through_near_degenerate_limit():
    """The L'Hopital branch of _interband_ratio must not blow up or
    discontinuously jump as two bands are tuned through exact degeneracy."""
    real_lattice = np.diag([5.0, 20.0, 20.0])
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    kwargs = dict(mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS,
                  num_elec_per_state=1, mesh=(20, 1, 1))
    values = []
    for gap in [1e-3, 1e-6, 0.0, -1e-6, -1e-3]:
        hr, _, _ = _two_band_hr(gap=gap)
        ds = interband_conductivity_correction(hr, real_lattice, recip_lattice, **kwargs)
        assert np.isfinite(ds).all()
        values.append(ds[0, 0])
    # smooth through zero gap: no sign flip / discontinuous jump
    assert max(values) - min(values) < 0.1 * max(abs(v) for v in values)
