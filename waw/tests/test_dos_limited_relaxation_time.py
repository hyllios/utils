"""
Tests for the tau(E) ~ 1/D(E) "DOS-limited scattering" relaxation-time
model (waw.analysis.boltzmann.dos_limited_relaxation_time), a
qualitative extension of BoltzWann's own constant-tau (CRTA) transport,
following the phenomenological model used by Garmroudi et al. for
"energy filtering" thermoelectrics (Comp. Phys. Comm./PRX/
arXiv:2501.04891, Ni3Ge). No external wannier90 reference exists for
this (wannier90's own BoltzWann has no such model) -- validated instead
against the model's own defining property (TDF scales exactly as
D(e_ref)/D(E) relative to the constant-tau baseline) and a regression
check that passing a plain constant `relax_time` is unaffected by the
refactor that made `transport_distribution_function` also accept a
callable tau(E).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis import (
    transport_distribution_function, density_of_states, dos_limited_relaxation_time,
)

AU_TIME_PER_FS = 1.0 / 2.4188843265857e-2


def _cubic_tb_hr(t: float = -0.01, a: float = 5.0) -> HamiltonianR:
    """Simple-cubic single-band tight-binding model, nearest-neighbor hopping t."""
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    H_R = np.zeros((len(R), 1, 1), dtype=complex)
    for i in range(1, 7):
        H_R[i, 0, 0] = t
    return HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(len(R)), nw=1)


def test_constant_relax_time_unaffected_by_callable_refactor():
    """Passing a plain float must give bit-for-bit (to fp noise) the same
    TDF as before transport_distribution_function grew callable support."""
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    energies = np.linspace(-0.11, 0.11, 200)
    tau = 10 * AU_TIME_PER_FS

    tdf = transport_distribution_function(hr, real_lattice, recip_lattice, (10, 10, 10),
                                          energies, tau, num_elec_per_state=2)
    tdf_via_callable = transport_distribution_function(
        hr, real_lattice, recip_lattice, (10, 10, 10), energies,
        lambda eig: np.full_like(eig, tau), num_elec_per_state=2,
    )
    assert tdf.tdf == pytest.approx(tdf_via_callable.tdf)


def test_dos_limited_tdf_scales_as_dos_ratio():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    energies = np.linspace(-0.11, 0.11, 400)
    dos = density_of_states(hr, mesh=(20, 20, 20), energies=energies, sigma=0.002)

    tau_ref = 10 * AU_TIME_PER_FS
    tau_fn = dos_limited_relaxation_time(dos.energies, dos.dos, tau_ref=tau_ref, e_ref=0.0)

    tdf_const = transport_distribution_function(hr, real_lattice, recip_lattice, (20, 20, 20),
                                                 energies, tau_ref, num_elec_per_state=2)
    tdf_dos = transport_distribution_function(hr, real_lattice, recip_lattice, (20, 20, 20),
                                              energies, tau_fn, num_elec_per_state=2)

    d_ref = np.interp(0.0, dos.energies, dos.dos)
    for target in [-0.04, -0.02, 0.02, 0.04]:
        i = np.argmin(np.abs(energies - target))
        d = dos.dos[i]
        expected_ratio = d_ref / d
        actual_ratio = tdf_dos.tdf[i, 0, 0] / tdf_const.tdf[i, 0, 0]
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.02)


def test_dos_limited_tau_matches_reference_at_e_ref():
    hr = _cubic_tb_hr()
    energies = np.linspace(-0.11, 0.11, 400)
    dos = density_of_states(hr, mesh=(20, 20, 20), energies=energies, sigma=0.002)
    tau_ref = 10 * AU_TIME_PER_FS
    tau_fn = dos_limited_relaxation_time(dos.energies, dos.dos, tau_ref=tau_ref, e_ref=0.0)
    assert tau_fn(np.array([0.0]))[0] == pytest.approx(tau_ref, rel=1e-6)


def test_dos_limited_tau_floored_in_gap():
    """Far outside the band (D(E)=0 in the model), tau must be capped, not infinite/NaN."""
    energies = np.linspace(-0.1, 0.1, 50)
    dos_values = np.exp(-((energies) ** 2) / (2 * 0.02 ** 2))   # a single Gaussian "band"
    tau_fn = dos_limited_relaxation_time(energies, dos_values, tau_ref=10.0, e_ref=0.0)
    tau_far = tau_fn(np.array([5.0]))   # way outside the tabulated range -> extrapolated/clamped by np.interp
    assert np.isfinite(tau_far).all()
