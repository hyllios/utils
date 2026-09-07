"""
Tests for waw.analysis.viscosity (electronic viscosity, CRTA,
following Phoebe's electron_viscosity.cpp). No external reference exists
(wannier90/BoltzWann has no such observable), so these check physical
invariants only: reality (automatic -- no complex numbers anywhere in
this construction), diagonal positivity (provable, asserted internally),
linear scaling with relax_time, and that BZ-folding actually brings
points closer to the origin.
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis import electronic_viscosity
from waw.analysis.viscosity import _fold_to_first_bz
from waw.units import to_si_units

K_B_HARTREE = 3.166811563e-6
AU_TIME_PER_FS = 1.0 / 2.4188843265857e-2


def _cubic_tb_hr(t: float = -0.01, a: float = 5.0) -> HamiltonianR:
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    H_R = np.zeros((len(R), 1, 1), dtype=complex)
    for i in range(1, 7):
        H_R[i, 0, 0] = t
    return HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(len(R)), nw=1)


def test_eta_is_real_and_finite():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    eta = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                               mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS)
    assert eta.shape == (3, 3, 3, 3)
    assert np.isrealobj(eta)
    assert np.isfinite(eta).all()


def test_eta_diagonal_is_nonnegative():
    """Asserted internally by electronic_viscosity; not raising is the test."""
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    eta = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                               mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS)
    for a in range(3):
        for b in range(3):
            assert eta[a, b, a, b] >= -1e-12


def test_eta_scales_linearly_with_relax_time():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    kwargs = dict(mu=0.0, kT=300 * K_B_HARTREE)
    eta1 = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                                relax_time=5 * AU_TIME_PER_FS, **kwargs)
    eta2 = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                                relax_time=10 * AU_TIME_PER_FS, **kwargs)
    assert eta2 == pytest.approx(2 * eta1, abs=1e-20)


def test_eta_accepts_callable_relax_time():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    tau = 10 * AU_TIME_PER_FS
    eta_const = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                                     mu=0.0, kT=300 * K_B_HARTREE, relax_time=tau)
    eta_callable = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                                        mu=0.0, kT=300 * K_B_HARTREE,
                                        relax_time=lambda eig: np.full_like(eig, tau))
    assert eta_const == pytest.approx(eta_callable)


def test_to_si_is_positive_scale():
    hr = _cubic_tb_hr()
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    eta_au = electronic_viscosity(hr, real_lattice, recip_lattice, (10, 10, 10),
                                  mu=0.0, kT=300 * K_B_HARTREE, relax_time=10 * AU_TIME_PER_FS)
    eta_si = to_si_units(eta_au, "viscosity")
    assert eta_si == pytest.approx(eta_au * 1.054571817e-34 / (5.29177210903e-11) ** 3)


def test_fold_to_first_bz_reduces_norm():
    """A k-point well outside the first BZ must fold to something no
    farther from the origin than its own image at the nearest lattice
    point -- i.e. folding must not increase |k|."""
    real_lattice = np.eye(3) * 5.0
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    kpts_frac = np.array([[1.4, -2.3, 0.6], [0.1, 0.05, -0.05]])
    folded = _fold_to_first_bz(kpts_frac, recip_lattice)
    naive_cart = kpts_frac @ recip_lattice
    assert np.linalg.norm(folded[0]) < np.linalg.norm(naive_cart[0])
    # a point already near the origin should be left essentially alone
    assert np.linalg.norm(folded[1] - naive_cart[1]) < 1e-8


def test_fold_to_first_bz_handles_nonorthogonal_lattice():
    """Hexagonal-like non-orthogonal lattice, where naive per-component
    wrapping is NOT the true Wigner-Seitz cell -- folding must still
    reduce every point's norm below its unfolded value where relevant."""
    a = 5.0
    real_lattice = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]])
    recip_lattice = 2 * np.pi * np.linalg.inv(real_lattice).T
    rng = np.random.default_rng(0)
    kpts_frac = rng.uniform(-3, 3, size=(50, 3))
    kpts_frac[:, 2] = 0.0
    folded = _fold_to_first_bz(kpts_frac, recip_lattice)
    naive_cart = kpts_frac @ recip_lattice
    assert np.all(np.linalg.norm(folded, axis=-1) <= np.linalg.norm(naive_cart, axis=-1) + 1e-9)
