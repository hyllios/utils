"""
Tests for waw.analysis.cpa (single-site CPA for substitutional
alloys in a Wannier basis). Validated against exact / analytic CPA limits:

  1. no disorder (v_A == v_B): Sigma == 0, CPA == VCA;
  2. ATOMIC LIMIT (no hopping): CPA is exact ->
     DOS(E) = x_A delta(E - V_A) + x_B delta(E - V_B), i.e. two peaks at the
     bare on-site levels with weights x_A, x_B;
  3. spectral-weight conservation and positive-definite DOS at all E;
  4. SPLIT-BAND regime (disorder >> bandwidth): two sub-bands carrying
     weights x_A, x_B separated by a (nearly) empty gap;
  5. CPA self-consistency: |sum_a x_a t_a| driven below tol;
  6. Bloch spectral function reduces to the VCA band in the clean limit.
"""

import numpy as np
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.cpa import (
    build_alloy, virtual_crystal, coherent_potential,
    cpa_bloch_spectral_function,
)


def _cubic_1band(e0=0.0, t=-0.5, a=1.0):
    """Single-band simple cubic: H(k) = e0 + 2t(cos kx + cos ky + cos kz)."""
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    H_R = np.zeros((len(R), 1, 1), dtype=complex)
    H_R[0, 0, 0] = e0
    for i in range(1, 7):
        H_R[i, 0, 0] = t
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R,
                      degen=np.ones(len(R)), nw=1)
    return hr, real_lattice


def _trapz(y, x):
    return float(np.trapezoid(y, x))


def test_no_disorder_gives_zero_selfenergy():
    """Identical species -> no scattering -> Sigma == 0, CPA == VCA."""
    hrA, _ = _cubic_1band(e0=0.3, t=-0.5)
    hrB, _ = _cubic_1band(e0=0.3, t=-0.5)
    model = build_alloy([hrA, hrB], [0.5, 0.5])
    E = np.linspace(-4, 4, 60)
    res = coherent_potential(model, (12, 12, 12), E, eta=1e-2)
    assert np.abs(res.Sigma).max() < 1e-6
    assert (res.dos >= -1e-9).all()


def test_atomic_limit_split_dos_exact():
    """No hopping: CPA is exact -> two peaks at the bare on-site levels
    V_A=+2, V_B=-2 with integrated weights x_A=0.3, x_B=0.7."""
    hrA, _ = _cubic_1band(e0=+2.0, t=0.0)
    hrB, _ = _cubic_1band(e0=-2.0, t=0.0)
    xA = 0.3
    model = build_alloy([hrA, hrB], [xA, 1 - xA])
    E = np.linspace(-4, 4, 1601)
    res = coherent_potential(model, (4, 4, 4), E, eta=2e-2)
    dos = res.dos
    # two peaks near +2 and -2
    peakA = E[np.argmax(np.where(E > 0, dos, 0))]
    peakB = E[np.argmax(np.where(E < 0, dos, 0))]
    assert abs(peakA - 2.0) < 0.05
    assert abs(peakB + 2.0) < 0.05
    # integrated weights of each peak
    wA = _trapz(dos[E > 0], E[E > 0])
    wB = _trapz(dos[E < 0], E[E < 0])
    assert abs(wA - xA) < 0.02
    assert abs(wB - (1 - xA)) < 0.02


def test_dos_positive_and_normalized():
    """DOS >= 0 everywhere and integrates to 1 (one band per cell)."""
    hrA, _ = _cubic_1band(e0=+1.0, t=-0.25)
    hrB, _ = _cubic_1band(e0=-1.0, t=-0.25)
    model = build_alloy([hrA, hrB], [0.5, 0.5])
    E = np.linspace(-5, 5, 2001)
    res = coherent_potential(model, (16, 16, 16), E, eta=1e-2)
    assert (res.dos >= -1e-9).all()
    assert abs(_trapz(res.dos, E) - 1.0) < 0.03


def test_split_band_weights():
    """Disorder >> bandwidth (W=6, delta=8) -> two sub-bands of weights
    x_A, x_B separated by an almost-empty gap around E=0."""
    hrA, _ = _cubic_1band(e0=+4.0, t=-0.5)   # W = 12|t| = 6
    hrB, _ = _cubic_1band(e0=-4.0, t=-0.5)
    xA = 0.5
    model = build_alloy([hrA, hrB], [xA, 1 - xA])
    E = np.linspace(-9, 9, 3001)
    res = coherent_potential(model, (16, 16, 16), E, eta=2e-2)
    dos = res.dos
    upper = _trapz(dos[E > 0], E[E > 0])
    lower = _trapz(dos[E < 0], E[E < 0])
    assert abs(upper - xA) < 0.03
    assert abs(lower - (1 - xA)) < 0.03
    # near-empty gap at the band centre
    gap = dos[np.abs(E) < 0.5].max()
    assert gap < 0.05 * dos.max()


def test_self_consistency_residual():
    hrA, _ = _cubic_1band(e0=+1.5, t=-0.5)
    hrB, _ = _cubic_1band(e0=-1.5, t=-0.5)
    model = build_alloy([hrA, hrB], [0.4, 0.6])
    E = np.linspace(-5, 5, 200)
    res = coherent_potential(model, (14, 14, 14), E, eta=5e-3, tol=1e-9)
    assert res.residual.max() < 1e-8


def test_bloch_spectral_reduces_to_vca_band():
    """In the clean limit (tiny disorder) A_B(k, E) peaks at the VCA band
    energy e0 + 2t sum cos(k) at each k."""
    hrA, _ = _cubic_1band(e0=+1e-3, t=-0.5)
    hrB, _ = _cubic_1band(e0=-1e-3, t=-0.5)
    model = build_alloy([hrA, hrB], [0.5, 0.5])
    kpath = np.array([[0.1, 0.2, 0.0], [0.25, 0.0, 0.0], [0.3, 0.3, 0.1]])
    E = np.linspace(-4, 4, 801)
    sf = cpa_bloch_spectral_function(model, kpath, E, (10, 10, 10), eta=2e-2)
    for ik, k in enumerate(kpath):
        eband = 2 * (-0.5) * np.sum(np.cos(2 * np.pi * k))
        epk = E[np.argmax(sf.A[ik])]
        assert abs(epk - eband) < 0.05


def test_virtual_crystal_average():
    hrA, _ = _cubic_1band(e0=+1.0, t=-0.5)
    hrB, _ = _cubic_1band(e0=-3.0, t=-0.5)
    vc = virtual_crystal([hrA, hrB], [0.25, 0.75])
    i0 = int(np.where((np.asarray(vc.R_vectors) == 0).all(axis=1))[0][0])
    assert abs(vc.H_R[i0, 0, 0].item().real - (0.25 * 1.0 + 0.75 * -3.0)) < 1e-12
