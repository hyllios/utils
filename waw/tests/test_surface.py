"""
Tests for waw.analysis.surface (semi-infinite (hkl) surface spectral
function via Sancho-Rubio decimation). Validated via:

  1. geometry: the (hkl) transformation C is unimodular and its in-plane
     rows are perpendicular to the surface normal (cubic 100/110/111);
  2. the spectral function is real, non-negative, and finite;
  3. ANALYTIC: for a single-band simple-cubic tight-binding model, the
     (001) bulk-projected spectral function A_bulk(k_par, E) is non-zero
     exactly inside the known kz-broadened bulk band window
     [e0 - 2|t|, e0 + 2|t|], e0 = 2t(cos kx + cos ky), and vanishes
     outside it.
"""

import numpy as np
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.surface import (
    surface_transformation, build_surface_layers, surface_spectral_function,
)


def _cubic_1band(t=-0.5, a=1.0):
    """Single-band simple-cubic TB: H(k) = 2t (cos kx + cos ky + cos kz)."""
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    H_R = np.zeros((len(R), 1, 1), dtype=complex)
    for i in range(1, 7):
        H_R[i, 0, 0] = t
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(len(R)), nw=1)
    return hr, real_lattice


def test_surface_transformation_cubic():
    A = np.eye(3)
    for mi in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 2, 0)]:
        C = surface_transformation(A, mi)
        assert abs(round(np.linalg.det(C))) == 1
        B = C @ A
        normal = np.cross(B[0], B[1])
        # in-plane rows perpendicular to the surface normal
        assert abs(np.dot(B[0], normal)) < 1e-9
        assert abs(np.dot(B[1], normal)) < 1e-9
        # stacking vector has a non-zero component along the normal
        assert abs(np.dot(B[2], normal)) > 1e-9


def test_surface_transformation_reduces_miller():
    A = np.eye(3)
    assert np.array_equal(surface_transformation(A, (2, 2, 0)),
                          surface_transformation(A, (1, 1, 0)))


def test_layer_blocks_hermitian():
    hr, A = _cubic_1band()
    layers = build_surface_layers(hr, A, (0, 0, 1))
    H00, H01 = layers.blocks(np.array([0.13, 0.27]))
    assert np.allclose(H00, H00.conj().T, atol=1e-12)


def test_spectral_function_real_nonneg_finite():
    hr, A = _cubic_1band()
    energies = np.linspace(-3.5, 3.5, 40)
    kpath = np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.0]])
    sf = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.03)
    for X in (sf.A_surface, sf.A_bulk):
        assert np.isrealobj(X)
        assert np.isfinite(X).all()
        assert (X >= -1e-9).all()


def test_bulk_projected_matches_analytic_band_window():
    """A_bulk(k_par, E) must be non-zero only inside the kz-broadened bulk
    band [e0 - 2|t|, e0 + 2|t|], e0 = 2t(cos kx + cos ky)."""
    t = -0.5
    hr, A = _cubic_1band(t=t)
    kx = 2 * np.pi * 0.2     # cartesian k = 2pi * frac (a=1)
    ky = 2 * np.pi * 0.1
    e0 = 2 * t * (np.cos(kx) + np.cos(ky))
    lo, hi = e0 - 2 * abs(t), e0 + 2 * abs(t)

    energies = np.linspace(-4, 4, 400)
    kpath = np.array([[0.2, 0.1]])
    sf = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.02)
    A_bulk = sf.A_bulk[0]

    inside = (energies > lo + 0.15) & (energies < hi - 0.15)
    outside = (energies < lo - 0.3) | (energies > hi + 0.3)
    # substantial continuum weight inside the band, negligible outside
    assert A_bulk[inside].max() > 0.3
    assert A_bulk[outside].max() < 0.05 * A_bulk[inside].max()


def _cubic_2band_spin(t=-0.5, dz=0.6, a=1.0):
    """Two 'spin' bands (spatial x up/down) on simple cubic, split by an
    exchange field dz*sigma_z; spin operator SS(R) is sigma_z on-site."""
    real_lattice = np.eye(3) * a
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    I2 = np.eye(2)
    sz = np.diag([1.0, -1.0])
    H_R = np.zeros((len(R), 2, 2), dtype=complex)
    H_R[0] = dz * sz                       # on-site exchange split
    for i in range(1, 7):
        H_R[i] = t * I2                    # spin-diagonal hopping
    hr = HamiltonianR(H_R=torch.tensor(H_R), R_vectors=R, degen=np.ones(len(R)), nw=2)
    # SS(R): only on-site sigma_z, three Pauli components (x, y, z)
    ss = np.zeros((len(R), 2, 2, 3), dtype=complex)
    ss[0, :, :, 0] = np.array([[0, 1], [1, 0]])       # sigma_x
    ss[0, :, :, 1] = np.array([[0, -1j], [1j, 0]])    # sigma_y
    ss[0, :, :, 2] = sz                               # sigma_z
    return hr, real_lattice, torch.tensor(ss)


def test_spin_resolved_sum_rule_and_split():
    """A_up + A_dn must equal A_surface (partition of unity), and the +z /
    -z channels must peak at the exchange-split band centres."""
    hr, A, ss = _cubic_2band_spin(t=-0.5, dz=0.6)
    energies = np.linspace(-4, 4, 200)
    kpath = np.array([[0.0, 0.0], [0.25, 0.1]])
    sf = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.05,
                                   spin_op_r=ss, spin_axis=(0, 0, 1))
    assert sf.A_up is not None and sf.A_dn is not None
    assert np.allclose(sf.A_up + sf.A_dn, sf.A_surface, atol=1e-9)
    # the two channels must differ (exchange-split into +z / -z manifolds)
    assert not np.allclose(sf.A_up[0], sf.A_dn[0], atol=1e-3)
    wup = sf.A_up[0]; wdn = sf.A_dn[0]
    # centre of mass of each channel in energy differs by ~2*dz
    com_up = (energies * wup).sum() / wup.sum()
    com_dn = (energies * wdn).sum() / wdn.sum()
    assert abs(abs(com_up - com_dn) - 2 * 0.6) < 0.2


def _two_sublayer(eA=2.0, eB=-2.0, tz=-0.4, tp=-0.15, a=1.0):
    """Cubic cell, 2 orbitals stacked along z: A at height 0 (on-site eA),
    B at height 1/2 (on-site eB), coupled A_n-B_n and B_n-A_{n+1} along z,
    both dispersing in-plane. The (001) surface has two distinct
    terminations (A-exposed vs B-exposed)."""
    real_lattice = np.eye(3) * a
    centres = np.array([[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * a]])
    R, H = [], []

    def add(r, h):
        R.append(r)
        H.append(np.array(h, dtype=complex))

    add((0, 0, 0), [[eA, tz], [tz, eB]])
    add((0, 0, 1), [[0, 0], [tz, 0]])
    add((0, 0, -1), [[0, tz], [0, 0]])
    for r in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
        add(r, [[tp, 0], [0, tp]])
    hr = HamiltonianR(H_R=torch.tensor(np.array(H)), R_vectors=np.array(R),
                      degen=np.ones(len(R)), nw=2)
    return hr, real_lattice, centres


def test_wf_sublayers_two_planes():
    from waw.analysis.surface import wf_sublayers
    _, A, centres = _two_sublayer()
    frac_z, heights = wf_sublayers(A, (0, 0, 1), centres)
    assert len(heights) == 2
    assert np.allclose(np.sort(heights), [0.0, 0.5], atol=1e-6)


def test_termination_requires_wf_centres():
    hr, A = _cubic_1band()
    with pytest.raises(ValueError):
        surface_spectral_function(hr, A, (0, 0, 1), np.array([[0.1, 0.2]]),
                                  np.linspace(-3, 3, 5), termination=1)


def test_termination_exposes_distinct_sublayers():
    """The two (001) terminations of the 2-sublayer model expose the A vs B
    sublayer -- their surface spectral functions must peak near the two
    different on-site energies eA=+2 (A-exposed) and eB=-2 (B-exposed)."""
    hr, A, centres = _two_sublayer(eA=2.0, eB=-2.0)
    energies = np.linspace(-4, 4, 400)
    kpath = np.array([[0.0, 0.0]])
    sf0 = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.03,
                                    hr_cutoff=1e-9, termination=0, wf_centres=centres)
    sf1 = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.03,
                                    hr_cutoff=1e-9, termination=1, wf_centres=centres)
    # the two terminations give genuinely different surface spectra
    assert np.abs(sf0.A_surface - sf1.A_surface).max() > 0.5
    # each termination's dominant surface peak sits near a different on-site level
    peak0 = energies[np.argmax(sf0.A_surface[0])]
    peak1 = energies[np.argmax(sf1.A_surface[0])]
    assert abs(peak0 - peak1) > 2.0
    assert (peak0 < 0 < peak1) or (peak1 < 0 < peak0)


def test_termination_bulk_projection_unaffected():
    """A_bulk (both leads) is the same physical bulk regardless of termination."""
    hr, A, centres = _two_sublayer()
    energies = np.linspace(-4, 4, 60)
    kpath = np.array([[0.1, 0.2]])
    sf0 = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.05,
                                    hr_cutoff=1e-9, termination=0, wf_centres=centres)
    sf1 = surface_spectral_function(hr, A, (0, 0, 1), kpath, energies, eta=0.05,
                                    hr_cutoff=1e-9, termination=1, wf_centres=centres)
    assert np.allclose(sf0.A_bulk, sf1.A_bulk, atol=1e-6)
