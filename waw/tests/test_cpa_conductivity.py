"""
Tests for the CPA Kubo-Greenwood conductivity and its Velicky vertex ladder
(waw/analysis/cpa.py).

The ladder is easy to write plausibly and wrongly, so these are the checks
that caught a real bug rather than decoration:

  1. WARD IDENTITY. The ladder's spectral radius must be 1 -- the diffusion
     pole of particle conservation. Butler, Phys. Rev. B 31, 3260 (1985),
     Eq. (67) requires the pair propagator to EXCLUDE the site-diagonal term,
     because same-site repeated scattering is already inside the single-site
     t-matrix. Without that subtraction the radius comes out at 1.5-3.9 on
     the very same models and the ladder sums nothing. This test fails loudly
     if the subtraction is ever removed.
  2. The vertex correction vanishes as the disorder is switched off.
  3. Clean limit: with a small constant -i*Gamma in place of a CPA medium the
     bubble reproduces the Boltzmann conductivity sum_kn v^2 tau delta(e-eF).
  4. Zero disorder leaves the two species identical, so Sigma = 0 and the
     medium is the virtual crystal.
"""

from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.cpa import (build_alloy, coherent_potential,
                              cpa_conductivity)

A = 6.0


def _cubic_band(nw=1, t=0.25, onsite=None):
    """Simple-cubic nearest-neighbour model, nw orbitals, no hybridisation."""
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                  [0, 0, 1], [0, 0, -1]], dtype=np.int64)
    H = np.zeros((len(R), nw, nw), dtype=np.complex128)
    H[0] = np.diag(np.zeros(nw) if onsite is None else np.asarray(onsite))
    for i in range(1, 7):
        H[i] = -t * np.eye(nw)
    return HamiltonianR(H_R=torch.tensor(H), R_vectors=R,
                        degen=np.ones(len(R), dtype=np.int64), nw=nw,
                        real_lattice=A * np.eye(3))


def _recip():
    return 2 * np.pi / A * np.eye(3)


def test_ward_identity_single_band():
    """The ladder eigenvalue is 1: particle conservation. Butler Eq. (67)."""
    for dv, x, E in ((0.5, 0.5, 0.0), (1.0, 0.5, -1.0), (1.0, 0.3, 0.0)):
        model = build_alloy([_cubic_band(1, onsite=[+dv / 2]),
                             _cubic_band(1, onsite=[-dv / 2])], [x, 1 - x])
        en = np.array([E])
        cpa = coherent_potential(model, (20, 20, 20), en, eta=1e-3, tol=1e-12)
        assert cpa.residual[0] < 1e-10
        c = cpa_conductivity(model, cpa, 0, _recip(), (20, 20, 20))
        assert abs(c.ward - 1.0) < 5e-2, f'ward = {c.ward} for dv={dv}, x={x}'


def test_ward_identity_multiorbital():
    """Same, with orbital structure in the disorder -- the case where the
    pair-space index convention could go wrong without showing up."""
    model = build_alloy([_cubic_band(3, onsite=[0.4, 0.2, 0.2]),
                         _cubic_band(3, onsite=[-0.4, -0.2, -0.2])],
                        [0.5, 0.5])
    en = np.array([0.0])
    cpa = coherent_potential(model, (16, 16, 16), en, eta=1e-3, tol=1e-12)
    c = cpa_conductivity(model, cpa, 0, _recip(), (16, 16, 16))
    assert abs(c.ward - 1.0) < 5e-2, f'ward = {c.ward}'


def test_vertex_correction_vanishes_without_disorder():
    a = _cubic_band(2, onsite=[0.3, -0.1])
    model = build_alloy([a, a], [0.5, 0.5])          # identical species
    en = np.array([0.0])
    cpa = coherent_potential(model, (16, 16, 16), en, eta=2e-3, tol=1e-12)
    assert np.abs(cpa.Sigma[0]).max() < 1e-10        # no disorder, no Sigma
    c = cpa_conductivity(model, cpa, 0, _recip(), (16, 16, 16))
    assert abs(c.vertex_frac) < 1e-8
    assert np.isclose(c.sigma, c.sigma_bubble, rtol=1e-10)


def test_bubble_matches_boltzmann_in_the_clean_limit():
    """With a small constant -i*Gamma the bubble must reproduce
    sigma = (1/V N) sum_kn v_x^2 delta(e - eF) / (2 Gamma).

    Evaluated AWAY from the band centre: the bubble carries A^2 where
    Boltzmann carries tau*delta, and the two agree only over a locally flat
    density of states. E = 0 is the van Hove singularity of a simple-cubic
    band, which is the worst possible place to compare them (22% there).
    """
    from waw.analysis.cpa import CPAResult
    from waw.core.distributions import gaussian_smearing
    from waw.analysis.elph import band_eigensystem
    from waw.analysis.wigner_transport import velocity_matrix
    from waw.analysis.dos import _uniform_mesh

    hr = _cubic_band(1)
    model = build_alloy([hr, hr], [0.5, 0.5])
    mesh, E = (28, 28, 28), -0.55
    for gam in (0.02, 0.01):
        cpa = CPAResult(energies=np.array([float(E)]),
                        Sigma=np.array([[[-1j * gam]]]),
                        G_loc=np.zeros((1, 1, 1), dtype=complex),
                        dos=np.zeros(1), residual=np.zeros(1),
                        iters=np.zeros(1, dtype=int), eta=1e-9)
        cpa.energies[0] = E
        c = cpa_conductivity(model, cpa, 0, _recip(), mesh,
                             vertex_corrections=False)
        k = _uniform_mesh(mesh)
        eig, _ = band_eigensystem(hr, k)
        _, vm = velocity_matrix(hr, k, _recip())
        vx = np.real(np.einsum('knna->kna', vm))[:, :, 0]
        tau = 1.0 / (2.0 * gam)
        vol = A ** 3
        # Lorentzian delta, matching the bubble's own broadening
        lor = (1.0 / np.pi) * gam / ((eig - E) ** 2 + gam ** 2)
        s_bz = float((vx ** 2 * lor).sum()) / len(k) * tau / vol
        assert abs(c.sigma_bubble / s_bz - 1.0) < 0.15, \
            f'KG {c.sigma_bubble:.5g} vs Boltzmann {s_bz:.5g}'
