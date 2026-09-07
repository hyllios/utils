"""
Tests for the guiding-centres mechanism (core/spread.py::_guided_phase,
refine_guiding_centres), added to prevent the MLWF "runaway centre"
branch-cut pathology first found on tutorial14's periodic Na chain
(a real Omega_D blow-up: Omega_I matched wannier90.x exactly but Omega_total
was ~9x too large, with a WF centre drifting past a full lattice vector).

Synthetic, self-contained (no DFT fixture needed) -- these test the
phase-unwrapping mathematics directly. The end-to-end reproduction of the
real pathology and its fix lives in tests/test_tutorial14_periodic.py.
"""

from pathlib import Path

import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.spread import _guided_phase, refine_guiding_centres


def test_guided_phase_reduces_to_angle_when_rguide_zero():
    torch.manual_seed(0)
    M_diag = torch.randn(4, 6, 3, dtype=torch.complex128)
    bvecs  = torch.randn(4, 6, 3, dtype=torch.float64)
    rguide = torch.zeros(3, 3, dtype=torch.float64)

    guided = _guided_phase(M_diag, bvecs, rguide)
    naive  = torch.angle(M_diag)
    torch.testing.assert_close(guided, naive)


def test_guided_phase_unwraps_beyond_pi():
    """
    Construct a case where the true phase b.r exceeds pi (impossible for
    torch.angle to represent directly) and check the guided formula, given
    a reference close to the truth, recovers the correct unwrapped value
    rather than the wrapped principal-branch one.
    """
    b = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).reshape(1, 1, 3)
    true_r = torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float64)   # b.r = 4.0 rad > pi
    sheet_true = torch.einsum("kba,na->kbn", b, true_r)          # (1,1,1) = 4.0
    M_diag = torch.exp(-1j * sheet_true).to(torch.complex128)    # phase = -4.0 (wraps to ~2.28)

    naive = torch.angle(M_diag)
    assert abs(naive.item() - (-4.0)) > 1.0   # confirms it's actually wrapped

    # A reference guess close to the truth (as refine_guiding_centres would
    # produce) should recover phase == -4.0 rather than the wrapped value.
    rguide = torch.tensor([[3.9, 0.0, 0.0]], dtype=torch.float64)
    guided = _guided_phase(M_diag, b, rguide)
    assert abs(guided.item() - (-4.0)) < 1e-6


def test_refine_guiding_centres_recovers_synthetic_centre():
    """
    Build synthetic M_nn = exp(-i b.r_true) for several b-directions spanning
    3D, at realistic b-vector/centre magnitudes (b ~ 0.1-0.3 Bohr^-1,
    r ~ Bohr-scale, so b.r < pi for every b-vector individually -- the
    condition this algorithm actually requires, matching Wannier90's own
    "arbitrary branch for the first three" step, which uses the *true*
    M_nn's own wrapped phase for those three regardless of any reference
    guess -- so correctness needs b.r_true < pi there, not merely a good
    guess). This is realistic: the guiding-centres mechanism is a
    *preventive*, frequently-refreshed tracker (Wannier90 refreshes every
    single iteration by default) that keeps drift from ever reaching the
    ambiguous regime, not a one-shot corrector of an arbitrarily large
    accumulated jump.
    """
    torch.manual_seed(0)
    r_true = torch.tensor([[5.3, -2.1, 0.7]], dtype=torch.float64)   # (nw=1, 3), Bohr

    bvecs = 0.25 * torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
    ], dtype=torch.float64).reshape(1, 6, 3)   # nk=1, nnb=6, Bohr^-1

    sheet = torch.einsum("kba,na->kbn", bvecs, r_true)   # (1,6,1); max |sheet| < pi
    assert sheet.abs().max() < torch.pi
    M_diag = torch.exp(-1j * sheet).to(torch.complex128)

    rguide_init = torch.zeros(1, 3, dtype=torch.float64)
    rguide = refine_guiding_centres(M_diag, bvecs, rguide_init)

    torch.testing.assert_close(rguide, r_true, atol=1e-6, rtol=0)
