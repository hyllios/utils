"""Site symmetry combined with a frozen window.

This used to raise outright. What made it awkward is that
`extract_symmetrized_subspace` symmetrizes V's COLUMNS with d_matrix_wann
(nw x nw), and with frozen bands only nw - nf of those columns are free, so
there is no d_matrix_wann to apply. The resolution is that Omega_I does not
depend on the gauge at all -- only on span(V(k)) -- and a subspace transforms
under d_band alone, d_wann cancelling in V V^dag. So the disentanglement stage
symmetrizes the SUBSPACE (which is well defined with any number of frozen bands)
and the Wannier-side gauge is left to the spread-minimization stage, where
nw x nw is the right size.

What must therefore hold, and is measured here rather than argued:
  * the frozen bands stay exactly inside span(V) at every k;
  * P = V V^dag is stabilizer-covariant at self-mapped k and related by d_band
    across a star -- even though the input Mmn is not symmetric, because Z is
    symmetrized before the subspace is chosen;
  * the two data preconditions raise instead of silently producing a
    non-equivariant answer.
"""
import numpy as np
import pytest
import torch

from waw.core.disentangle import disentangle
from waw.core.sitesym import SiteSymmetry

NB, NW, NK = 4, 2, 4
# Band 2 is the one the frozen window catches, at every k. Bands 0 and 1 are
# DEGENERATE on purpose: d_band may only mix bands of equal energy (because
# eps(Sk, n) = eps(k, n)), so a fixture that mixes them has to make them
# degenerate, and a fixture with four distinct energies can only carry a
# diagonal d_band -- too weak to test covariance with.
EIG = torch.tensor([[0.0, 0.0, -5.0, 2.0]] * NK, dtype=torch.float64)
FROZEN = (-6.0, -4.0)
OUTER = (-10.0, 10.0)


def inversion_sitesym(nb=NB, nw=NW):
    """k = 0, .25, .5, .75 with inversion: 0->0, .25<->.75, .5->.5.

    d_band at the two self-mapped k-points is diag(1,-1,1,1) and the 0<->1 row
    swap. Both leave band 2 alone, so {2} is a legitimate frozen set; the swap
    acts non-trivially inside the free block, which is what makes the covariance
    check meaningful rather than vacuous.
    """
    nsym, nkptirr = 2, 3
    Dband = torch.zeros(nb, nb, nsym, nkptirr, dtype=torch.complex128)
    Dband[:, :, 0, :] = torch.eye(nb, dtype=torch.complex128).unsqueeze(-1)
    D0 = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0], dtype=torch.complex128))
    D2 = torch.eye(nb, dtype=torch.complex128)
    D2[[0, 1]] = D2[[1, 0]]
    Dband[:, :, 1, 0] = D0
    Dband[:, :, 1, 2] = D2
    # The element carrying k=.25 -> k=.75 must respect the degenerate blocks:
    # an arbitrary unitary here would mix band 2 into the others, which no real
    # .dmn can do, and would then lose the frozen band at the image k-point.
    rng = np.random.default_rng(0)
    blk = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    D1 = torch.eye(nb, dtype=torch.complex128)
    D1[:2, :2] = torch.tensor(blk, dtype=torch.complex128)     # inside {0, 1}
    D1[2, 2] = np.exp(0.7j)                                   # phases elsewhere
    D1[3, 3] = np.exp(-1.3j)
    Dband[:, :, 1, 1] = D1
    return SiteSymmetry(
        nsymmetry=nsym, nkptirr=nkptirr, num_kpts=NK, num_bands=nb, num_wann=nw,
        ik2ir=torch.tensor([0, 1, 2, 1]), ir2ik=torch.tensor([0, 1, 2]),
        kptsym=torch.tensor([[0, 1, 2], [0, 3, 2]]),
        d_matrix_wann=torch.eye(nw, dtype=torch.complex128
                                ).reshape(nw, nw, 1, 1).expand(nw, nw, nsym, nkptirr
                                                               ).contiguous(),
        d_matrix_band=Dband,
    )


def overlaps(seed=3, nb=NB):
    rng = np.random.default_rng(seed)
    nnb = 2
    M = rng.normal(size=(NK, nnb, nb, nb)) + 1j * rng.normal(size=(NK, nnb, nb, nb))
    Mmn = torch.tensor(M, dtype=torch.complex128)
    kb_idx = torch.tensor([[(k + 1) % NK, (k - 1) % NK] for k in range(NK)])
    wb = torch.ones(nnb, dtype=torch.float64)
    return Mmn, wb, kb_idx


def run(sitesym=None, frozen=FROZEN, nw=NW, eig=EIG, **kw):
    Mmn, wb, kb_idx = overlaps(nb=eig.shape[1])
    return disentangle(Mmn, eig, wb, kb_idx, nw, outer_window=OUTER,
                       frozen_window=frozen, sitesym=sitesym, n_iter=60, **kw)


def test_frozen_plus_sitesym_no_longer_raises():
    res = run(inversion_sitesym())
    assert res.V.shape == (NK, NB, NW)
    assert np.isfinite(res.omega_i)


def test_frozen_bands_stay_inside_the_subspace():
    """The whole point of freezing: P must reproduce the frozen band exactly."""
    res = run(inversion_sitesym())
    P = res.V @ res.V.conj().transpose(-1, -2)
    e2 = torch.zeros(NB, dtype=torch.complex128)
    e2[2] = 1.0
    for ik in range(NK):
        assert torch.allclose(P[ik] @ e2, e2, atol=1e-10), \
            f"frozen band lost at k={ik}: {(P[ik] @ e2 - e2).abs().max()}"


def test_subspace_is_stabilizer_covariant_at_self_mapped_kpoints():
    """P must commute with d_band at k that map to themselves. This holds even
    though Mmn here is NOT symmetric, because Z is symmetrized first -- so it
    tests the symmetrization, not the input."""
    ss = inversion_sitesym()
    res = run(ss)
    P = res.V @ res.V.conj().transpose(-1, -2)
    for ir, ik in ((0, 0), (2, 2)):
        D = ss.d_matrix_band[:, :, 1, ir]
        lhs = D @ P[ik] @ D.conj().transpose(-1, -2)
        assert torch.allclose(lhs, P[ik], atol=1e-9), \
            f"subspace not covariant at k={ik}: {(lhs - P[ik]).abs().max()}"


def test_subspace_is_related_by_d_band_across_a_star():
    ss = inversion_sitesym()
    res = run(ss)
    P = res.V @ res.V.conj().transpose(-1, -2)
    D = ss.d_matrix_band[:, :, 1, 1]                 # maps k=.25 -> k=.75
    got = D @ P[1] @ D.conj().transpose(-1, -2)
    assert torch.allclose(got, P[3], atol=1e-9), (got - P[3]).abs().max()


def test_projector_is_a_projector_of_the_right_rank():
    res = run(inversion_sitesym())
    P = res.V @ res.V.conj().transpose(-1, -2)
    for ik in range(NK):
        assert torch.allclose(P[ik] @ P[ik], P[ik], atol=1e-10)
        assert P[ik].diagonal().real.sum().item() == pytest.approx(NW, abs=1e-9)


def test_symmetrised_run_costs_omega_i_relative_to_the_free_one():
    """Sanity on the objective: imposing symmetry cannot LOWER Omega_I below the
    unconstrained optimum. Catches a symmetrization that silently drops the
    constraint (which would look like a suspiciously good result)."""
    free = run(None)
    sym = run(inversion_sitesym())
    assert sym.omega_i >= free.omega_i - 1e-8


# --- the two preconditions --------------------------------------------------

def test_frozen_set_split_by_a_symmetry_raises():
    """Freezing ONE member of a degenerate pair that d_band mixes must be
    rejected: {0} is not invariant under the 0<->1 swap at k=.5, so neither
    restricting Z to the free block nor broadcasting over the star is valid.
    An energy window cannot express this (0 and 1 are degenerate), but
    projectability can -- which is exactly why the check is on d_band and not
    on the window."""
    Mmn, wb, kb_idx = overlaps()
    Amn = torch.zeros(NK, NB, NW, dtype=torch.complex128)
    Amn[:, 0, 0] = 1.0                       # band 0 fully projectable, alone
    Amn[:, 3, 1] = 0.1
    with pytest.raises(ValueError, match="not symmetry invariant|mixes frozen"):
        disentangle(Mmn, EIG, wb, kb_idx, NW, Amn=Amn, outer_window=OUTER,
                    frozen_window=None, proj_max=0.5,
                    sitesym=inversion_sitesym(), n_iter=5)


def test_star_freezing_different_counts_raises():
    """An energy window cannot really do this (eps is symmetry invariant), but a
    projectability-based frozen set can, and the assumption must be checked."""
    eig = EIG.clone()
    eig[3, 3] = -5.0                     # freeze an extra band at k=.75 only
    with pytest.raises(ValueError, match="freezes different numbers of bands"):
        run(inversion_sitesym(), eig=eig)


def test_degenerate_z_cut_raises():
    """If the eigenvalue cut selecting the free states lands inside a multiplet,
    `eigh` picks an arbitrary basis and the subspace is not equivariant. Forced
    here by making the free block proportional to the identity."""
    ss = inversion_sitesym()
    Mmn, wb, kb_idx = overlaps()
    Mmn = torch.zeros_like(Mmn)                    # Z == 0 -> fully degenerate
    with pytest.raises(ValueError, match="inside a degenerate"):
        disentangle(Mmn, EIG, wb, kb_idx, 3, outer_window=OUTER,
                    frozen_window=FROZEN, sitesym=ss, n_iter=5)


def test_sitesym_without_frozen_bands_still_uses_the_frame_route():
    """The no-frozen path is unchanged and must keep working: it symmetrizes the
    frame with d_matrix_wann, not just the subspace."""
    res = run(inversion_sitesym(), frozen=None)
    assert res.V.shape == (NK, NB, NW)
    P = res.V @ res.V.conj().transpose(-1, -2)
    for ik in range(NK):
        assert torch.allclose(P[ik] @ P[ik], P[ik], atol=1e-10)
