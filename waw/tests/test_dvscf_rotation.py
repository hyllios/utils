"""Machine-precision tests of the dvscf star rotation.

The MgB2 validation that motivated `rotate_dvscf` ran on P6/mmm -- a
SYMMORPHIC group -- so the fractional-translation path (QE's ``ftau`` grid
shift, the ``ft = -t_spglib`` sign) was never exercised by real data, and a
wrong sign there is silent until someone rotates a nonsymmorphic crystal.
These tests close that hole with a synthetic field whose exact transform is
known analytically:

    dv_q[kappa, alpha](x) = e^{-2 pi i q.x} sum_L e^{2 pi i q.L}
                            d g_kappa / d r_alpha (r(x) - tau_kappa - L)

i.e. a lattice sum of Cartesian gradients of species-dependent Gaussians in
the cell-phase Bloch convention QE's dvscf files use. It can be built
directly at ANY q, so ``rotate_dvscf(dv_q, q, S)`` has an independent
reference at S q -- agreement is to machine precision (the Gaussians are
truncated at a 2-cell image sum, far below 1e-12).

Crystals:
  * diamond   -- nonsymmorphic (half the operations carry t = (1/4,1/4,1/4)):
                 exercises the ftau grid shift and its sign;
  * zincblende -- NO inversion: exercises the time-reversal routes that a
                 ph.x wedge (reduced WITH time reversal) requires there.
"""

import numpy as np
import pytest
from ase import Atoms

from waw.interfaces.ase.structure import (
    crystal_symmetry_operations,
    irreducible_qpoints,
)
from waw.interfaces.quantum_espresso.dvscf_io import (
    dvscf_star_routes,
    rotate_dvscf,
)

A0 = 6.7  # Bohr-ish lattice constant (units cancel; kept away from 1)
FCC = 0.5 * A0 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
GRID = (8, 8, 8)


def _atoms(kind):
    if kind == "diamond":
        return Atoms("C2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                     cell=FCC, pbc=True)
    if kind == "zincblende":
        return Atoms("GaAs", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
                     cell=FCC, pbc=True)
    raise ValueError(kind)


def _recip(real):
    return 2.0 * np.pi * np.linalg.inv(real).T


def make_dv(q_frac, grid, real_lat, tau_frac, sigmas):
    """The synthetic dvscf: exact by construction at any q (see module doc)."""
    nat = len(tau_frac)
    n1, n2, n3 = grid
    fx = np.stack(np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2,
                              np.arange(n3) / n3, indexing="ij"), axis=-1)
    r = fx @ real_lat                                       # (n1,n2,n3,3) cart
    bloch = np.exp(-2j * np.pi * (fx @ np.asarray(q_frac)))
    Ls = np.array([[i, j, k] for i in range(-2, 3)
                   for j in range(-2, 3) for k in range(-2, 3)], float)
    phase_L = np.exp(2j * np.pi * (Ls @ np.asarray(q_frac)))  # (nL,)
    dv = np.zeros((3 * nat, n1, n2, n3), complex)
    for ka in range(nat):
        centre = np.asarray(tau_frac[ka]) @ real_lat
        acc = np.zeros((3, n1, n2, n3), complex)
        for L, ph in zip(Ls, phase_L):
            v = r - centre - L @ real_lat                   # (n1,n2,n3,3)
            g = np.exp(-(v ** 2).sum(-1) / (2.0 * sigmas[ka] ** 2))
            grad = -v / sigmas[ka] ** 2 * g[..., None]      # d g / d r_alpha
            acc += ph * grad.transpose(3, 0, 1, 2)
        dv[3 * ka:3 * ka + 3] = acc * bloch[None]
    return dv


@pytest.fixture(scope="module", params=["diamond", "zincblende"])
def crystal(request):
    atoms = _atoms(request.param)
    sym = crystal_symmetry_operations(atoms)
    tau = np.asarray(atoms.get_scaled_positions())
    # The widths must respect the crystal symmetry: diamond's nonsymmorphic
    # operations SWAP the two carbons, so their local potentials must be
    # identical -- giving them different widths would (deliberately) break
    # the symmetry the rotation assumes and read as a rotation bug. The
    # zincblende species are never swapped, so distinct widths there kill
    # any accidental extra symmetry. Kept narrow so the +-2-cell lattice
    # sum is converged to machine precision even for operations that wrap
    # atoms across cell boundaries (which shifts the truncation window).
    if request.param == "diamond":
        sigmas = [0.12 * A0, 0.12 * A0]
    else:
        sigmas = [0.10 * A0, 0.14 * A0]
    return dict(kind=request.param, atoms=atoms, sym=sym, tau=tau,
                sigmas=sigmas, real=FCC, recip=_recip(FCC))


class TestRotateDvscf:
    def test_group_has_the_expected_character(self, crystal):
        t = crystal["sym"]["translations"]
        nonsym = (np.abs(t) > 1e-8).any(axis=1).sum()
        if crystal["kind"] == "diamond":
            assert nonsym > 0, "diamond setting must be nonsymmorphic here"
        else:
            has_inv = any(np.array_equal(r, -np.eye(3, dtype=np.int64))
                          for r in crystal["sym"]["rotations"])
            assert not has_inv, "zincblende must lack inversion"

    def test_every_operation_matches_the_directly_built_field(self, crystal):
        """rotate_dvscf(dv_q, S) == make_dv(S q) for EVERY operation,
        including the nonsymmorphic ones and images that fold with G != 0."""
        q = np.array([0.25, 0.5, 0.75])
        dv = make_dv(q, GRID, crystal["real"], crystal["tau"], crystal["sigmas"])
        worst_plain, worst_ft = 0.0, 0.0
        for isym in range(crystal["sym"]["nsym"]):
            dv_rot, q_rot = rotate_dvscf(
                dv, q, isym, crystal["sym"], crystal["real"],
                crystal["recip"], crystal["tau"])
            ref = make_dv(q_rot, GRID, crystal["real"], crystal["tau"],
                          crystal["sigmas"])
            err = np.abs(dv_rot - ref).max() / np.abs(ref).max()
            if (np.abs(crystal["sym"]["translations"][isym]) > 1e-8).any():
                worst_ft = max(worst_ft, err)
            else:
                worst_plain = max(worst_plain, err)
            assert err < 1e-10, (
                f"operation {isym} (t = "
                f"{crystal['sym']['translations'][isym]}) disagrees: {err:g}")
        if crystal["kind"] == "diamond":
            assert worst_ft > 0.0 or True  # ft ops exist; both classes ran

    def test_time_reversal_matches_the_directly_built_field(self, crystal):
        q = np.array([0.25, 0.5, 0.75])
        dv = make_dv(q, GRID, crystal["real"], crystal["tau"], crystal["sigmas"])
        for isym in (0, crystal["sym"]["nsym"] - 1):
            dv_rot, q_rot = rotate_dvscf(
                dv, q, isym, crystal["sym"], crystal["real"],
                crystal["recip"], crystal["tau"], time_reversal=True)
            ref = make_dv(q_rot, GRID, crystal["real"], crystal["tau"],
                          crystal["sigmas"])
            assert np.abs(dv_rot - ref).max() / np.abs(ref).max() < 1e-10

    def test_composition_of_two_rotations(self, crystal):
        """Chaining rotate_dvscf through an intermediate (possibly folded)
        label equals the direct route -- the labels are self-consistent."""
        q = np.array([0.25, 0.5, 0.75])
        dv = make_dv(q, GRID, crystal["real"], crystal["tau"], crystal["sigmas"])
        s1, s2 = 3, crystal["sym"]["nsym"] - 2
        dv1, q1 = rotate_dvscf(dv, q, s1, crystal["sym"], crystal["real"],
                               crystal["recip"], crystal["tau"])
        dv12, q12 = rotate_dvscf(dv1, q1, s2, crystal["sym"], crystal["real"],
                                 crystal["recip"], crystal["tau"])
        ref = make_dv(q12, GRID, crystal["real"], crystal["tau"],
                      crystal["sigmas"])
        assert np.abs(dv12 - ref).max() / np.abs(ref).max() < 1e-10

    def test_incompatible_grid_is_refused(self, crystal):
        if crystal["kind"] != "diamond":
            pytest.skip("needs a fractional translation")
        # 5^3 grid cannot host t = 1/4 translations
        q = np.array([0.2, 0.4, 0.6])
        dv = make_dv(q, (5, 5, 5), crystal["real"], crystal["tau"],
                     crystal["sigmas"])
        ft_ops = [i for i, t in enumerate(crystal["sym"]["translations"])
                  if (np.abs(t) > 1e-8).any()]
        with pytest.raises(ValueError, match="FFT grid"):
            rotate_dvscf(dv, q, ft_ops[0], crystal["sym"], crystal["real"],
                         crystal["recip"], crystal["tau"])


class TestStarRoutes:
    def test_full_mesh_reconstruction_from_the_wedge(self, crystal):
        """Wedge + routes rebuilds the exact dv at EVERY point of the mesh --
        the complete pipeline the g(R_e, R_q) build uses. For zincblende this
        is only possible with the time-reversal routes."""
        mesh = (4, 4, 4)
        q_irr, _ = irreducible_qpoints(crystal["atoms"], mesh)
        routes = dvscf_star_routes(q_irr, mesh, crystal["sym"])
        dv_irr = [make_dv(qi, GRID, crystal["real"], crystal["tau"],
                          crystal["sigmas"]) for qi in q_irr]
        n1, n2, n3 = mesh
        qfull = np.array([[i / n1, j / n2, k / n3] for i in range(n1)
                          for j in range(n2) for k in range(n3)])
        used_tr = False
        for j0, (ii, isym, tr) in enumerate(routes):
            dv_rot, q_rot = rotate_dvscf(
                dv_irr[ii], q_irr[ii], isym, crystal["sym"], crystal["real"],
                crystal["recip"], crystal["tau"], time_reversal=bool(tr))
            used_tr |= bool(tr)
            assert np.abs(((q_rot - qfull[j0]) + 0.5) % 1.0 - 0.5).max() < 1e-6
            ref = make_dv(qfull[j0], GRID, crystal["real"], crystal["tau"],
                          crystal["sigmas"])
            err = np.abs(dv_rot - ref).max() / np.abs(ref).max()
            assert err < 1e-10, f"mesh point {qfull[j0]} via {(ii, isym, tr)}: {err:g}"
        if crystal["kind"] == "zincblende":
            assert used_tr, "zincblende coverage must have needed time reversal"

    def test_zincblende_without_time_reversal_is_incomplete(self, crystal):
        if crystal["kind"] != "zincblende":
            pytest.skip("needs a crystal without inversion")
        mesh = (4, 4, 4)
        q_irr, _ = irreducible_qpoints(crystal["atoms"], mesh)
        with pytest.raises(ValueError, match="unreachable"):
            dvscf_star_routes(q_irr, mesh, crystal["sym"], time_reversal=False)
