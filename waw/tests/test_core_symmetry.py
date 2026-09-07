"""
k-point symmetry: exactness of the wedge -> full-BZ map.

The load-bearing test is `test_matches_spglib_reduction`: our integer
construction must reproduce spglib's own irreducible count and multiplicities on
several lattices. Everything else checks the properties that make the map
usable -- that the claimed operation really carries the representative to the
point, that invariant sums are unchanged, and that directed quantities rotate.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from waw.core.symmetry import (build_irreducible_mesh, reciprocal_rotations)
from waw.interfaces.ase.structure import (crystal_symmetry_operations,
                                          real_lattice,
                                          irreducible_kmesh,
                                          irreducible_qpoints,
                                          monkhorst_pack, recip_lattice)


def _systems():
    return [
        ("fcc Ca", bulk("Ca", "fcc", a=5.4538), (6, 6, 6)),
        ("fcc Ca 12", bulk("Ca", "fcc", a=5.4538), (12, 12, 12)),
        ("bcc Nb", bulk("Nb", "bcc", a=3.30), (8, 8, 8)),
        ("hcp Mg", bulk("Mg", "hcp", a=3.21, c=5.21), (6, 6, 4)),
        ("sc H", Atoms("H", scaled_positions=[[0, 0, 0]],
                       cell=np.eye(3) * 3.0, pbc=True), (4, 4, 4)),
    ]


@pytest.mark.parametrize("name,atoms,mesh", _systems(),
                         ids=[s[0] for s in _systems()])
def test_matches_spglib_reduction(name, atoms, mesh):
    """Our integer reduction must agree with spglib's, count and weights."""
    irr = irreducible_kmesh(atoms, mesh)
    q_ref, w_ref = irreducible_qpoints(atoms, mesh)
    assert irr.n_irr == len(q_ref), (name, irr.n_irr, len(q_ref))
    assert np.isclose(irr.weights.sum(), 1.0)
    # multiplicity multiset must match (representative choice may differ)
    assert np.allclose(np.sort(irr.weights), np.sort(w_ref)), name


@pytest.mark.parametrize("name,atoms,mesh", _systems(),
                         ids=[s[0] for s in _systems()])
def test_rotation_actually_maps_representative_to_point(name, atoms, mesh):
    """The stored operation must carry the representative onto the point,
    exactly, on the mesh."""
    irr = irreducible_kmesh(atoms, mesh)
    N = np.asarray(irr.mesh)
    rep = irr.kpts_irr[irr.to_irr]
    S = irr.rotations_k[irr.rotation]
    img = np.einsum("kij,kj->ki", S, rep)
    img = np.where(irr.time_reversal[:, None], -img, img)
    delta = (img - irr.kpts_full) * N
    assert np.allclose(delta, np.rint(delta), atol=1e-9), name
    assert np.allclose(np.rint(delta) % N, 0), name


@pytest.mark.parametrize("name,atoms,mesh", _systems(),
                         ids=[s[0] for s in _systems()])
def test_expand_scalars_preserves_symmetric_sum(name, atoms, mesh):
    """A weighted wedge sum and a full-mesh mean must agree for an invariant."""
    irr = irreducible_kmesh(atoms, mesh)
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(irr.n_irr, 3))
    full = irr.expand_scalars(vals)
    assert full.shape == (irr.n_full, 3)
    assert np.allclose(full.mean(axis=0), irr.weights @ vals)


def test_expand_scalars_is_constant_over_each_star():
    irr = irreducible_kmesh(bulk("Ca", "fcc", a=5.4538), (6, 6, 6))
    tag = irr.expand_scalars(np.arange(irr.n_irr, dtype=float))
    for i in range(irr.n_irr):
        assert np.all(tag[irr.to_irr == i] == i)


def test_expand_vectors_rotates_and_preserves_norm():
    """Rotating preserves the Cartesian norm.

    NOTE what is deliberately NOT asserted here: that the full-BZ average of an
    expanded vector field vanishes. That holds only for stars with a trivial
    little group -- at Gamma the star is a single point, so a random `v_irr`
    is not a legitimate symmetric field there. The end-to-end check against a
    genuine symmetric field is `test_expand_vectors_matches_direct_gradient`.
    """
    atoms = bulk("Ca", "fcc", a=5.4538)
    irr = irreducible_kmesh(atoms, (6, 6, 6))
    B = recip_lattice(atoms)
    rng = np.random.default_rng(1)
    v_irr = rng.normal(size=(irr.n_irr, 3))
    v_full = irr.expand_vectors(v_irr, cartesian_lattice=B)
    assert v_full.shape == (irr.n_full, 3)
    n_irr = np.linalg.norm(v_irr, axis=1)[irr.to_irr]
    assert np.allclose(np.linalg.norm(v_full, axis=1), n_irr, atol=1e-9)


@pytest.mark.parametrize("name,atoms,mesh", _systems(),
                         ids=[s[0] for s in _systems()])
def test_expand_vectors_matches_direct_gradient(name, atoms, mesh):
    """THE test of the rotation convention.

    Build a genuinely symmetric band E(k) = sum_R cos(k.R) over a full
    neighbour shell, whose Cartesian gradient v(k) = -sum_R R sin(k.R) is a
    real symmetric vector field. Computing v on the wedge and EXPANDING must
    reproduce v computed directly on the full mesh. Any error in
    R_k = (R^-1)^T, in the fractional->Cartesian conjugation, or in the
    time-reversal sign shows up here immediately.
    """
    irr = irreducible_kmesh(atoms, mesh)
    B = recip_lattice(atoms)                    # rows = b_i, Bohr^-1
    A = real_lattice(atoms)                     # rows = a_i, Bohr
    # COMPLETE neighbour shells, so the set is point-group invariant. NB the
    # naive {-1,0,1}^3 set of primitive-vector combinations is NOT invariant --
    # a rotation can send a1+a2 to a vector needing a coefficient of 2 -- and
    # using it makes E(k) non-invariant and the test meaningless.
    ijk = np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    cand = ijk @ A
    d = np.linalg.norm(cand, axis=1)
    dmin = d[d > 1e-8].min()
    R_cart = cand[(d > 1e-8) & (d < 1.8 * dmin)]   # all vectors within a cutoff

    def band_and_velocity(kfrac):
        k_cart = kfrac @ B                      # (nk, 3)
        ph = k_cart @ R_cart.T                  # (nk, nR)
        E = np.cos(ph).sum(axis=1)
        v = -np.einsum("kr,rd->kd", np.sin(ph), R_cart)
        return E, v

    E_full, v_full = band_and_velocity(irr.kpts_full)
    E_irr, v_irr = band_and_velocity(irr.kpts_irr)

    # the scalar must be invariant across each star
    assert np.allclose(irr.expand_scalars(E_irr), E_full, atol=1e-10), name
    # and the vector must rotate onto the directly computed field
    v_exp = irr.expand_vectors(v_irr, cartesian_lattice=B)
    assert np.allclose(v_exp, v_full, atol=1e-9), (
        name, np.abs(v_exp - v_full).max())


def test_expand_vectors_beats_scalar_expansion_for_velocities():
    """Expanding a velocity as a scalar is WRONG -- it hands every star member
    the representative's direction. Measured against the directly computed
    symmetric field, so the comparison is to ground truth rather than to a
    symmetry identity that only holds for stars with a trivial little group."""
    atoms = bulk("Ca", "fcc", a=5.4538)
    irr = irreducible_kmesh(atoms, (6, 6, 6))
    B, A = recip_lattice(atoms), real_lattice(atoms)
    ijk = np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    cand = ijk @ A
    d = np.linalg.norm(cand, axis=1)
    R_cart = cand[(d > 1e-8) & (d < 1.8 * d[d > 1e-8].min())]

    def vel(kfrac):
        ph = (kfrac @ B) @ R_cart.T
        return -np.einsum("kr,rd->kd", np.sin(ph), R_cart)

    v_full, v_irr = vel(irr.kpts_full), vel(irr.kpts_irr)
    right = irr.expand_vectors(v_irr, cartesian_lattice=B)
    wrong = irr.expand_scalars(v_irr)
    assert np.abs(right - v_full).max() < 1e-9
    assert np.abs(wrong - v_full).max() > 0.1 * np.abs(v_full).max()


def test_index_of_roundtrip_and_external_list():
    """`index_of` must invert the mesh ordering, and map a shuffled/wrapped
    external k-list (another code's convention) correctly."""
    atoms = bulk("Ca", "fcc", a=5.4538)
    irr = irreducible_kmesh(atoms, (12, 12, 12))
    assert np.array_equal(irr.index_of(irr.kpts_full), np.arange(irr.n_full))
    rng = np.random.default_rng(3)
    perm = rng.permutation(irr.n_full)
    shifted = irr.kpts_full[perm] - np.rint(irr.kpts_full[perm])   # to (-.5,.5]
    assert np.array_equal(irr.index_of(shifted), perm)
    # an external IBZ list lands on the right wedge points
    assert np.array_equal(np.sort(np.unique(irr.irr_index_of(irr.kpts_irr))),
                          np.arange(irr.n_irr))


def test_index_of_rejects_off_mesh_points():
    irr = irreducible_kmesh(bulk("Ca", "fcc", a=5.4538), (6, 6, 6))
    with pytest.raises(ValueError, match="not on the"):
        irr.index_of(np.array([[0.05, 0.0, 0.0]]))


def test_reciprocal_rotations_are_integral_and_group():
    ops = crystal_symmetry_operations(bulk("Mg", "hcp", a=3.21, c=5.21))
    Rk = reciprocal_rotations(ops["rotations"])
    assert Rk.dtype == np.int64
    assert np.allclose(np.abs(np.linalg.det(Rk.astype(float))), 1.0)
    # closure: products stay in the set
    keys = {Rk[i].tobytes() for i in range(len(Rk))}
    prod = np.einsum("aij,bjk->abik", Rk, Rk).reshape(-1, 3, 3)
    assert all(prod[i].tobytes() in keys for i in range(len(prod)))


def test_reciprocal_rotations_rejects_cartesian_input():
    """A Cartesian (non-integral) rotation must be refused, not silently
    rounded -- that is the convention trap this guards."""
    theta = 0.3
    c, s = np.cos(theta), np.sin(theta)
    bad = np.array([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]])
    with pytest.raises(ValueError, match="not integral"):
        reciprocal_rotations(bad)


def test_time_reversal_off_gives_more_points():
    atoms = Atoms("H", scaled_positions=[[0, 0, 0]],
                  cell=np.eye(3) * 3.0, pbc=True)
    a = irreducible_kmesh(atoms, (4, 4, 4), time_reversal=True)
    b = irreducible_kmesh(atoms, (4, 4, 4), time_reversal=False)
    assert b.n_irr >= a.n_irr
    for m in (a, b):
        assert np.isclose(m.weights.sum(), 1.0)


def test_incommensurate_mesh_raises():
    """A mesh that the point group does not map onto itself must fail loudly."""
    atoms = bulk("Mg", "hcp", a=3.21, c=5.21)
    ops = crystal_symmetry_operations(atoms)
    Rk = reciprocal_rotations(ops["rotations"])
    with pytest.raises(ValueError, match="do not map this mesh"):
        build_irreducible_mesh((3, 4, 5), Rk, kpts_full=monkhorst_pack((3, 4, 5)))
