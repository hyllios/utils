"""
k-point symmetry: irreducible meshes, and exact expansion back to the full BZ.

Everything here is INTEGER arithmetic on the mesh. A Gamma-centred N1xN2xN3
mesh point is the triple g with k = g / N, and a reciprocal-space point-group
operation R (integer, acting on fractional k) maps it to (R g) mod N exactly.
No tolerances, no coordinate conventions, no eigenvalue matching -- a mesh point
either is or is not the image of another, and the answer is exact.

WHY THIS MODULE EXISTS. Symmetry is what makes fine-mesh electron-phonon work
affordable: for fcc the irreducible wedge is up to 48x smaller. Repeatedly
hand-rolling the wedge->full-BZ bookkeeping in scripts is how one ends up
either computing the whole zone unnecessarily or matching eigenvalues and hoping
the assignment is unique. Both are avoidable with the mapping spglib already
returns.

WHAT MAY AND MAY NOT BE EXPANDED. `expand_scalars` is for quantities INVARIANT
under the point group -- band energies, quasiparticle corrections, |g|^2,
occupations, linewidths. Anything carrying a direction must be rotated:
`expand_vectors` does that for velocities, dipoles, forces. Expanding a
velocity as if it were a scalar is silently wrong (it gives every star member
the representative's direction), so the two entry points are kept separate
rather than switched on an argument.

CONVENTIONS. If a space-group operation acts on FRACTIONAL REAL-SPACE
coordinates as x' = R x + t (spglib's convention, and what
`interfaces.ase.structure.crystal_symmetry_operations` returns), then
fractional RECIPROCAL coordinates transform with

    R_k = (R^-1)^T                                                      (*)

`reciprocal_rotations` derives (*) and asserts it is integral; the mesh closure
is then asserted in `build_irreducible_mesh`, and the resulting multiplicities
are cross-checked against spglib's own reduction by the test suite. All
quantities here are dimensionless (fractional coordinates), so the atomic-unit
rule does not bite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def reciprocal_rotations(rotations_real: np.ndarray) -> np.ndarray:
    """
    Point-group operations acting on FRACTIONAL RECIPROCAL coordinates.

    From real-space fractional rotations R (x' = R x + t) this returns
    R_k = (R^-1)^T, which is integral whenever R is (det R = +-1).

    Parameters
    ----------
    rotations_real : (nsym, 3, 3) integer array, spglib's ``rotations``.

    Returns
    -------
    (nsym, 3, 3) int64.
    """
    R = np.asarray(rotations_real)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(
            f"reciprocal_rotations: expected (nsym, 3, 3); got {R.shape}")
    Rf = R.astype(np.float64)
    det = np.linalg.det(Rf)
    if not np.allclose(np.abs(det), 1.0, atol=1e-8):
        raise ValueError(
            "reciprocal_rotations: |det R| != 1 for some operation "
            f"(min {np.abs(det).min():.6f}, max {np.abs(det).max():.6f}); "
            "these are not fractional lattice rotations.")
    Rk = np.linalg.inv(Rf).transpose(0, 2, 1)
    Rk_int = np.rint(Rk).astype(np.int64)
    err = np.abs(Rk - Rk_int).max()
    if err > 1e-8:
        raise ValueError(
            f"reciprocal_rotations: (R^-1)^T is not integral (max deviation "
            f"{err:.3e}). The input is probably in a Cartesian or transposed "
            "convention rather than spglib's fractional one.")
    return Rk_int


@dataclass(frozen=True)
class IrreducibleMesh:
    """
    A Gamma-centred mesh together with its irreducible wedge and the exact map
    between them.

    Attributes
    ----------
    mesh          : (N1, N2, N3)
    kpts_full     : (nk, 3) fractional in [0, 1), in `monkhorst_pack` order.
    kpts_irr      : (n_irr, 3) fractional, the chosen representatives.
    weights       : (n_irr,) star multiplicities / nk, summing to 1.
    to_irr        : (nk,) index into `kpts_irr` for every full-mesh point.
    rotation      : (nk,) index into `rotations_k`: the operation carrying the
                    representative TO this point, i.e.
                    kpts_full[i] == rotations_k[rotation[i]] @ kpts_irr[to_irr[i]]
                    (mod 1), with an extra minus sign where `time_reversal[i]`.
    time_reversal : (nk,) bool, whether that mapping needs k -> -k.
    rotations_k   : (nsym, 3, 3) int, acting on fractional k.
    """

    mesh: tuple
    kpts_full: np.ndarray
    kpts_irr: np.ndarray
    weights: np.ndarray
    to_irr: np.ndarray
    rotation: np.ndarray
    time_reversal: np.ndarray
    rotations_k: np.ndarray

    @property
    def n_irr(self) -> int:
        return len(self.kpts_irr)

    @property
    def n_full(self) -> int:
        return len(self.kpts_full)

    def expand_scalars(self, values_irr: np.ndarray) -> np.ndarray:
        """
        Expand a point-group INVARIANT quantity from the wedge to the full mesh.

        ``values_irr`` has leading axis ``n_irr`` and any trailing shape (bands,
        modes, ...); the result has leading axis ``n_full``. Valid for energies,
        QP corrections, |g|^2, occupations, linewidths -- NOT for anything with
        a direction (use `expand_vectors`).
        """
        v = np.asarray(values_irr)
        if v.shape[0] != self.n_irr:
            raise ValueError(
                f"expand_scalars: leading axis {v.shape[0]} != n_irr "
                f"{self.n_irr}")
        return v[self.to_irr]

    def expand_vectors(self, vectors_irr: np.ndarray, *,
                       cartesian_lattice: np.ndarray | None = None
                       ) -> np.ndarray:
        """
        Expand a quantity carrying a DIRECTION (velocity, dipole, force).

        ``vectors_irr`` is (n_irr, ..., 3). The trailing axis is rotated by the
        operation that maps each representative onto its star member, and
        negated where time reversal was used.

        By default the trailing axis is taken to be in the same FRACTIONAL
        reciprocal basis as the k-points, so `rotations_k` applies directly.
        Pass ``cartesian_lattice`` (rows = reciprocal lattice vectors, i.e.
        ``recip_lattice(atoms)``) if the vectors are CARTESIAN, and the
        rotation is conjugated into that basis before use.
        """
        v = np.asarray(vectors_irr)
        if v.shape[0] != self.n_irr or v.shape[-1] != 3:
            raise ValueError(
                f"expand_vectors: expected (n_irr={self.n_irr}, ..., 3); "
                f"got {v.shape}")
        Rk = self.rotations_k.astype(np.float64)
        if cartesian_lattice is not None:
            B = np.asarray(cartesian_lattice, dtype=np.float64)
            Binv = np.linalg.inv(B)
            # x_cart = B^T x_frac  =>  S_cart = B^T R_k B^-T
            Rk = np.einsum('ij,sjk,kl->sil', B.T, Rk, Binv.T)
        out = v[self.to_irr]
        S = Rk[self.rotation]                                   # (nk, 3, 3)
        out = np.einsum('kij,k...j->k...i', S, out)
        sign = np.where(self.time_reversal, -1.0, 1.0)
        return out * sign.reshape((-1,) + (1,) * (out.ndim - 1))

    def index_of(self, kpts: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
        """
        Locate arbitrary k-points on this mesh, returning full-mesh indices.

        Used to line an EXTERNAL code's k-list (yambo's IBZ, EPW's, ...) up with
        ours without depending on either code's ordering. Points must lie on the
        mesh to within `tol`; anything else raises, because a silently
        mismatched k-list is exactly the failure this module exists to prevent.
        """
        k = np.asarray(kpts, dtype=np.float64).reshape(-1, 3)
        N = np.asarray(self.mesh, dtype=np.float64)
        g = k * N
        gi = np.rint(g).astype(np.int64)
        off = np.abs(g - gi).max()
        if off > tol:
            raise ValueError(
                f"index_of: k-points are not on the {self.mesh} mesh "
                f"(max fractional offset {off:.3e} > tol {tol:g}).")
        gi %= np.asarray(self.mesh, dtype=np.int64)
        lin = _linear_index(gi, self.mesh)
        lut = _mesh_lookup(self.mesh, self.kpts_full)
        return lut[lin]

    def irr_index_of(self, kpts: np.ndarray, *, tol: float = 1e-6
                     ) -> np.ndarray:
        """`index_of` composed with `to_irr`: which wedge point each k belongs
        to. This is the call that maps an external irreducible list onto ours."""
        return self.to_irr[self.index_of(kpts, tol=tol)]


def _linear_index(g: np.ndarray, mesh) -> np.ndarray:
    N1, N2, N3 = (int(m) for m in mesh)
    return (g[:, 0] * N2 + g[:, 1]) * N3 + g[:, 2]


def _mesh_lookup(mesh, kpts_full: np.ndarray) -> np.ndarray:
    """Map linear mesh index -> position in `kpts_full`."""
    N = np.asarray(mesh, dtype=np.int64)
    g = np.rint(np.asarray(kpts_full, dtype=np.float64) * N).astype(np.int64) % N
    lut = np.full(int(np.prod(N)), -1, dtype=np.int64)
    lut[_linear_index(g, mesh)] = np.arange(len(kpts_full))
    if (lut < 0).any():
        raise ValueError("_mesh_lookup: kpts_full does not cover the mesh.")
    return lut


def build_irreducible_mesh(mesh, rotations_k: np.ndarray, *,
                           time_reversal: bool = True,
                           kpts_full: np.ndarray | None = None
                           ) -> IrreducibleMesh:
    """
    Reduce a Gamma-centred mesh by the given reciprocal-space operations.

    Exact integer construction: the orbit of mesh point g under the group is
    ``{(R g) mod N}`` (plus ``{(-R g) mod N}`` with time reversal), and the
    representative is the orbit member with the smallest linear index.

    Parameters
    ----------
    mesh          : (N1, N2, N3)
    rotations_k   : (nsym, 3, 3) int, acting on fractional k -- from
                    `reciprocal_rotations`.
    time_reversal : fold k and -k together (correct without magnetism/SOC).
    kpts_full     : optional explicit full-mesh ordering to adopt (so the result
                    lines up with `monkhorst_pack`); built if omitted.

    Returns
    -------
    IrreducibleMesh
    """
    N = np.asarray([int(m) for m in mesh], dtype=np.int64)
    if (N <= 0).any():
        raise ValueError(f"build_irreducible_mesh: bad mesh {tuple(mesh)}")
    Rk = np.asarray(rotations_k, dtype=np.int64)
    if Rk.ndim != 3 or Rk.shape[1:] != (3, 3):
        raise ValueError(
            f"build_irreducible_mesh: rotations_k must be (nsym, 3, 3); "
            f"got {Rk.shape}")

    if kpts_full is None:
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in N], indexing='ij'),
                       axis=-1).reshape(-1, 3)
        kpts_full = idx / N
    else:
        kpts_full = np.asarray(kpts_full, dtype=np.float64)
        idx = np.rint(kpts_full * N).astype(np.int64) % N
    nk = len(idx)

    # images of every mesh point under every operation, exactly
    img = np.einsum('sij,kj->ski', Rk, idx) % N               # (nsym, nk, 3)
    if not np.array_equal(np.sort(_linear_index(img.reshape(-1, 3), N)
                                  .reshape(len(Rk), nk), axis=1),
                          np.sort(np.broadcast_to(
                              _linear_index(idx, N), (len(Rk), nk)), axis=1)):
        raise ValueError(
            "build_irreducible_mesh: the operations do not map this mesh onto "
            "itself. The mesh is incommensurate with the point group (try a "
            "mesh whose divisions respect the lattice symmetry).")

    lin_full = _linear_index(idx, N)
    lut = np.full(int(np.prod(N)), -1, dtype=np.int64)
    lut[lin_full] = np.arange(nk)

    orbits = [lut[_linear_index(img[s], N)] for s in range(len(Rk))]
    if time_reversal:
        img_t = (-img) % N
        orbits += [lut[_linear_index(img_t[s], N)] for s in range(len(Rk))]
    orb = np.stack(orbits, axis=0)                            # (nops, nk)
    nsym = len(Rk)

    # representative = orbit member with the smallest position in kpts_full
    rep = orb.min(axis=0)
    # iterate to a fixed point (the orbit of a member is the same set, so one
    # pass over a full group is enough, but this is cheap insurance)
    for _ in range(8):
        new = rep[rep]
        if np.array_equal(new, rep):
            break
        rep = new

    reps, inverse, counts = np.unique(rep, return_inverse=True,
                                      return_counts=True)
    to_irr = inverse.astype(np.int64)
    kpts_irr = kpts_full[reps]
    weights = counts.astype(np.float64) / nk

    # for each point, an operation carrying its representative onto it
    rotation = np.full(nk, -1, dtype=np.int64)
    tr = np.zeros(nk, dtype=bool)
    for op in range(orb.shape[0]):
        # op maps point j -> orb[op, j]; we want rep -> point
        tgt = orb[op, reps[to_irr]]
        hit = (rotation < 0) & (tgt == np.arange(nk))
        rotation[hit] = op % nsym
        tr[hit] = op >= nsym
    if (rotation < 0).any():
        raise ValueError(
            f"build_irreducible_mesh: no operation found for "
            f"{(rotation < 0).sum()} points -- internal inconsistency.")

    return IrreducibleMesh(
        mesh=tuple(int(m) for m in mesh), kpts_full=kpts_full,
        kpts_irr=kpts_irr, weights=weights, to_irr=to_irr,
        rotation=rotation, time_reversal=tr, rotations_k=Rk)
