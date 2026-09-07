"""
Structure and k-space helpers bridging ASE and the atomic-unit core.

ASE works in Angstrom; the core works in Bohr.  These helpers convert at the
boundary (via ``waw.units``, itself ase.units-backed) so the arrays handed
to the core are already atomic:

  * ``real_lattice(atoms)``   — real-space lattice rows in Bohr
  * ``recip_lattice(atoms)``  — reciprocal lattice rows in Bohr^-1 (2π convention)
  * ``monkhorst_pack(mesh)``  — Γ-centred MP mesh in crystal coordinates
  * ``band_path(atoms, …)``   — an ASE BandPath whose ``.kpts`` (crystal coords)
                                feed ``interpolate_bands`` directly
"""

from __future__ import annotations

import numpy as np

from ...units import ANG_TO_BOHR


def real_lattice(atoms) -> np.ndarray:
    """
    Real-space lattice matrix (rows = a1, a2, a3) in Bohr from an ase.Atoms.

    ``atoms.cell`` is in Angstrom; this converts to the core's Bohr convention.
    """
    return np.asarray(atoms.cell[:], dtype=np.float64) * ANG_TO_BOHR


def recip_lattice(atoms) -> np.ndarray:
    """
    Reciprocal lattice matrix (rows = b1, b2, b3) in Bohr^-1 from an ase.Atoms.

    Uses the 2π convention (b_i · a_j = 2π δ_ij), matching the core and the
    Wannier90 loader's ``parse_recip_lattice``.
    """
    a = real_lattice(atoms)
    return 2.0 * np.pi * np.linalg.inv(a).T


def monkhorst_pack(mesh: tuple[int, int, int]) -> np.ndarray:
    """
    Γ-centred Monkhorst-Pack mesh in crystal coordinates, shape (prod(mesh), 3).

    k_i = m_i / N_i for m_i = 0 … N_i-1 (the [0, 1) Γ-centred convention used by
    Wannier90 / QE and by the core's H(R) phases), enumerated with the first
    index slowest — the same order the .win ``begin kpoints`` block uses.

    Note this differs from ``ase.dft.kpoints.monkhorst_pack``, which centres the
    mesh on the origin with half-grid shifts.

    Vectorised, not a triple Python loop: this is called with N^3 up to tens of
    millions by the el-ph convergence sweeps, where the comprehension it used to
    be cost 23-29x more (2.65 s vs 0.11 s at 150^3) and allocated one Python
    list per k-point. Output is bit-identical, ordering included.
    """
    n1, n2, n3 = (int(n) for n in mesh)
    i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    return np.stack([i.ravel() / n1, j.ravel() / n2, k.ravel() / n3],
                    axis=-1).astype(np.float64)


def band_path(atoms, path: str | None = None, npoints: int = 100):
    """
    Build an ASE BandPath for band-structure interpolation.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure whose cell defines the Brillouin zone.
    path : str, optional
        High-symmetry path string, e.g. "GXWLGK".  If None, ASE picks the
        default path for the cell's Bravais lattice.
    npoints : int
        Total number of k-points along the path.

    Returns
    -------
    ase.dft.kpoints.BandPath
        Its ``.kpts`` (crystal coordinates) can be passed straight to
        ``waw.interpolate_bands``; ``.get_linear_kpoint_axis()`` gives the
        x-axis and tick positions/labels for plotting.
    """
    return atoms.cell.bandpath(path=path, npoints=npoints)


def band_path_segments(atoms, path: str | None = None) -> list[str]:
    """
    ASE's standard high-symmetry path for ``atoms``, as a list of
    ``waw.analysis.kpath.parse_kpoint_path``-style segment strings
    (``"LABEL1 kx ky kz  LABEL2 kx ky kz"``, crystal coordinates).

    For analysis functions that take an explicit ``kpoint_path: list[str]``
    (e.g. ``waw.analysis.spin_texture.spin_colored_bands``) rather than a
    ready-made k-point array, so they too use ASE's standard path (the
    Setyawan-Curtarolo convention for the cell's Bravais lattice) instead of
    a hand-copied one. Discontinuous sections of ``BandPath.path`` (comma
    separated, e.g. ``"GXWKGLUWLK,UX"``) become independent segments.
    """
    bp = atoms.cell.bandpath(path=path)
    segments = []
    for section in bp.path.split(","):
        for label1, label2 in zip(section, section[1:]):
            k1 = bp.special_points[label1]
            k2 = bp.special_points[label2]
            segments.append(
                f"{label1} {k1[0]:.8f} {k1[1]:.8f} {k1[2]:.8f}  "
                f"{label2} {k2[0]:.8f} {k2[1]:.8f} {k2[2]:.8f}"
            )
    return segments


def irreducible_qpoints(
    atoms, mesh: tuple[int, int, int], *, time_reversal: bool = True,
    symprec: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Irreducible wedge of a Gamma-centred q-mesh, with multiplicity weights.

    Returns ``(q_irr, weights)`` -- fractional coordinates in ``[0, 1)`` and
    weights summing to 1, such that for any function F invariant under the
    crystal point group,

        (1/Nq) sum_(all q) F(q)  ==  sum_(q in wedge) weights[i] * F(q_irr[i])

    Why this is exact for alpha2F, and what it requires. Write the
    Eliashberg sum as ``(1/Nq) sum_q F(q)`` with

        F(q) = (1/Nk) sum_(k,mn,nu) |g^nu_mn(k,q)|^2 delta(eps_km - eF)
               delta(eps_(k+q)n - eF) delta(omega - omega_(q,nu)).

    Under a point-group operation S the substitution k -> Sk maps the FULL
    k-mesh onto itself, while eps, omega and |g|^2 are all invariant, so
    F(Sq) = F(q) and only the star representative need be evaluated. The
    requirement is therefore that the k-sum runs over the COMPLETE mesh --
    reduce q, never k. (`analysis.elph.alpha2f` takes these weights as
    ``q_weights`` and does exactly that.)

    For fcc this is a factor up to 48, which is the difference between a
    feasible and an infeasible fine-mesh el-ph calculation; it is the same
    saving EPW gets from ``mp_mesh_k``.

    One caveat worth testing rather than assuming: the identity holds for the
    exact band structure and vertex, but a Wannier gauge that does not
    respect the crystal symmetry breaks it slightly. Compare a wedge result
    against a full-BZ one on a mesh you can afford before trusting it on one
    you cannot -- the test suite does this.

    Parameters
    ----------
    atoms : ase.Atoms -- cell, scaled positions and species define the group.
    mesh : (N1, N2, N3), Gamma-centred (the convention `monkhorst_pack` uses).
    time_reversal : fold q and -q together (correct for a non-magnetic,
        non-SOC system; switch off otherwise).
    symprec : spglib tolerance.

    Returns
    -------
    q_irr : (n_irr, 3) float64, fractional in [0, 1)
    weights : (n_irr,) float64, summing to 1
    """
    import spglib

    cell = (
        np.asarray(atoms.get_cell()),
        np.asarray(atoms.get_scaled_positions()),
        np.asarray(atoms.get_atomic_numbers()),
    )
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        tuple(int(m) for m in mesh), cell, is_shift=[0, 0, 0],
        is_time_reversal=bool(time_reversal), symprec=symprec,
    )
    mapping = np.asarray(mapping)
    reps, counts = np.unique(mapping, return_counts=True)
    q_irr = (np.asarray(grid, dtype=np.float64)[reps]
             / np.asarray(mesh, dtype=np.float64)) % 1.0
    weights = counts.astype(np.float64) / mapping.size
    return q_irr, weights


def irreducible_kmesh(atoms, mesh: tuple[int, int, int], *,
                      time_reversal: bool = True, symprec: float = 1e-5):
    """
    Irreducible wedge of a Gamma-centred mesh PLUS the exact map back to the
    full BZ, as a `core.symmetry.IrreducibleMesh`.

    `irreducible_qpoints` returns only the wedge and its weights, which is all a
    symmetric SUM needs. Use this instead whenever a per-k quantity computed on
    the wedge has to be placed back on the full mesh -- quasiparticle
    corrections, self-energies, linewidths, anything later Fourier-transformed
    to real space. It carries, for every full-mesh point, which representative
    it came from and which operation took it there, so the expansion is an exact
    integer lookup rather than a fit or an eigenvalue match.

    The ordering of ``kpts_full`` matches `monkhorst_pack(mesh)`, so the result
    can be used directly alongside anything built on that mesh.

    See `core.symmetry.IrreducibleMesh.expand_scalars` (invariants) and
    `expand_vectors` (velocities and other directed quantities), and
    `index_of` / `irr_index_of` for lining an external code's k-list up with
    this one.
    """
    from ...core.symmetry import build_irreducible_mesh, reciprocal_rotations

    ops = crystal_symmetry_operations(atoms, symprec=symprec)
    return build_irreducible_mesh(
        mesh, reciprocal_rotations(ops["rotations"]),
        time_reversal=time_reversal, kpts_full=monkhorst_pack(mesh))


def crystal_symmetry_operations(atoms, *, symprec: float = 1e-5) -> dict:
    """
    Space-group operations of a crystal, in the exact form the dvscf star
    rotation needs (`interfaces.quantum_espresso.dvscf_io.rotate_dvscf_star`).

    Returns a dict with, for ``nsym`` operations:

    ``rotations``    (nsym, 3, 3) int   -- R, acting on FRACTIONAL real-space
                     coordinates as ``x' = R x + t`` (spglib's convention, a
                     plain matrix-vector product; note QE stores the TRANSPOSE
                     of this in its own ``s`` array -- ``symm_base.f90`` builds
                     the rotated position as ``rau(:,na) = s(1,:,irot)*xau(1,na)
                     + s(2,:,irot)*xau(2,na) + s(3,:,irot)*xau(3,na)``, i.e.
                     contracting the FIRST index. Mixing the two up is silent
                     and gives a wrong answer only on non-symmorphic or
                     low-symmetry cells, so it is pinned by the round-trip
                     assertions below.)
    ``translations`` (nsym, 3) float    -- t, fractional. QE's ``ft`` is ``-t``.
    ``irt``          (nsym, nat) int    -- atom that each operation maps an atom
                     ONTO: ``tau[irt[i, na]] == R_i tau[na] + t_i`` (mod 1).
    ``invs``         (nsym,) int        -- index of each operation's inverse.

    Every one of those relations is asserted here rather than assumed, because
    all three are convention traps and none of them fails loudly downstream.
    """
    import spglib

    cell = (
        np.asarray(atoms.get_cell()),
        np.asarray(atoms.get_scaled_positions()),
        np.asarray(atoms.get_atomic_numbers()),
    )
    ds = spglib.get_symmetry(cell, symprec=symprec)
    rot = np.asarray(ds["rotations"], dtype=np.int64)
    trans = np.asarray(ds["translations"], dtype=np.float64)
    nsym = len(rot)
    tau = np.asarray(atoms.get_scaled_positions(), dtype=np.float64)
    nat = len(tau)

    # irt: which atom each one maps onto
    irt = np.full((nsym, nat), -1, dtype=np.int64)
    for i in range(nsym):
        moved = (tau @ rot[i].T + trans[i]) % 1.0
        for na in range(nat):
            d = (tau - moved[na] + 0.5) % 1.0 - 0.5
            hit = np.flatnonzero(np.abs(d).max(axis=1) < 10.0 * symprec)
            if hit.size != 1:
                raise ValueError(
                    f"crystal_symmetry_operations: operation {i} maps atom {na} "
                    f"onto {hit.size} candidates; symprec={symprec:g} is too "
                    "loose or the cell is inconsistent."
                )
            irt[i, na] = hit[0]
        if len(set(irt[i].tolist())) != nat:
            raise ValueError(f"operation {i} does not permute the atoms")

    # invs: R_j R_i = I and the translations compose back to a lattice vector
    invs = np.full(nsym, -1, dtype=np.int64)
    for i in range(nsym):
        for j in range(nsym):
            if not np.array_equal(rot[j] @ rot[i], np.eye(3, dtype=np.int64)):
                continue
            resid = rot[j] @ trans[i] + trans[j]
            if np.abs(resid - np.rint(resid)).max() < 1e-5:
                invs[i] = j
                break
        if invs[i] < 0:
            raise ValueError(f"no inverse found for operation {i}")

    return {"rotations": rot, "translations": trans, "irt": irt, "invs": invs,
            "nsym": nsym}
