"""
Reader for QE's raw DFPT self-consistent-potential variation (``ph.x``'s
``fildvscf`` output, ``_ph0/<prefix>.dvscf<irr>``) -- the input needed to
build electron-phonon coupling matrix elements natively, the same
quantity EPW itself starts from.

Deliberately bypasses ``ph.x``'s own ``electron_phonon='Wannier'``
(``elph_mat``) feature, which writes electron-phonon matrix elements
directly (``<prefix>_elph.mat.q_N``) but crashes with a heap-corruption
error in this QE version (confirmed reproducible serial and parallel,
`ep_matrix_element_wannier.f90`'s dvscf-reading loop) -- EPW itself does
not drive that legacy (Calandra-Mauri-era) codepath either; it reads
this same plain ``dvscf`` output and builds the Bloch-gauge matrix
element itself. A **plain** phonon run (``trans=.true.``, `fildvscf` set,
no `electron_phonon`/`dvscf_star` needed) is all that's required upstream.

File format (transcribed from QE source):

  - ``PHonon/PH/openfilq.f90``'s non-``elph_mat`` branch opens
    ``<prefix>.dvscf<irr>`` as a Fortran direct-access unformatted file,
    record length ``lrdrho = 2*dfftp%nnr*nspin_mag`` real(8) words.
  - ``davcio_drho.f90`` reads/writes ``ddrho(nr1x*nr2x*nr3x, nspin_mag)``
    per record, one record per perturbation index within the
    representation (Fortran column-major: the real-space grid point
    varies fastest, matching `interfaces.wannier90.io.read_unk`'s own
    ``u_nk`` grid-point ordering, since both are dense/no-double-grid
    QE FFT arrays -- no leading Fortran record markers, direct-access
    files are fixed-length raw records).
  - For q-point index 1 (QE's own 1-based ``ldisp`` numbering, matching
    the order printed under "Dynamical matrices for (...) uniform grid
    of q-points"), the file lives directly under ``outdir/_ph0/``. For
    q-point index > 1, ``ph.x`` uses a separate per-q scratch
    subdirectory, ``outdir/_ph0/<prefix>.q_<iq>/``.
  - ALL representations (irreps) at a q-point share a SINGLE file,
    ``<prefix>.dvscf1`` (confirmed empirically: a q-point with three
    separate 1-dimensional irreps still has only one ``dvscf1`` file, no
    ``dvscf2``/``dvscf3`` -- the "1" is a fixed literal, NOT the irrep
    index, despite looking like one at Gamma where there happens to be
    only a single irrep). Records are numbered by a RUNNING mode
    counter across all irreps in order (``imode0 + ipert``, matching
    `ep_matrix_element_wannier.f90`'s own reading convention) -- irrep 1's
    ``n_pert`` modes occupy the first records, irrep 2's occupy the next
    ``n_pert`` records, and so on. Each irrep's OWN records are in that
    irrep's symmetry-adapted PATTERN basis, not bare Cartesian atomic
    displacements -- ``outdir/_ph0/<prefix>.phsave/patterns.<iq>.xml``
    gives each irrep's ``DISPLACEMENT_PATTERN``. That matrix is UNITARY
    and, away from high-symmetry q, GENUINELY COMPLEX -- it is real only
    at Gamma and on symmetry lines, so an early "verified real" reading of
    it (taken from a Gamma-only check) is wrong. Cartesian-basis dV is
    recovered by applying the conjugate transpose of each irrep's slice.

Collinear ``nspin_mag == 2`` is supported via `read_dvscf`'s ``nspin``/``ispin``
(one vertex per spin channel); noncollinear/SOC is not -- the
project's own conventions extend record length by ``nspin_mag`` when
that's needed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BYTES_PER_COMPLEX128 = 16


def read_patterns(phsave_dir: str | Path, iq: int) -> list[dict]:
    """
    Parse ``<prefix>.phsave/patterns.<iq>.xml``.

    Returns a list (one entry per irrep, 1-based order) of dicts:
      n_pert  : int, number of perturbations (= modes) in this irrep
      pattern : (n_pert, n_pert_total) complex128, row p = the Cartesian-
                displacement coefficients of perturbation p. GENUINELY
                COMPLEX away from high-symmetry q -- do not discard the
                imaginary part (it reaches 0.23 in magnitude on a plain
                6x6x6 Al mesh, and dropping it breaks the rotation's
                unitarity by 5% and inflates lambda at those q by ~3x).

    ``n_pert_total`` (== 3*nat) is inferred from the pattern vector length.
    """
    path = Path(phsave_dir) / f"patterns.{iq}.xml"
    text = path.read_text()

    irreps = []
    for irr_block in re.findall(r"<REPRESENTION\.\d+>(.*?)</REPRESENTION\.\d+>", text, re.S):
        n_pert = int(re.search(r"<NUMBER_OF_PERTURBATIONS>(\d+)</NUMBER_OF_PERTURBATIONS>", irr_block).group(1))
        rows = []
        for pat_block in re.findall(r"<DISPLACEMENT_PATTERN>(.*?)</DISPLACEMENT_PATTERN>", irr_block, re.S):
            nums = [float(x) for x in pat_block.split()]
            re_part, im_part = nums[0::2], nums[1::2]
            rows.append(np.array(re_part) + 1j * np.array(im_part))
        irreps.append({"n_pert": n_pert, "pattern": np.stack(rows)})
    return irreps


def _dvscf_dir(outdir: str | Path, prefix: str, iq: int) -> Path:
    ph0 = Path(outdir) / "_ph0"
    return ph0 if iq == 1 else ph0 / f"{prefix}.q_{iq}"


def _read_dvscf_records(
    path: Path, first_record: int, n_records: int, grid: tuple[int, int, int],
    nspin: int = 1, ispin: int = 0,
) -> np.ndarray:
    """
    Read records ``[first_record, first_record + n_records)`` (0-based)
    from the shared per-q dvscf file -> (n_records, nr1x, nr2x, nr3x)
    complex128, Fortran ordering. The file may hold MORE records than any
    single irrep needs (other irreps' records live at other offsets in
    the SAME file) -- only a minimum-size check is made, not an exact
    match.

    SPIN. QE's record length is ``lrdrho = 2*nnr*nspin_mag`` REAL(8) words, i.e.
    ``nnr*nspin_mag`` complex numbers, with the spin blocks CONTIGUOUS inside one
    record. So a collinear magnet (``nspin=2``) doubles the stride, and reading
    it with the non-magnetic stride does not fail -- it silently returns the
    right bytes for mode 0 spin 0 and misaligned garbage for everything after.
    Hence ``nspin`` is required to be stated, and ``ispin`` selects the block.
    """
    if not 0 <= ispin < nspin:
        raise ValueError(f"_read_dvscf_records: ispin {ispin} outside nspin {nspin}")
    nr1x, nr2x, nr3x = grid
    npts = nr1x * nr2x * nr3x
    stride = npts * nspin                       # complex numbers per record
    raw = path.read_bytes()
    needed = (first_record + n_records) * stride * BYTES_PER_COMPLEX128
    if len(raw) < needed:
        raise ValueError(
            f"{path}: need at least {needed} bytes (record {first_record} + "
            f"{n_records}, {npts} points x nspin {nspin} each), got {len(raw)}. "
            f"A shortfall of exactly a factor {nspin} means nspin is wrong."
        )
    out = np.empty((n_records, npts), dtype=np.complex128)
    for r in range(n_records):
        start = (first_record + r) * stride + ispin * npts
        out[r] = np.frombuffer(
            raw, dtype=np.complex128, count=npts,
            offset=start * BYTES_PER_COMPLEX128)
    return np.stack([out[r].reshape((nr1x, nr2x, nr3x), order="F")
                     for r in range(n_records)])


def read_dvscf(
    outdir: str | Path, prefix: str, iq: int, grid: tuple[int, int, int], nat: int,
    nspin: int = 1, ispin: int = 0,
) -> np.ndarray:
    """
    Read all irreps at q-point ``iq`` and assemble the Cartesian-basis
    potential variation.

    Returns ``dv_cart``, shape ``(3*nat, nr1x, nr2x, nr3x)`` complex128:
    ``dv_cart[mu]`` is dV/du_mu(r) for a unit Cartesian displacement of
    atom ``mu // 3`` along axis ``mu % 3`` -- Hartree/Bohr, real-space FFT
    grid (QE stores this in its own internal Rydberg units, Ry/Bohr;
    converted to Hartree/Bohr here at the read boundary, matching
    `phonon_io.read_force_constants`'s own Ry -> Hartree convention).

    COLLINEAR MAGNETS. Pass ``nspin=2`` for a ``ph.x`` run on an ``nspin=2``
    ground state, and ``ispin`` (0 = up, 1 = down) to choose the channel. The
    two channels are genuinely different perturbations -- dV_scf^sigma =
    dV_bare + dV_H + dV_xc^sigma, and only the xc piece carries the spin index,
    but that is enough to give each channel its own electron-phonon vertex
    g^sigma = <psi^sigma| dV_scf^sigma |psi^sigma>. Build g(Re,Rq) once per
    channel, with that channel's own Wannier gauge, and feed the pair to
    `analysis.elph_boltzmann.spin_resolved_conductivity`.

    Getting ``nspin`` wrong is silent rather than fatal: the record stride is
    ``nnr*nspin_mag`` complex numbers, so mode 0 spin 0 still reads correctly
    and every later mode is misaligned. There is no way to detect nspin from the
    dvscf file alone -- it is not in the header -- so it must be stated. Cheap
    validation: run a NON-magnetic system at ``nspin=2`` with zero starting
    magnetization; both channels must then equal each other and the ``nspin=1``
    result to machine precision (`tests/test_dvscf_spin.py`).
    """
    ddir = _dvscf_dir(outdir, prefix, iq)
    phsave = Path(outdir) / "_ph0" / f"{prefix}.phsave"
    irreps = read_patterns(phsave, iq)
    path = ddir / f"{prefix}.dvscf1"

    n_modes = 3 * nat
    dv_cart = np.zeros((n_modes,) + tuple(grid), dtype=np.complex128)
    imode0 = 0
    for irr in irreps:
        n_pert = irr["n_pert"]
        dv_pattern = _read_dvscf_records(path, imode0, n_pert, grid,
                                         nspin=nspin, ispin=ispin)
        # pattern: (n_pert, n_modes), row p = Cartesian coefficients of
        # perturbation p, so dv_pattern = P @ dv_cart. P is UNITARY (not
        # merely real-orthogonal: away from high-symmetry q the patterns
        # are genuinely complex -- |Im P| reaches 0.23 on a 6^3 Al mesh),
        # so the inverse rotation is the conjugate transpose.
        pattern = irr["pattern"]
        dv_cart += np.einsum("pm,p...->m...", pattern.conj(), dv_pattern)
        imode0 += n_pert
    return dv_cart * 0.5   # Ry/Bohr -> Hartree/Bohr


def rotate_dvscf(
    dv_cart: np.ndarray,
    q_frac: np.ndarray,
    isym: int,
    sym: dict,
    real_lattice: np.ndarray,
    recip_lattice: np.ndarray,
    tau_frac: np.ndarray,
    time_reversal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a Cartesian-basis dvscf from ``q`` to the star member ``S q``
    (or, with ``time_reversal``, to ``-S q``).

    This is what makes the irreducible wedge usable for the electron-phonon
    vertex. The alpha2F q-sum reduces trivially (its summand is an invariant
    scalar -- `interfaces.ase.structure.irreducible_qpoints`), but BUILDING
    ``g(R_e, R_q)`` is a Fourier transform to a real-space object and needs
    dvscf at every q of the coarse mesh individually. Without this, DFPT must
    run on the full grid: 144 q instead of 21 for MgB2 (6.9x), 216 instead of
    16 for fcc Al (13.5x).

    A transcription of QE's ``PHonon/PH/dfile_star.f90`` (the machinery behind
    ``dvscf_star``). For target ``S q``, with ``Sinv = S^-1``:

      1. Cartesian -> crystal:   ``dv_j = a_j . dv_cart``
      2. strip the source phase: atom kappa times ``exp(+2 pi i q.tau_kappa)``
      3. rotate:  ``dv'[na, p](x) = sum_j R_Sinv[j, p] dv[irt[Sinv, na], j](x')``
         evaluated at ``x' = R_Sinv x - ft_Sinv``
      4. reapply at the new q:   atom na times ``exp(-2 pi i (Sq).tau_na)``
      5. crystal -> Cartesian:   ``dv_cart_j = sum_k dv_k b_k[j]``

    Three conventions are pinned from QE's source rather than assumed, each of
    which is silent when wrong:

      * QE's ``s`` is the TRANSPOSE of the ``x' = R x`` rotation --
        ``symm_base.f90`` builds ``rau(:,na) = s(1,:,irot)*xau(1,na) + ...``,
        contracting the FIRST index -- so ``dfile_star``'s ``s(ipol, j, Sinv)``
        is ``R_Sinv[j, ipol]`` in the convention used here;
      * ``rotate_grid_point`` with QE's scaled matrix reduces to
        ``x' = R x - ft``, not ``R x + ft``;
      * QE's ``ft`` is MINUS spglib's translation.

    The FFT grid must map onto itself under the operation. That is a genuine
    restriction -- a grid whose dimensions do not share the symmetry breaks it
    -- so it is checked rather than assumed.

    Parameters
    ----------
    dv_cart : (3*nat, n1, n2, n3) complex128, `read_dvscf`'s output at ``q``.
    q_frac : (3,) source q, fractional.
    isym : index into ``sym`` of the operation to apply.
    sym : `interfaces.ase.structure.crystal_symmetry_operations`'s output.
    real_lattice : (3, 3) rows = a_i, Bohr.
    recip_lattice : (3, 3) rows = b_i, 2*pi/Bohr.
    tau_frac : (nat, 3) fractional positions, in the order the dvscf mode index
        ``mu = na*3 + ipol`` uses.
    time_reversal : bool
        Additionally apply time reversal AFTER the rotation: ``dv(-q) =
        conj(dv(q))`` (QE's own ``dfile_minus_q``, ``dfile_star.f90`` writes
        ``CONJG(dfile_rot)`` at ``-sxq``). This is what completes a
        NON-CENTROSYMMETRIC crystal's coverage of the full q-mesh: ph.x
        reduces its q-grid with time reversal included, so for a crystal
        without inversion the point-group star of the ph.x wedge alone does
        not reach the -q half of the mesh. Valid for a real (no magnetic
        field, collinear-or-none spin) Hamiltonian.

    Returns
    -------
    dv_rot : (3*nat, n1, n2, n3) complex128 at ``S q`` (or ``-S q``)
    q_rot : (3,) image of ``q``, fractional, folded into [0, 1)
    """
    dv_cart = np.asarray(dv_cart)
    n_modes = dv_cart.shape[0]
    grid = tuple(dv_cart.shape[1:])
    nat = n_modes // 3
    tau = np.asarray(tau_frac, dtype=np.float64)
    A = np.asarray(real_lattice, dtype=np.float64)
    B = np.asarray(recip_lattice, dtype=np.float64) / (2.0 * np.pi)
    if not np.allclose(A @ B.T, np.eye(3), atol=1e-8):
        raise ValueError("rotate_dvscf: need real_lattice @ recip_lattice.T == 2*pi*I")
    if tau.shape != (nat, 3):
        raise ValueError(f"rotate_dvscf: tau_frac {tau.shape} vs {nat} atoms")

    R = np.asarray(sym["rotations"][isym], dtype=np.int64)
    inv = int(sym["invs"][isym])
    R_inv = np.asarray(sym["rotations"][inv], dtype=np.float64)
    ft_inv = -np.asarray(sym["translations"][inv], dtype=np.float64)   # QE sign
    irt_inv = np.asarray(sym["irt"][inv], dtype=np.int64)

    # q.x is invariant under x -> R x, so q -> (R^-1)^T q
    q = np.asarray(q_frac, dtype=np.float64)
    q_raw = np.linalg.inv(R.astype(np.float64)).T @ q

    ph_shape = (nat, 1) + (1,) * len(grid)

    # 1. Cartesian -> crystal
    dv = np.einsum("ja,ka...->kj...", A, dv_cart.reshape(nat, 3, *grid))

    # 2. strip the source-q phase
    dv = dv * np.exp(2j * np.pi * (tau @ q)).reshape(ph_shape)

    # 3a. rotated grid point  x' = R_inv x - ft_inv
    mesh = np.stack(np.meshgrid(*[np.arange(n) for n in grid], indexing="ij"),
                    axis=-1).astype(np.float64)
    nr = np.array(grid, dtype=np.float64)
    xp = (mesh / nr) @ R_inv.T - ft_inv          # fractional
    src_f = xp * nr
    if not np.allclose(src_f, np.rint(src_f), atol=1e-6):
        raise ValueError(
            f"rotate_dvscf: operation {isym} does not map the {grid} FFT grid "
            "onto itself -- grid dimensions incompatible with the space group."
        )
    src = np.rint(src_f).astype(np.int64) % np.array(grid, dtype=np.int64)
    dv_src = dv[:, :, src[..., 0], src[..., 1], src[..., 2]]

    # 3b. permute atoms and rotate the polarisation index
    dv_rot = np.einsum("jp,kj...->kp...", R_inv, dv_src[irt_inv])

    # 4. reapply the phase at the new q (before folding: the atom-position
    #    phase belongs to the unfolded image)
    dv_rot = dv_rot * np.exp(-2j * np.pi * (tau @ q_raw)).reshape(ph_shape)

    # 4b. time reversal: dv(-q) = conj(dv(q)) -- QE's dfile_minus_q, applied
    #     to the complete unfolded field (the real crystal->Cartesian matrix
    #     of step 5 commutes with the conjugation).
    if time_reversal:
        dv_rot = dv_rot.conj()
        q_raw = -q_raw

    # 4c. umklapp phase for folding q_raw -> q_rot into [0, 1). The fold is
    # NOT free: dvscf stores the PERIODIC part of a Bloch-form perturbation,
    # and that part is not periodic in q:
    #     dV_(q+G)(r) = dV_q(r) exp(-2 pi i G.x).
    # Verified on Al: the exact identity dv(-q) = conj(dv(q)) appears to fail
    # by a factor 2 when -q folds with G /= 0, and holds to the DFPT residual
    # (~3e-3) once this phase is restored. `wannier_transform_elph` carries
    # the same phase for u_(k+q); it is the same effect.
    q_rot = q_raw % 1.0
    q_rot[q_rot > 1.0 - 1e-9] = 0.0        # x % 1.0 can return 1.0 - eps
    G_fold = np.rint(q_rot - q_raw)
    if np.abs(G_fold - (q_rot - q_raw)).max() > 1e-6:
        raise ValueError(
            f"rotate_dvscf: fold vector {q_rot - q_raw} is not a reciprocal "
            "lattice vector -- q_frac is inconsistent with its own image."
        )
    if np.abs(G_fold).max() > 0.5:
        x = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid],
                                 indexing="ij"), axis=-1)
        dv_rot = dv_rot * np.exp(-2j * np.pi * (x @ G_fold))[None, None]

    # 5. crystal -> Cartesian
    dv_rot = np.einsum("kj,nk...->nj...", B, dv_rot)

    return dv_rot.reshape(n_modes, *grid), q_rot


def dvscf_star_routes(
    q_irr: np.ndarray,
    mesh: tuple[int, int, int],
    sym: dict,
    *,
    time_reversal: bool = True,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    For every point of a Gamma-centred q-mesh, pick a symmetry route from the
    irreducible set: ``routes[j] = (i, isym, tr)`` means
    ``rotate_dvscf(dv_irr[i], q_irr[i], isym, ..., time_reversal=bool(tr))``
    lands on mesh point ``j``.

    Routes that need NO umklapp fold are preferred, and plain rotations are
    preferred over time-reversed ones. Both kinds are correct,
    but they are not equally accurate, and the reason is worth stating because
    it looks like a bug when first met:

      * no fold  -- agreement with an independently computed dvscf is
        ~1e-10 (machine level for this data);
      * with fold -- L2 agreement ~1.4e-2, MEDIAN 2.2e-4, but the max-norm
        error reaches 7.5e-2. That max is set by about three grid points out
        of 291600, all at or adjacent to a nucleus. There QE's own stored
        dvscf violates the symmetry it should obey (at the Mg site of MgB2 the
        in-plane components of dV/du must vanish for a 2-fold about z, and
        they do not), so the discrepancy is in the data's treatment of the
        nuclear cusp, not in the rotation. It is harmless for the el-ph
        vertex, an integral over the whole grid in which three points carry
        weight ~1e-5 each -- worst case ~3% on lambda, and only on folded
        routes.

      Judge such comparisons with an L2 or percentile metric. ``max|d - ref| /
      max|ref|`` is the wrong tool for a field with near-singular structure at
      the nuclei and will read as a convention error when there is none.

    Parameters
    ----------
    q_irr : (n_irr, 3) fractional q as ph.x ACTUALLY used them -- do NOT fold
        into [0, 1) first. ph.x reports some star representatives with negative
        components, and dvscf stores the periodic part, which is not periodic
        in q; folding the label without applying ``exp(-2 pi i G.x)`` silently
        corrupts the data.
    mesh : (N1, N2, N3) of the full q-mesh, matching
        `interfaces.ase.structure.monkhorst_pack`'s enumeration.
    sym : `interfaces.ase.structure.crystal_symmetry_operations`'s output.
    time_reversal : also allow ``q -> -S q`` routes via ``dv(-q) =
        conj(dv(q))`` (`rotate_dvscf`'s own flag). ph.x reduces its q-grid
        WITH time reversal, so on a non-centrosymmetric crystal the ph.x
        wedge is only complete with this on. Harmless when the group already
        contains inversion (those routes are never preferred). Switch off
        for a Hamiltonian that is not time-reversal symmetric.
    tol : matching tolerance on fractional q. The default admits q_irr known
        to ~7 digits (e.g. converted from a ph.x stdout in cartesian
        2*pi/alat); it must stay well below the mesh spacing ``1/max(mesh)``.

    Returns
    -------
    (prod(mesh), 3) int64 -- ``(i, isym, tr)`` per mesh point.

    Raises
    ------
    ValueError if some mesh point is unreachable, which means ``q_irr`` is not
    a complete irreducible set for this mesh and symmetry group.
    """
    mesh_a = np.asarray(mesh, dtype=np.int64)
    q_irr = np.asarray(q_irr, dtype=np.float64)
    rot = np.asarray(sym["rotations"], dtype=np.float64)
    nq = int(np.prod(mesh_a))
    if tol >= 0.5 / mesh_a.max():
        raise ValueError(f"dvscf_star_routes: tol={tol:g} is not well below "
                         f"the mesh spacing 1/{mesh_a.max()}.")

    n1, n2, n3 = (int(m) for m in mesh_a)
    i, j, k = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
    qfull = np.stack([i.ravel() / n1, j.ravel() / n2, k.ravel() / n3], axis=-1)

    routes = np.full((nq, 3), -1, dtype=np.int64)
    # Route quality, worst to best: TR+fold(0) < plain+fold(1) < TR no-fold(2)
    # < plain no-fold(3). No-fold beats plain-ness: time reversal is an exact
    # conjugation while a fold rides on the nuclear-cusp grid data (docstring).
    quality = np.full(nq, -1, dtype=np.int64)
    tr_options = (0, 1) if time_reversal else (0,)
    for ii, qi in enumerate(q_irr):
        for isym in range(len(rot)):
            q_sym = np.linalg.inv(rot[isym]).T @ qi
            for tr in tr_options:
                qr = -q_sym if tr else q_sym
                d = np.abs(((qfull - qr) + 0.5) % 1.0 - 0.5).max(axis=1)
                hit = np.flatnonzero(d < tol)
                if hit.size != 1:
                    continue
                j0 = int(hit[0])
                nofold = bool(np.abs(qr - qfull[j0]).max() < tol)
                qual = 2 * int(nofold) + (1 - tr)
                if qual > quality[j0]:
                    routes[j0] = (ii, isym, tr)
                    quality[j0] = qual
    if (routes[:, 0] < 0).any():
        n = int((routes[:, 0] < 0).sum())
        raise ValueError(
            f"dvscf_star_routes: {n} of {nq} mesh points are unreachable from "
            "the given q_irr -- it is not a complete irreducible set for this "
            "mesh and symmetry group."
        )
    return routes
