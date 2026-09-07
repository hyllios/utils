"""
Direct-input Quantum ESPRESSO driver: low-level QE I/O.

Writes ``pw.x`` scf/nscf and ``pw2wannier90.x`` inputs from an ``ase.Atoms``
plus explicit parameter dicts, and runs them under MPI (builds the input
text directly rather than via ASE's Espresso calculator, for full control
over noncollinear/spin-orbit namelists and the NSCF k-point card).

QE executables run as ``mpirun --mca pml ob1 --bind-to none -np N pw.x -in
in`` (``-in file``, never ``< file``); ``module load
quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k`` provides the binaries.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers


def mpirun_prefix(ncores: int) -> list[str]:
    return ["mpirun", "--mca", "pml", "ob1", "--bind-to", "none", "-np", str(ncores)]


def _mpi_env() -> dict:
    """
    Environment for QE subprocesses: pure MPI, one thread per rank. Forces
    thread envs to 1 so an inherited ``waw.set_num_threads`` OMP setting
    doesn't multiply with the ``-np N`` MPI ranks.
    """
    return {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


# ---------------------------------------------------------------------------
# QE input text
# ---------------------------------------------------------------------------

def _fmt_value(v) -> str:
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        # unwrap numpy scalars so repr() doesn't emit "np.float64(...)"
        v = v.item()
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, (list, tuple)):
        # Fortran list-directed input wants space-separated values
        return " ".join(_fmt_value(x) for x in v)
    return repr(v)


def _namelist(name: str, d: dict) -> str:
    lines = [f"&{name}"]
    for k, v in d.items():
        lines.append(f"   {k} = {_fmt_value(v)}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def _species_of(atoms):
    """Per-atom QE species label and the ordered unique label list.

    A nonzero ASE tag splits otherwise-identical atoms into distinct QE
    species (e.g. ``Ni1``/``Ni2`` for the two sublattices of an A-type
    antiferromagnet, so each can carry its own ``starting_magnetization``);
    tag 0 keeps the bare element symbol, so untagged cells produce exactly
    the same ``sorted(set(symbols))`` species list as before.
    """
    symbols = atoms.get_chemical_symbols()
    tags = atoms.get_tags()
    atom_labels = [f"{s}{t}" if t else s for s, t in zip(symbols, tags)]
    return sorted(set(atom_labels)), atom_labels


def _element_of(label: str) -> str:
    return label.rstrip("0123456789")


def magnetization_keys(atoms, moments, *, angle1=None, angle2=None) -> dict:
    """QE per-species magnetic namelist keys for a (possibly tag-split) cell.

    ``moments`` (and optional ``angle1``/``angle2``, degrees) map a species
    label (``'Ni1'``) or bare element (``'Ni'``) to a value; the returned
    ``{'starting_magnetization(i)': ..., 'angle1(i)': ...}`` dict is indexed
    to match ``write_pw_input``'s species ordering, ready to merge into
    ``system_extra``. Keeps the label->QE-index bookkeeping in one place so
    callers never hard-code species indices.
    """
    labels, _ = _species_of(atoms)

    def lookup(d, label):
        return d.get(label, d.get(_element_of(label)))

    keys = {}
    for i, label in enumerate(labels, start=1):
        m = lookup(moments, label)
        if m is not None:
            keys[f"starting_magnetization({i})"] = m
        if angle1 is not None and lookup(angle1, label) is not None:
            keys[f"angle1({i})"] = lookup(angle1, label)
        if angle2 is not None and lookup(angle2, label) is not None:
            keys[f"angle2({i})"] = lookup(angle2, label)
    return keys


def write_pw_input(
    path, atoms, *, control: dict, system: dict, electrons: dict,
    pseudopotentials: dict, kpoints,
    hubbard: dict | None = None, hubbard_projector: str = "ortho-atomic",
) -> Path:
    """
    Write a pw.x input. ``kpoints`` is one of ``("automatic", (n1,n2,n3),
    (s1,s2,s3))``, ``("crystal", array(nk,3))`` (explicit list, weight 1), or
    ``("gamma",)`` (``K_POINTS {gamma}``, QE's real-wavefunction Gamma-point
    trick for single-k-point molecules).

    ``hubbard`` (optional): DFT+U via QE's modern ``HUBBARD`` card (QE >= 7.1;
    the deprecated ``&system`` ``Hubbard_U(i)`` route is not used). Maps a
    manifold to either a bare U (eV) for the simplified Dudarev scheme --
    ``{"Ni-3d": 3.24}`` -- or a dict of card parameters for the full
    rotationally-invariant (Liechtenstein) scheme, e.g.
    ``{"Ni-3d": {"U": 3.24, "J": 0.68}}`` (a ``J`` line makes QE use
    ``lda_plus_u_kind=1``; ``J0``/``B``/``E2``/``E3`` are likewise passed
    through in that order). An element head (``Ni``) expands to every matching
    split species label (``Ni1-3d``, ``Ni2-3d`` for a tag-split AFM cell); a
    full label head (``Ni1-3d``) targets just that species.
    ``hubbard_projector`` is the card's projector type (``ortho-atomic``
    default, also ``atomic``/``norm-atomic``/``wf``/``pseudo``). The card
    presence alone enables +U -- no namelist flag.
    Cell in Angstrom; positions in crystal coordinates. Default ``ibrav=0``
    writes an explicit ``CELL_PARAMETERS`` card from ``atoms.get_cell()``;
    passing an explicit ``ibrav`` (plus the matching ``celldm(...)`` keys)
    in ``system`` omits that card instead, letting QE construct the cell
    analytically from ``ibrav``+``celldm`` -- needed when ``ibrav=0``'s
    finite-precision ``CELL_PARAMETERS`` text causes QE's symmetry finder to
    miss point-group operations it should find (seen with ``ph.x``, which
    is far more sensitive to this than ``pw.x`` scf/nscf). The caller must
    then make sure ``atoms.get_cell()`` is EXACTLY the cell that ``ibrav``
    generates (same vectors, not just the same lattice up to a rotation),
    since ``recip_lattice``/``real_lattice`` and the ``.nnkp`` b-vectors are
    all built from ``atoms.get_cell()`` directly.
    """
    path = Path(path)
    species, atom_labels = _species_of(atoms)
    sys_full = dict(system)
    sys_full.setdefault("ibrav", 0)
    sys_full["nat"] = len(atoms)
    sys_full["ntyp"] = len(species)

    text = _namelist("control", control)
    text += _namelist("system", sys_full)
    text += _namelist("electrons", electrons)

    text += "ATOMIC_SPECIES\n"
    for s in species:
        el = _element_of(s)
        pseudo = pseudopotentials.get(s, pseudopotentials.get(el))
        text += f" {s} {atomic_masses[atomic_numbers[el]]:.4f} {pseudo}\n"

    if sys_full.get("ibrav", 0) == 0:
        cell = atoms.get_cell()[:]
        text += "CELL_PARAMETERS angstrom\n"
        for row in cell:
            text += f" {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n"
    # else: ibrav != 0 -- cell comes from ibrav + celldm(...) in `system`,
    # given exactly (no CELL_PARAMETERS card); caller is responsible for
    # making atoms.get_cell() match that ibrav's convention exactly, since
    # everything downstream (recip_lattice/real_lattice, .nnkp b-vectors)
    # reads atoms.cell directly.

    text += "ATOMIC_POSITIONS crystal\n"
    for s, p in zip(atom_labels, atoms.get_scaled_positions()):
        text += f" {s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n"

    kind = kpoints[0]
    if kind == "automatic":
        _, grid, shift = kpoints
        text += "K_POINTS automatic\n"
        text += f" {grid[0]} {grid[1]} {grid[2]} {shift[0]} {shift[1]} {shift[2]}\n"
    elif kind == "crystal":
        kpts = np.asarray(kpoints[1], dtype=float)
        text += "K_POINTS crystal\n"
        text += f"{len(kpts)}\n"
        w = 1.0 / len(kpts)
        for k in kpts:
            text += f" {k[0]:.10f} {k[1]:.10f} {k[2]:.10f} {w:.10e}\n"
    elif kind == "gamma":
        text += "K_POINTS {gamma}\n"
    else:
        raise ValueError(f"unknown kpoints kind {kind!r}")

    if hubbard:
        text += f"HUBBARD ({hubbard_projector})\n"
        for manifold, spec in hubbard.items():
            head, orb = manifold.rsplit("-", 1)
            targets = [head] if head in species else \
                [s for s in species if _element_of(s) == head]
            params = spec if isinstance(spec, dict) else {"U": spec}
            for s in targets:
                for key in ("U", "J", "J0", "B", "E2", "E3"):
                    if key in params:
                        text += f"{key} {s}-{orb} {params[key]}\n"

    path.write_text(text)
    return path


def write_ph_input(path, *, prefix: str, outdir: str, fildyn: str,
                   nq: tuple[int, int, int], tr2_ph: float = 1.0e-14,
                   extra: dict | None = None) -> Path:
    """
    Write a ``ph.x`` (DFPT phonon) input: a q-mesh dynamical-matrix run
    (``ldisp=.true.``), the standard `q2r.x`-ready recipe (`interfaces.
    quantum_espresso.phonon_io.read_force_constants`'s own docstring
    explains the full q2r.x/matdyn.x-style pipeline this feeds).

    ``fildyn`` is the dynamical-matrix file prefix `q2r.x`'s own
    ``fildyn`` input must match. ``extra`` merges in any additional
    ``&inputph`` keys (e.g. ``epsil=True`` for a polar insulator's
    LO-TO splitting term -- not needed for a metal).
    """
    path = Path(path)
    inputph = {
        "prefix": prefix, "outdir": outdir, "fildyn": fildyn,
        "ldisp": True, "nq1": nq[0], "nq2": nq[1], "nq3": nq[2],
        "tr2_ph": tr2_ph, **(extra or {}),
    }
    path.write_text(_namelist("inputph", inputph))
    return path


def write_ph_input_explicit_q(path, *, prefix: str, outdir: str, fildyn: str,
                              qpoints_tpiba: np.ndarray, tr2_ph: float = 1.0e-14,
                              extra: dict | None = None) -> Path:
    """
    Write a ``ph.x`` input computing an EXPLICIT list of q-points
    (``qplot=.true.``), each independently (``q_in_band_form=.false.``,
    QE's own docs: "only the list of q points given as input is
    calculated. The weights are not used.") -- unlike ``write_ph_input``'s
    ``ldisp=.true.``, this does NOT reduce the list to a symmetry-
    irreducible wedge, so a full coarse mesh (all N1*N2*N3 points, not
    just the irreducible ones) gets a genuine per-q DFPT/dvscf run at
    every point.

    Needed for `analysis.elph`'s native double real-space Wannier
    transform of the electron-phonon coupling: g(k,q) has to be
    Fourier-summed over the FULL uniform q-mesh (matching the electron
    k-mesh), since `ph.x`'s own symmetry-star reconstruction of dV at
    the non-irreducible q's (``dvscf_star``) lives in the same
    ``electron_phonon='Wannier'``/``elph_mat`` codepath this project
    avoids (see `interfaces.quantum_espresso.dvscf_io`'s docstring).

    ``qpoints_tpiba`` : (nq, 3) float, Cartesian coordinates in units of
    2*pi/alat (QE's own ``qplot`` convention) -- NOT crystal/fractional;
    convert via ``q_frac @ recip_lattice(atoms) * alat_bohr / (2*pi)``.
    """
    path = Path(path)
    inputph = {
        "prefix": prefix, "outdir": outdir, "fildyn": fildyn,
        "ldisp": True, "qplot": True, "q_in_band_form": False,
        "tr2_ph": tr2_ph, **(extra or {}),
    }
    text = _namelist("inputph", inputph)
    text += f"{len(qpoints_tpiba)}\n"
    for q in qpoints_tpiba:
        text += f" {q[0]:.10f} {q[1]:.10f} {q[2]:.10f}  1\n"
    path.write_text(text)
    return path


def write_q2r_input(path, *, fildyn: str, flfrc: str, zasr: str = "crystal") -> Path:
    """Write a ``q2r.x`` input: Fourier-transform `ph.x`'s q-mesh dynamical
    matrices (``fildyn``) to real-space force constants (``flfrc``, the
    file `interfaces.quantum_espresso.phonon_io.read_force_constants`
    parses)."""
    path = Path(path)
    path.write_text(_namelist("input", {"fildyn": fildyn, "zasr": zasr, "flfrc": flfrc}))
    return path


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_pw(input_path, output_path, *, ncores: int = 16, pw: str = "pw.x") -> Path:
    input_path, output_path = Path(input_path), Path(output_path)
    with open(output_path, "w") as out:
        subprocess.run(mpirun_prefix(ncores) + [pw, "-in", input_path.name],
                       cwd=input_path.parent, stdout=out, stderr=subprocess.STDOUT,
                       env=_mpi_env(), check=True)
    from waw.utils.runs import autostamp
    autostamp(input_path.parent, code="qe",
              settings={"ncores": ncores, "input": input_path.name})
    return output_path


def read_fermi_energy(pw_out) -> float:
    """
    Read the Fermi energy (eV) from a ``pw.x`` scf/nscf output.

    Metals print ``the Fermi energy is  X ev`` directly. A fixed-occupation
    insulator prints no Fermi level, so this falls back to the mid-gap of
    ``highest occupied, lowest unoccupied level`` or, failing that, the
    ``highest occupied level`` (VBM). Takes the last (converged) match.
    """
    text = Path(pw_out).read_text()

    fermi = [l for l in text.splitlines() if "the Fermi energy is" in l]
    if fermi:
        return float(fermi[-1].split()[-2])

    hol = [l for l in text.splitlines() if "highest occupied, lowest unoccupied" in l]
    if hol:
        toks = hol[-1].split()
        return 0.5 * (float(toks[-2]) + float(toks[-1]))

    ho = [l for l in text.splitlines() if "highest occupied level" in l]
    if ho:
        return float(ho[-1].split()[-1])

    raise ValueError(f"no Fermi energy / band edge found in {pw_out}")


def read_bands_eigenvalues(pw_out: str | Path, nbnd: int) -> np.ndarray:
    """
    Parse eigenvalues (eV) from a ``pw.x`` ``calculation='bands'`` (or
    ``'nscf'``) run, one row per k-point in the ORDER the ``K_POINTS``
    card listed them (matching a ``('crystal', kpts)`` `write_pw_input`
    call, e.g. an ASE ``band_path``'s own k-points) -- for a direct
    DFT-vs-Wannier-interpolation band comparison plot.

    QE prints each k-point as::

        k = 0.0000 0.0000 0.0000 (   123 PWs)   bands (ev):

        -5.65   5.65   5.65 ...

    Needs ``verbosity='high'`` in ``&control`` -- QE otherwise suppresses
    this per-k printout once there are more than 100 k-points (a `bands`
    run along a several-hundred-point path always has more).
    """
    lines = Path(pw_out).read_text().splitlines()
    bands = []
    i = 0
    while i < len(lines):
        if "bands (ev)" in lines[i]:
            vals: list[float] = []
            i += 1
            while len(vals) < nbnd and i < len(lines):
                if lines[i].strip():
                    vals.extend(float(x) for x in lines[i].split())
                i += 1
            bands.append(vals[:nbnd])
        else:
            i += 1
    if not bands:
        raise ValueError(f"no 'bands (ev)' blocks found in {pw_out}")
    return np.array(bands, dtype=np.float64)


def gamma_only_half_shell_nnkp() -> dict:
    """
    Neighbour topology for a Gamma-only (``mp_grid = (1,1,1)``) run.

    QE's real-wavefunction storage only holds half the reciprocal sphere, so
    ``pw2wannier90.x`` aborts on the full +-b shell ``generate_nnkp()`` would
    otherwise pick. Emits the half shell of 3 unit-cell b-vectors
    (+x,+y,+z) instead, matching real ``wannier90.x -pp``'s Gamma-only
    convention; waw's weight solver and Z-matrix handle this half-shell
    convention (``half_shell``, auto-detected from a missing -b partner) on
    the consuming end.
    """
    return {"nnkpts": np.zeros((1, 3), dtype=np.int64),
            "g_vectors": np.eye(3, dtype=np.int64)[None, :, :]}


def run_pw2wannier90(workdir, seedname, inp: dict, *, ncores: int = 4,
                     pw2wannier90: str = "pw2wannier90.x") -> None:
    """
    Run pw2wannier90 (reads the .nnkp, writes .mmn/.amn/.eig[/.spn]).

    Must run under ``mpirun`` -- pw2wannier90.x aborts in ``MPI_Init`` if run
    bare in this openmpi build.
    """
    workdir = Path(workdir)
    (workdir / f"{seedname}.pw2wan.in").write_text(_namelist("inputpp", inp))
    with open(workdir / f"{seedname}.pw2wan.out", "w") as fout:
        subprocess.run(
            mpirun_prefix(ncores) + [pw2wannier90, "-in", f"{seedname}.pw2wan.in"],
            cwd=workdir, stdout=fout, stderr=subprocess.STDOUT, env=_mpi_env(), check=True)


def _read_fortran_records(path):
    """Yield the byte body of each record of a Fortran sequential-unformatted
    file (gfortran: each record framed by a leading and trailing int32 length
    marker)."""
    import struct
    data = Path(path).read_bytes()
    pos, n = 0, len(data)
    while pos < n:
        (rl,) = struct.unpack("<i", data[pos:pos + 4]); pos += 4
        body = data[pos:pos + rl]; pos += rl
        (rl2,) = struct.unpack("<i", data[pos:pos + 4]); pos += 4
        if rl != rl2:
            raise ValueError(f"{path}: corrupt Fortran record ({rl} != {rl2})")
        yield body


def read_charge_density(save_dir, grid_shape, spin: int = 0) -> np.ndarray:
    """SCF valence charge density rho(r) (electrons/Bohr^3) on ``grid_shape``,
    read from QE's ``charge-density.dat`` in a ``<prefix>.save`` directory
    (Fortran binary, no HDF5 needed). QE writes rho(G) at Miller indices; we
    scatter onto the FFT grid and inverse-transform.

    Record layout (QE ``Modules/io_base.f90::write_rhog``): (1) gamma_only,
    ngm_g, nspin; (2) b1,b2,b3; (3) mill_g(3,ngm_g); (4..) rho(G) per spin.

    ``grid_shape`` must be the dense FFT grid the G-vectors live on (the same
    grid as the ``dvscf``/UNK, e.g. `interfaces.wannier90.io.read_unk`'s).
    """
    import struct
    recs = _read_fortran_records(Path(save_dir) / "charge-density.dat")
    r1 = next(recs)
    gamma_only = bool(r1[0])
    ngm_g, nspin = struct.unpack("<ii", r1[4:12])
    next(recs)                                              # b1,b2,b3 (unused)
    mill = np.frombuffer(next(recs), dtype="<i4").reshape(ngm_g, 3)   # (m1,m2,m3) per G
    rhog = None
    for s in range(nspin):
        body = next(recs)
        if s == spin:
            rhog = np.frombuffer(body, dtype="<c16").copy()
    if gamma_only:
        raise NotImplementedError(
            "read_charge_density: gamma_only charge density not supported "
            "(the el-ph runs use a k/q mesh, not gamma_only).")

    grid = tuple(int(n) for n in grid_shape)
    ntot = int(np.prod(grid))
    idx = tuple(mill[:, a] % grid[a] for a in range(3))
    rho_g = np.zeros(grid, dtype=np.complex128)
    rho_g[idx] = rhog
    return (np.fft.ifftn(rho_g) * ntot).real
