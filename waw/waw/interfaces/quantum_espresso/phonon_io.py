"""
Reader for QE's real-space interatomic force-constant file (``q2r.x``'s
``flfrc`` output, e.g. ``<seed>.fc``) -- the phonon analogue of a Wannier
Hamiltonian ``H(R)``: q2r.x Fourier-transforms `ph.x`'s DFPT dynamical
matrices from a coarse q-mesh to real-space force constants C(R), exactly
mirroring how `pw2wannier90.x` + waw's own `compute_hr` Fourier-transform
electronic overlaps from a coarse k-mesh to H(R). `analysis.phonon` does
the reverse (native) interpolation back to an arbitrary q-point plus
diagonalization -- the phonon analogue of `core.hamiltonian.operator_k`/
`interpolate_bands`.

File format (transcribed directly from QE source, `PHonon/PH/do_q2r.f90`'s
write statements -- no official machine-readable spec exists):

    ntyp nat ibrav celldm(1..6)
    <for each type>   type_index  'Symbol'  mass_amu*AMU_RY
    <for each atom>   atom_index  type_index  x  y  z        (alat units)
    T/F                                                       (has_zstar)
    <if T: dielectric tensor (3x3), then per-atom Born
     effective-charge tensors (3x3 each)>
    nr1 nr2 nr3                                                (real-space grid)
    <for ipol,jpol in 1..3, na,nb in 1..nat, in that nesting order:>
        ipol jpol na nb
        <nr1*nr2*nr3 lines:>  m1 m2 m3  C(R)_{ipol,jpol,na,nb}

``mass_amu*AMU_RY`` uses QE's own `AMU_RY = AMU_AU/2` convention (verified
against QE's own bundled diamond example, `PHonon/examples/example19/
reference/diam.ifc`: file mass 10947.156286 / carbon's 12.0107 amu =
911.444, matching AMU_RY to 6 significant figures) -- `read_force_constants`
divides this back out to return masses in plain amu.

The `T`/dielectric-and-Born-charge block only appears for polar
insulators (`ph.x`'s ``epsil=.true.``/LO-TO splitting term); skipped
entirely (and not needed) for a metal like MgB2 -- `read_force_constants`
handles both branches.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_force_constants(path: str | Path) -> dict:
    """
    Parse a q2r.x real-space force-constant file.

    Returns a dict with:
      ntyp, nat            : int
      ibrav                 : int
      celldm                : (6,) float (Bohr / dimensionless angles, QE's own celldm convention)
      species               : list of str, length ntyp
      masses_amu            : (ntyp,) float, physical atomic mass units (AMU_RY factor divided out)
      types                 : (nat,) int, 0-based type index of each atom
      tau_alat              : (nat, 3) float, atomic positions in units of celldm(1) (alat)
      at_alat               : (3, 3) float or None -- lattice vectors in units of celldm(1)
                               (alat), rows = a1/a2/a3, ONLY present when ibrav == 0 (this
                               project's own universal convention, `write_pw_input` always
                               sets `ibrav=0`); q2r.x writes these 3 extra lines right after
                               the header specifically for ibrav=0 (`matdyn.f90`'s `writefc`:
                               `IF (ibrav == 0) WRITE(1,'(2x,3f15.9)') ((at(i,j),i=1,3),j=1,3)`)
                               -- None for a standard Bravais-lattice ibrav (e.g. the bundled
                               diamond example fixture, ibrav=2, which has no such lines)
      has_zstar             : bool
      epsilon               : (3, 3) float or None -- dielectric tensor, if has_zstar
      born_charges          : (nat, 3, 3) float or None -- Born effective charges, if has_zstar
      grid                  : (3,) int, (nr1, nr2, nr3)
      fc                    : (3, 3, nat, nat, nr1, nr2, nr3) float, Hartree/Bohr^2
                               (Ry/Bohr^2 in the file, converted here to Hartree, i.e. /2)
                               fc[ipol, jpol, na, nb, m1, m2, m3] = C(R)_{ipol,jpol}(na,nb)
                               with R = m1*a1 + m2*a2 + m3*a3 (crystal/lattice-vector
                               multiples, 0-based m1/m2/m3 here; the file uses 1-based
                               indices m1,m2,m3 in {1,...,nr}, understood as R spanning
                               a Wigner-Seitz-centered range via q2r.x's own convention --
                               see `analysis.phonon`'s docstring for how this maps to a
                               centered R-vector list)
    """
    path = Path(path)
    lines = path.read_text().split("\n")
    idx = 0

    def _next():
        nonlocal idx
        while lines[idx].strip() == "":
            idx += 1
        ln = lines[idx]
        idx += 1
        return ln

    header = _next().split()
    ntyp, nat, ibrav = int(header[0]), int(header[1]), int(header[2])
    celldm = np.array([float(x) for x in header[3:9]], dtype=np.float64)

    at_alat = None
    if ibrav == 0:
        at_alat = np.array([[float(x) for x in _next().split()] for _ in range(3)])

    AMU_RY = 911.4442  # QE's own AMU_AU/2 (AMU_AU = 1822.888...)

    species = []
    masses_amu = np.empty(ntyp)
    for _ in range(ntyp):
        # nt  'Symbol'  mass -- Fortran's list-directed WRITE quotes the
        # symbol explicitly, but a species symbol shorter than the internal
        # CHARACTER(LEN=3) field can print as ALL SPACES between the quotes
        # (a genuine q2r.x/matdyn.f90 cosmetic quirk, observed for e.g. 'Mg'
        # in a real MgB2 run) -- naive quote-stripping then split() silently
        # collapses that blank field away, misaligning nt/symbol/mass.
        # Splitting on the quote characters themselves keeps the 3 fields
        # distinct regardless of how much whitespace the symbol field holds.
        line = _next()
        before, symbol, after = line.split("'", 2)
        species.append(symbol.strip())
        nt = int(before.split()[0])
        masses_amu[nt - 1] = float(after.split()[-1]) / AMU_RY

    types = np.empty(nat, dtype=np.int64)
    tau_alat = np.empty((nat, 3))
    for _ in range(nat):
        parts = _next().split()
        ia = int(parts[0]) - 1
        types[ia] = int(parts[1]) - 1
        tau_alat[ia] = [float(x) for x in parts[2:5]]

    has_zstar = _next().strip().upper().startswith("T")
    epsilon = None
    born_charges = None
    if has_zstar:
        epsilon = np.array([[float(x) for x in _next().split()] for _ in range(3)])
        born_charges = np.empty((nat, 3, 3))
        for ia in range(nat):
            _next()  # atom index line
            born_charges[ia] = [[float(x) for x in _next().split()] for _ in range(3)]

    nr1, nr2, nr3 = (int(x) for x in _next().split())

    fc = np.empty((3, 3, nat, nat, nr1, nr2, nr3), dtype=np.float64)
    n_blocks = 3 * 3 * nat * nat
    for _ in range(n_blocks):
        ipol, jpol, na, nb = (int(x) - 1 for x in _next().split())
        for _ in range(nr1 * nr2 * nr3):
            parts = _next().split()
            m1, m2, m3 = (int(x) - 1 for x in parts[:3])
            fc[ipol, jpol, na, nb, m1, m2, m3] = float(parts[3])

    fc *= 0.5   # Ry/Bohr^2 -> Hartree/Bohr^2

    return {
        "ntyp": ntyp, "nat": nat, "ibrav": ibrav, "celldm": celldm, "at_alat": at_alat,
        "species": species, "masses_amu": masses_amu,
        "types": types, "tau_alat": tau_alat,
        "has_zstar": has_zstar, "epsilon": epsilon, "born_charges": born_charges,
        "grid": np.array([nr1, nr2, nr3]), "fc": fc,
    }


def read_ph_frequencies(ph_out: str, n_modes: int) -> np.ndarray:
    """
    Parse phonon frequencies (cm^-1) directly from `ph.x` stdout, one row
    per q-point in the ORDER they were computed (matching an explicit
    q-list run, e.g. `interfaces.quantum_espresso.io.
    write_ph_input_explicit_q`) -- genuine DFPT frequencies, not a
    force-constant Fourier interpolation, for a direct DFT-vs-interpolation
    phonon band comparison (the phonon analogue of `io.
    read_bands_eigenvalues` for electrons).

    `ph.x` prints each q-point's modes as::

        freq (    1) =       2.393647 [THz] =      79.843469 [cm-1]
        freq (    2) =       2.625675 [THz] =      87.583100 [cm-1]
        freq (    3) =       5.461610 [THz] =     182.179706 [cm-1]

    -- `n_modes` consecutive ``freq (...)`` lines per q-point (3*nat for
    a non-polar/no-LO-TO-splitting run).
    """
    pattern = re.compile(r"freq \(\s*\d+\) =\s*[-\d.]+ \[THz\] =\s*([-\d.]+) \[cm-1\]")
    matches = pattern.findall(Path(ph_out).read_text())
    if not matches:
        raise ValueError(f"no 'freq (...) [cm-1]' lines found in {ph_out}")
    freqs = np.array([float(x) for x in matches], dtype=np.float64)
    if len(freqs) % n_modes != 0:
        raise ValueError(
            f"{ph_out}: {len(freqs)} frequency lines not a multiple of n_modes={n_modes}"
        )
    return freqs.reshape(-1, n_modes)
