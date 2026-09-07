"""
Reader for VASP's WAVECAR: plane-wave coefficients of the Kohn-Sham states.

Needed because band unfolding works on the coefficients themselves, not on
eigenvalues -- `analysis.unfolding` contracts them against the primitive-cell
reciprocal lattice.

THE FILE FORMAT is a sequence of fixed-length records of `recl` bytes:

  record 1                  : recl, nspin, rtag                  (float64)
  record 2                  : nkpts, nbands, encut, cell(3x3)     (float64)
  then, for each spin s and k-point k:
    record                  : nplw, k_frac(3), then per band
                              (Re eps, Im eps, occupation)        (float64)
    nbands records          : the coefficients, complex64 if rtag == 45200,
                              complex128 if rtag == 45210

`nplw` counts the coefficients stored per band, so for a NON-COLLINEAR run
(LSORBIT) it is TWICE the number of plane waves: the two spinor components are
concatenated. That factor is the easiest thing in this format to get wrong, so
`gvectors` asserts its own G-count against it -- see `check_plane_wave_count`.

THE G-VECTOR ORDER IS NOT STORED and has to be regenerated exactly as VASP
enumerates it, otherwise coefficients are silently paired with the wrong G:
a triple loop with the THIRD index outermost over an FFT grid whose indices are
folded to [-n/2, n/2), keeping those with hbar^2|k+G|^2/2m < ENCUT. An ordering
mistake does not raise -- it quietly scrambles the wavefunction -- which is why
the count assertion matters and why `analysis.unfolding` divides by the total
pseudo-norm rather than assuming it is 1.

Units: eV and Angstrom on the file's own terms (this is an interface module);
convert at the boundary as usual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# hbar^2 / 2 m_e in eV Angstrom^2, BUILT FROM VASP'S OWN CONSTANTS and not from
# CODATA. VASP decides which G fall inside ENCUT using RYTOEV = 13.605826 and
# AUTOA = 0.529177249, giving 3.8099821..., where the CODATA value is
# 3.8100340... A relative difference of 1.4e-5 sounds ignorable and is not: at
# ENCUT = 450 eV it moves the sphere boundary by 0.006 eV, and on a 2x2 CuI slab
# exactly 4 G-vectors per k-point sit in that sliver at 6 of 28 k-points. Using
# the physically better number reproduced VASP's plane-wave count at only 22 of
# 28 k-points, and a wrong count means coefficients get paired with the wrong G.
RYTOEV = 13.605826
AUTOA = 0.529177249
HSQDTM = RYTOEV * AUTOA ** 2


@dataclass
class WavecarHeader:
    recl: int
    nspin: int
    rtag: int
    nkpts: int
    nbands: int
    encut: float
    cell: np.ndarray          # (3, 3) real lattice rows, Angstrom
    noncollinear: bool        # LSORBIT: two spinor components per band

    @property
    def coeff_dtype(self):
        return np.complex64 if self.rtag == 45200 else np.complex128

    @property
    def recip(self) -> np.ndarray:
        """Reciprocal lattice rows WITHOUT 2 pi, i.e. inv(cell).T."""
        return np.linalg.inv(self.cell).T


class Wavecar:
    """
    Lazy reader: the header and per-k metadata are read on construction, the
    coefficients only when asked for. A single band record here is ~1 MiB and
    these files run to hundreds of GiB, so nothing is cached by default.
    """

    def __init__(self, path, noncollinear: bool | None = None):
        self.path = str(path)
        self._fh = open(self.path, "rb")
        rec = np.fromfile(self._fh, dtype=np.float64, count=3)
        recl, nspin, rtag = int(rec[0]), int(rec[1]), int(rec[2])
        if rtag not in (45200, 45210):
            raise ValueError(f"{path}: unsupported rtag {rtag} (expected 45200 "
                             f"or 45210)")
        self._fh.seek(recl)
        hdr = np.fromfile(self._fh, dtype=np.float64, count=12)
        nkpts, nbands = int(hdr[0]), int(hdr[1])
        cell = hdr[3:12].reshape(3, 3)

        self.header = WavecarHeader(recl=recl, nspin=nspin, rtag=rtag,
                                    nkpts=nkpts, nbands=nbands, encut=hdr[2],
                                    cell=cell, noncollinear=False)
        self._read_kpoint_records()
        # Decide collinear vs spinor by which one makes VASP's own count add up,
        # unless told: with LSORBIT the record holds 2 x npw coefficients.
        if noncollinear is None:
            noncollinear = self._infer_noncollinear()
        self.header.noncollinear = bool(noncollinear)

    # ------------------------------------------------------------------ meta --
    def _read_kpoint_records(self):
        h = self.header
        self.nplw = np.zeros((h.nspin, h.nkpts), dtype=np.int64)
        self.kpoints = np.zeros((h.nspin, h.nkpts, 3))
        self.eigenvalues = np.zeros((h.nspin, h.nkpts, h.nbands))
        self.occupations = np.zeros((h.nspin, h.nkpts, h.nbands))
        for s in range(h.nspin):
            for k in range(h.nkpts):
                self._fh.seek(self._krecord(s, k) * h.recl)
                buf = np.fromfile(self._fh, dtype=np.float64,
                                  count=4 + 3 * h.nbands)
                self.nplw[s, k] = int(buf[0])
                self.kpoints[s, k] = buf[1:4]
                tri = buf[4:].reshape(h.nbands, 3)
                self.eigenvalues[s, k] = tri[:, 0]
                self.occupations[s, k] = tri[:, 2]

    def _krecord(self, spin, kpt) -> int:
        """0-based record index of the (spin, kpt) metadata record."""
        h = self.header
        per_k = 1 + h.nbands
        return 2 + spin * h.nkpts * per_k + kpt * per_k

    def _infer_noncollinear(self) -> bool:
        n_collinear = len(self._generate_gvectors(self.kpoints[0, 0]))
        n_stored = int(self.nplw[0, 0])
        if n_stored == n_collinear:
            return False
        if n_stored == 2 * n_collinear:
            return True
        raise ValueError(
            f"{self.path}: stored coefficients per band ({n_stored}) match "
            f"neither the plane-wave count ({n_collinear}) nor twice it "
            f"({2 * n_collinear}). The G-vector enumeration or ENCUT is wrong; "
            f"pairing coefficients with G in that state would silently scramble "
            f"the wavefunction.")

    # ------------------------------------------------------------ G vectors --
    def _fft_grid(self) -> np.ndarray:
        """Half-widths of the index range VASP scans, per direction."""
        h = self.header
        b = h.recip                                  # rows, 1/Angstrom, no 2 pi
        kmax = np.sqrt(h.encut / HSQDTM) / (2 * np.pi)
        # +2, not +1: the scan only has to CONTAIN the sphere -- the energy test
        # selects. A range that is tight by one silently drops the outermost
        # shell at some k (it cost 4 vectors at 6 of 28 k-points here) and the
        # count check then fails for a reason that looks like the cutoff.
        return np.array([int(np.ceil(kmax / np.linalg.norm(b[i]))) + 2
                         for i in range(3)])

    def _generate_gvectors(self, kvec) -> np.ndarray:
        """
        Integer G triplets in VASP's own order for a given k (fractional).

        The loop nesting is what fixes the order: third index outermost, first
        innermost, each folded to [-n/2, n/2).
        """
        h = self.header
        n = self._fft_grid()
        # FFT ORDER, not monotonic order: VASP walks each direction as
        # 0, 1, ..., n, -n, ..., -1 (the usual FFT frequency layout), so the SET
        # of kept G is independent of the scan width but their ORDER is not.
        # Enumerating -n..+n instead gives the same count -- the check still
        # passes -- while pairing every coefficient with the wrong G. It shows up
        # as spectral weights stuck near 1/det(M), the average over the folded
        # images, because the mask then selects an essentially random quarter of
        # the plane waves.
        def fft_order(m):
            return np.concatenate([np.arange(0, m + 1), np.arange(-m, 0)])
        f3, f2, f1 = fft_order(n[2]), fft_order(n[1]), fft_order(n[0])
        G3, G2, G1 = np.meshgrid(f3, f2, f1, indexing="ij")
        G = np.stack([G1.ravel(), G2.ravel(), G3.ravel()], axis=1)
        kg = (G + np.asarray(kvec)) @ (2 * np.pi * h.recip)
        energy = HSQDTM * (kg ** 2).sum(axis=1)
        return G[energy < h.encut]

    def gvectors(self, kpt: int, spin: int = 0) -> np.ndarray:
        """(npw, 3) integer G triplets matching `coefficients(kpt, band)`."""
        G = self._generate_gvectors(self.kpoints[spin, kpt])
        expect = int(self.nplw[spin, kpt])
        if self.header.noncollinear:
            expect //= 2
        if len(G) != expect:
            raise ValueError(
                f"{self.path}: generated {len(G)} G-vectors at k-point {kpt} "
                f"but VASP stored {expect}. Refusing to return a mismatched "
                f"set -- the coefficients would be paired with the wrong G.")
        return G

    def check_plane_wave_count(self, spin: int = 0) -> dict:
        """Regenerate G at every k and compare with VASP's count.

        The single strongest check on the reader: the enumeration has to agree
        exactly, at every k, or the G ordering is wrong somewhere.
        """
        out = {}
        for k in range(self.header.nkpts):
            expect = int(self.nplw[spin, k])
            if self.header.noncollinear:
                expect //= 2
            out[k] = (len(self._generate_gvectors(self.kpoints[spin, k])), expect)
        bad = {k: v for k, v in out.items() if v[0] != v[1]}
        return dict(ok=not bad, mismatches=bad, checked=len(out))

    # --------------------------------------------------------- coefficients --
    def coefficients(self, kpt: int, band: int, spin: int = 0) -> np.ndarray:
        """
        Plane-wave coefficients of one band.

        Returns (npw,) for a collinear run and (2, npw) for a spinor one, the
        two spinor components split apart.
        """
        h = self.header
        n = int(self.nplw[spin, kpt])
        offset = (self._krecord(spin, kpt) + 1 + band) * h.recl
        self._fh.seek(offset)
        c = np.fromfile(self._fh, dtype=h.coeff_dtype, count=n)
        if h.noncollinear:
            return c.reshape(2, n // 2)
        return c

    def pseudo_norm(self, kpt: int, band: int, spin: int = 0) -> float:
        """
        sum |C_G|^2 of the PSEUDO wavefunction. NOT 1, and not a bug.

        These are PAW pseudo-wavefunctions: the augmentation charge inside the
        spheres carries the remaining weight, so the plane-wave part alone sums
        to less than one, and the more localised the state the less it holds.
        Measured on this CuI slab: 0.28 for the -71 eV semicore state, 0.996 for
        a free-electron-like band at +13 eV. Anything that needs a normalised
        quantity (the unfolding spectral weight, for one) must divide by this
        rather than assume 1.
        """
        c = self.coefficients(kpt, band, spin)
        return float((np.abs(c) ** 2).sum())

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
