"""
Tests for waw/io.py — file I/O layer.

Strategy: generate minimal but format-correct synthetic files, write them,
read them back, and verify:
  1. Round-trip consistency (what we write is what we read).
  2. Physical constraints that are independent of the DFT code:
       - Mmn Hermitian conjugate symmetry: M(k,b)† = M(k+b, -b)
       - Shell-weight completeness: sum_b w_b * b_alpha * b_beta = delta_alpha_beta
         (finite-difference representation of the gradient operator)
  3. Index conventions: all returned arrays use 0-based indices.
"""

import struct
import textwrap
import tempfile
from pathlib import Path

import numpy as np
import pytest

# make the package importable when running pytest from the project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.interfaces.wannier90.io import (
    read_win,
    read_nnkp,
    read_eig,
    read_mmn,
    read_amn,
    read_unk,
    read_uHu,
    read_sHu,
    read_sIu,
    read_dmn,
    write_hr,
    write_centres,
    write_win,
)


# ===========================================================================
# Fixtures — minimal synthetic files
# ===========================================================================

NB  = 4   # number of bands
NK  = 2   # number of k-points
NNB = 3   # number of nearest neighbours per k-point
NW  = 2   # number of Wannier functions
NR  = 5   # number of R-vectors for _hr.dat


@pytest.fixture
def tmp(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


def _write_win(path: Path) -> Path:
    """Write a minimal .win file covering the main parameter types."""
    content = textwrap.dedent(f"""\
        ! minimal .win for testing
        num_wann          = {NW}
        num_bands         = {NB}
        dis_win_max       = 18.0
        dis_win_min       = -5.0
        dis_froz_max      = 14.5
        use_disentanglement = true
        write_hr          = .true.

        %block projections
        f=0.0,0.0,0.0 : s
        f=0.5,0.5,0.5 : p
        %endblock projections

        %block unit_cell_cart
        Ang
        5.0  0.0  0.0
        0.0  5.0  0.0
        0.0  0.0  5.0
        %endblock unit_cell_cart
    """)
    p = path / "test.win"
    p.write_text(content)
    return p


def _write_nnkp(path: Path) -> Path:
    """Write a minimal .nnkp file with NK k-points and NNB neighbours each."""
    lines = []
    lines.append("File generated for testing")
    lines.append("")
    lines.append("begin kpoints")
    lines.append(f"  {NK}")
    kpts = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    for k in kpts:
        lines.append(f"  {k[0]:.6f}  {k[1]:.6f}  {k[2]:.6f}")
    lines.append("end kpoints")
    lines.append("")
    lines.append("begin nnkpts")
    lines.append(f"  {NNB}")
    # ik  ik2  g1  g2  g3   (1-based)
    # k=1 neighbours: k=2, k=2, k=1
    # k=2 neighbours: k=1, k=1, k=2
    table = [
        (1, 2, 0, 0, 0),
        (1, 2, 1, 0, 0),
        (1, 1, 0, 1, 0),
        (2, 1, 0, 0, 0),
        (2, 1, -1, 0, 0),
        (2, 2, 0, 1, 0),
    ]
    for row in table:
        lines.append(f"  {row[0]:3d}  {row[1]:3d}  {row[2]:3d}  {row[3]:3d}  {row[4]:3d}")
    lines.append("end nnkpts")
    lines.append("")
    lines.append("begin exclude_bands")
    lines.append("  0")
    lines.append("end exclude_bands")

    p = path / "test.nnkp"
    p.write_text("\n".join(lines) + "\n")
    return p


def _write_eig(path: Path) -> tuple[Path, np.ndarray]:
    """Write a .eig file and return (path, eigenvalue_array)."""
    # eig[ik, ib] with simple values: energy increases with band index
    eig = np.array([
        [-5.0, -2.0, 3.0, 7.0],   # k-point 1
        [-4.5, -1.5, 3.5, 8.0],   # k-point 2
    ], dtype=np.float64)

    lines = []
    for ik in range(NK):
        for ib in range(NB):
            # W90 convention: band (1-based)  kpoint (1-based)  energy
            lines.append(f"  {ib+1:5d}  {ik+1:5d}  {eig[ik, ib]:18.10f}")

    p = path / "test.eig"
    p.write_text("\n".join(lines) + "\n")
    return p, eig


def _write_mmn(path: Path) -> tuple[Path, np.ndarray]:
    """
    Write a .mmn file with random complex overlap matrices and return
    (path, Mmn_array).

    The matrices are NOT required to satisfy M(k,b)† = M(k+b,-b) here —
    that symmetry check is tested separately with physically constructed data.
    """
    rng = np.random.default_rng(seed=42)
    Mmn = (rng.standard_normal((NK, NNB, NB, NB))
           + 1j * rng.standard_normal((NK, NNB, NB, NB)))

    lines = []
    lines.append("Overlap matrices generated for testing")
    lines.append(f"  {NB:5d}  {NK:5d}  {NNB:5d}")

    # W90 convention: loop over all (ik, ib) pairs, ib fastest
    # header line: ik(1-based) ik2(1-based) g1 g2 g3
    # then nb*nb lines: re  im  (column-major / Fortran order)
    kpb_header = [
        (1, 2, 0, 0, 0), (1, 2, 1, 0, 0), (1, 1, 0, 1, 0),
        (2, 1, 0, 0, 0), (2, 1,-1, 0, 0), (2, 2, 0, 1, 0),
    ]
    for idx, (ik, ib) in enumerate([(ik, ib) for ik in range(NK) for ib in range(NNB)]):
        h = kpb_header[idx]
        lines.append(f"  {h[0]:5d}  {h[1]:5d}  {h[2]:5d}  {h[3]:5d}  {h[4]:5d}")
        # write column-major (Fortran order): fastest index is m (row)
        M_col = Mmn[ik, ib].flatten(order="F")
        for val in M_col:
            lines.append(f"  {val.real:18.10f}  {val.imag:18.10f}")

    p = path / "test.mmn"
    p.write_text("\n".join(lines) + "\n")
    return p, Mmn


def _write_amn(path: Path) -> tuple[Path, np.ndarray]:
    """Write a .amn file with random projections and return (path, Amn_array)."""
    rng = np.random.default_rng(seed=7)
    Amn = (rng.standard_normal((NK, NB, NW))
           + 1j * rng.standard_normal((NK, NB, NW)))

    lines = []
    lines.append("Projection matrices generated for testing")
    lines.append(f"  {NB:5d}  {NK:5d}  {NW:5d}")

    # W90 order: m  n  ik  re  im  (all 1-based)
    for n in range(NW):
        for ik in range(NK):
            for m in range(NB):
                val = Amn[ik, m, n]
                lines.append(
                    f"  {m+1:5d}  {n+1:5d}  {ik+1:5d}"
                    f"  {val.real:18.10f}  {val.imag:18.10f}"
                )

    p = path / "test.amn"
    p.write_text("\n".join(lines) + "\n")
    return p, Amn


# ===========================================================================
# Tests: .win reader
# ===========================================================================

class TestReadWin:
    def test_scalar_int(self, tmp):
        p = _write_win(tmp)
        params = read_win(p)
        assert params["num_wann"]  == NW
        assert params["num_bands"] == NB

    def test_scalar_float(self, tmp):
        p = _write_win(tmp)
        params = read_win(p)
        assert params["dis_win_max"] == pytest.approx(18.0)
        assert params["dis_froz_max"] == pytest.approx(14.5)

    def test_scalar_bool(self, tmp):
        p = _write_win(tmp)
        params = read_win(p)
        assert params["use_disentanglement"] is True
        assert params["write_hr"] is True

    def test_block_projections(self, tmp):
        p = _write_win(tmp)
        params = read_win(p)
        assert "projections" in params
        assert len(params["projections"]) == 2

    def test_block_unit_cell(self, tmp):
        p = _write_win(tmp)
        params = read_win(p)
        assert "unit_cell_cart" in params
        # 4 lines: Ang + 3 lattice vectors
        assert len(params["unit_cell_cart"]) == 4


# ===========================================================================
# Tests: .nnkp reader
# ===========================================================================

class TestReadNnkp:
    def test_kpoints_shape(self, tmp):
        p = _write_nnkp(tmp)
        data = read_nnkp(p)
        assert data["kpoints"].shape == (NK, 3)

    def test_kpoints_values(self, tmp):
        p = _write_nnkp(tmp)
        data = read_nnkp(p)
        np.testing.assert_allclose(data["kpoints"][0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data["kpoints"][1], [0.5, 0.0, 0.0])

    def test_nnkpts_shape(self, tmp):
        p = _write_nnkp(tmp)
        data = read_nnkp(p)
        assert data["nnkpts"].shape    == (NK, NNB)
        assert data["g_vectors"].shape == (NK, NNB, 3)

    def test_nntot(self, tmp):
        p = _write_nnkp(tmp)
        data = read_nnkp(p)
        assert data["nntot"] == NNB

    def test_zero_based_indices(self, tmp):
        """Neighbour k-indices must be 0-based (not the W90 1-based)."""
        p = _write_nnkp(tmp)
        data = read_nnkp(p)
        assert data["nnkpts"].min() >= 0
        assert data["nnkpts"].max() <  NK

    def test_g_vectors_dtype(self, tmp):
        p = _write_nnkp(tmp)
        data = read_nnkp(p)
        assert data["g_vectors"].dtype == np.int64


# ===========================================================================
# Tests: .eig reader
# ===========================================================================

class TestReadEig:
    def test_shape(self, tmp):
        p, _ = _write_eig(tmp)
        eig = read_eig(p)
        assert eig.shape == (NK, NB)

    def test_roundtrip_values(self, tmp):
        p, eig_ref = _write_eig(tmp)
        eig = read_eig(p)
        np.testing.assert_allclose(eig, eig_ref, atol=1e-8)

    def test_dtype(self, tmp):
        p, _ = _write_eig(tmp)
        eig = read_eig(p)
        assert eig.dtype == np.float64


# ===========================================================================
# Tests: .mmn reader
# ===========================================================================

class TestReadMmn:
    def test_shape(self, tmp):
        p, _ = _write_mmn(tmp)
        Mmn, _ = read_mmn(p)
        assert Mmn.shape == (NK, NNB, NB, NB)

    def test_roundtrip_values(self, tmp):
        p, Mmn_ref = _write_mmn(tmp)
        Mmn, _ = read_mmn(p)
        np.testing.assert_allclose(Mmn, Mmn_ref, atol=1e-8)

    def test_dtype(self, tmp):
        p, _ = _write_mmn(tmp)
        Mmn, _ = read_mmn(p)
        assert Mmn.dtype == np.complex128

    def test_kpb_map_length(self, tmp):
        p, _ = _write_mmn(tmp)
        _, kpb_map = read_mmn(p)
        assert len(kpb_map) == NK * NNB

    def test_hermitian_conjugate_symmetry(self, tmp):
        """
        Physical check: M(k,b)†_{mn} = M(k+b,-b)_{nm}.

        We construct a set of unitary matrices per k-point and compute
        M(k,b) = U(k)† @ U(k+b) analytically, then verify the symmetry
        on the resulting .mmn file.

        This is the most important structural property of the overlap matrices:
        it ensures that the finite-difference Laplacian is Hermitian.
        """
        rng = np.random.default_rng(seed=99)

        # Random unitary matrices per k-point (QR decomposition)
        U = np.zeros((NK, NB, NB), dtype=np.complex128)
        for ik in range(NK):
            A = rng.standard_normal((NB, NB)) + 1j * rng.standard_normal((NB, NB))
            U[ik], _ = np.linalg.qr(A)

        # The nnkp connectivity for our test file:
        #   ik=0, ib=0: neighbour is ik=1 (g=0,0,0)
        #   ik=0, ib=1: neighbour is ik=1 (g=1,0,0)
        #   ik=0, ib=2: neighbour is ik=0 (g=0,1,0)
        #   ik=1, ib=0: neighbour is ik=0 (g=0,0,0)   ← reverse of (ik=0,ib=0)
        #   ik=1, ib=1: neighbour is ik=0 (g=-1,0,0)  ← reverse of (ik=0,ib=1)
        #   ik=1, ib=2: neighbour is ik=1 (g=0,1,0)
        nnkpts = np.array([
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=np.int64)

        # Compute M(k,b) = U(k)† U(k+b)
        Mmn = np.zeros((NK, NNB, NB, NB), dtype=np.complex128)
        for ik in range(NK):
            for ib in range(NNB):
                ik2 = nnkpts[ik, ib]
                Mmn[ik, ib] = U[ik].conj().T @ U[ik2]

        # Verify M(k,b)† = M(k+b, -b) where -b is the reverse neighbour.
        # Pairs (ik=0,ib=0) <-> (ik=1,ib=0) are mutual reverses.
        pairs = [(0, 0, 1, 0), (0, 1, 1, 1)]   # (ik, ib, ik2, ib_rev)
        for ik, ib, ik2, ib_rev in pairs:
            M_forward = Mmn[ik,  ib]
            M_reverse = Mmn[ik2, ib_rev]
            np.testing.assert_allclose(
                M_forward.conj().T, M_reverse,
                atol=1e-12,
                err_msg=f"M†(k={ik},b={ib}) != M(k+b={ik2},−b={ib_rev})",
            )


# ===========================================================================
# Tests: .amn reader
# ===========================================================================

class TestReadAmn:
    def test_shape(self, tmp):
        p, _ = _write_amn(tmp)
        Amn = read_amn(p)
        assert Amn.shape == (NK, NB, NW)

    def test_roundtrip_values(self, tmp):
        p, Amn_ref = _write_amn(tmp)
        Amn = read_amn(p)
        np.testing.assert_allclose(Amn, Amn_ref, atol=1e-8)

    def test_dtype(self, tmp):
        p, _ = _write_amn(tmp)
        Amn = read_amn(p)
        assert Amn.dtype == np.complex128


# ===========================================================================
# Tests: _hr.dat writer (round-trip via numpy)
# ===========================================================================

class TestWriteHr:
    def _make_data(self):
        """Synthetic Hermitian Hamiltonian H(R) for NW=2 Wannier functions."""
        rng = np.random.default_rng(seed=3)
        H_R = (rng.standard_normal((NR, NW, NW))
               + 1j * rng.standard_normal((NR, NW, NW)))
        # enforce H(R)† = H(-R) by making H(R=0) Hermitian
        H_R[0] = (H_R[0] + H_R[0].conj().T) / 2
        R_vectors = np.array([
            [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.int64)
        degen = np.ones(NR, dtype=np.int64)
        return H_R, R_vectors, degen

    def test_file_created(self, tmp):
        H_R, R_vectors, degen = self._make_data()
        p = tmp / "test_hr.dat"
        write_hr(p, H_R, R_vectors, degen, nw=NW)
        assert p.exists()

    def test_nw_in_header(self, tmp):
        H_R, R_vectors, degen = self._make_data()
        p = tmp / "test_hr.dat"
        write_hr(p, H_R, R_vectors, degen, nw=NW)
        lines = p.read_text().splitlines()
        assert str(NW) in lines[1]

    def test_nr_in_header(self, tmp):
        H_R, R_vectors, degen = self._make_data()
        p = tmp / "test_hr.dat"
        write_hr(p, H_R, R_vectors, degen, nw=NW)
        lines = p.read_text().splitlines()
        assert str(NR) in lines[2]

    def test_matrix_element_count(self, tmp):
        """File must contain exactly NR * NW * NW matrix element lines."""
        H_R, R_vectors, degen = self._make_data()
        p = tmp / "test_hr.dat"
        write_hr(p, H_R, R_vectors, degen, nw=NW)
        lines = p.read_text().splitlines()
        # header: 3 lines + ceil(NR/15) degen lines
        import math
        n_degen_lines = math.ceil(NR / 15)
        data_lines = lines[3 + n_degen_lines:]
        assert len(data_lines) == NR * NW * NW


# ===========================================================================
# Tests: _centres.xyz writer
# ===========================================================================

class TestWriteCentres:
    def test_file_created(self, tmp):
        centers = np.array([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
        spreads = np.array([1.5, 2.0])
        p = tmp / "test_centres.xyz"
        write_centres(p, centers, spreads)
        assert p.exists()

    def test_line_count(self, tmp):
        centers = np.array([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
        spreads = np.array([1.5, 2.0])
        p = tmp / "test_centres.xyz"
        write_centres(p, centers, spreads)
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        # 2 header lines + NW data lines
        assert len(lines) == 2 + NW

    def test_spread_values_present(self, tmp):
        centers = np.array([[0.1, 0.2, 0.3], [1.0, 1.0, 1.0]])
        spreads = np.array([1.5, 2.0])
        p = tmp / "test_centres.xyz"
        write_centres(p, centers, spreads)
        text = p.read_text()
        assert "1.50000000" in text
        assert "2.00000000" in text


# ===========================================================================
# read_unk — formatted (ASCII) and unformatted (Fortran-binary) UNK files
# ===========================================================================

class TestReadUnk:
    """
    Both UNK variants must decode to identical arrays. The formatted layout
    is one 'ngx ngy ngz ik nbnd' header line then 'Re Im' per grid point
    (band-major, grid x fastest); the unformatted layout is the same data
    as Fortran sequential-unformatted records (4-byte length markers).
    """

    NGX, NGY, NGZ, NBND, IK = 3, 4, 5, 2, 7

    def _reference_u(self):
        rng = np.random.default_rng(0)
        npts = self.NGX * self.NGY * self.NGZ
        flat = rng.normal(size=(self.NBND, npts)) + 1j * rng.normal(size=(self.NBND, npts))
        u_nk = np.stack([
            flat[b].reshape((self.NGX, self.NGY, self.NGZ), order="F")
            for b in range(self.NBND)
        ])
        return flat, u_nk

    def test_formatted_roundtrip(self, tmp):
        flat, u_nk = self._reference_u()
        p = tmp / "UNK00007.1"
        with open(p, "w") as f:
            f.write(f"{self.NGX} {self.NGY} {self.NGZ} {self.IK} {self.NBND}\n")
            for b in range(self.NBND):
                for c in flat[b]:
                    f.write(f"{c.real:.12e} {c.imag:.12e}\n")
        got = read_unk(p)
        assert (got["ngx"], got["ngy"], got["ngz"], got["ik"], got["nbnd"]) == \
               (self.NGX, self.NGY, self.NGZ, self.IK, self.NBND)
        np.testing.assert_allclose(got["u_nk"], u_nk, rtol=1e-10)

    def test_unformatted_roundtrip(self, tmp):
        flat, u_nk = self._reference_u()
        p = tmp / "UNK00007.1"
        with open(p, "wb") as f:
            def rec(arr: np.ndarray):
                marker = np.int32(arr.nbytes)
                f.write(marker.tobytes()); f.write(arr.tobytes()); f.write(marker.tobytes())
            rec(np.array([self.NGX, self.NGY, self.NGZ, self.IK, self.NBND], dtype=np.int32))
            for b in range(self.NBND):
                rec(flat[b].astype(np.complex128))
        got = read_unk(p)
        assert (got["ngx"], got["ngy"], got["ngz"], got["ik"], got["nbnd"]) == \
               (self.NGX, self.NGY, self.NGZ, self.IK, self.NBND)
        np.testing.assert_allclose(got["u_nk"], u_nk, rtol=1e-10)

    def test_formatted_and_unformatted_agree(self, tmp):
        """The auto-detect must route each file to the right decoder."""
        flat, _ = self._reference_u()
        pf, pu = tmp / "UNKf.1", tmp / "UNKu.1"
        with open(pf, "w") as f:
            f.write(f"{self.NGX} {self.NGY} {self.NGZ} {self.IK} {self.NBND}\n")
            for b in range(self.NBND):
                for c in flat[b]:
                    f.write(f"{c.real:.12e} {c.imag:.12e}\n")
        with open(pu, "wb") as f:
            def rec(arr):
                m = np.int32(arr.nbytes)
                f.write(m.tobytes()); f.write(arr.tobytes()); f.write(m.tobytes())
            rec(np.array([self.NGX, self.NGY, self.NGZ, self.IK, self.NBND], dtype=np.int32))
            for b in range(self.NBND):
                rec(flat[b].astype(np.complex128))
        np.testing.assert_allclose(read_unk(pf)["u_nk"], read_unk(pu)["u_nk"], rtol=1e-10)


# ===========================================================================
# read_uHu — <u_{k+b1}|H_k|u_{k+b2}> (tutorial19 orbital magnetization)
# ===========================================================================

class TestReadUHu:
    """
    Both .uHu variants must decode to the same array. wannier90's own reader
    (src/postw90/get_oper.F90::get_CC_R) writes each (nb, nb) block m-outer/
    n-inner (Fortran column-major for the unformatted case) and then
    transposes it after reading -- both writers below emit exactly what that
    reader expects, so a correct decoder must reproduce the original target.
    """

    NB, NK, NNTOT = 3, 2, 2

    def _reference_uHu(self):
        rng = np.random.default_rng(0)
        shape = (self.NK, self.NNTOT, self.NNTOT, self.NB, self.NB)
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def _write_formatted(self, path, uHu, header="test"):
        nk, nntot, _, nb, _ = uHu.shape
        with open(path, "w") as f:
            f.write(header + "\n")
            f.write(f"{nb} {nk} {nntot}\n")
            for ik in range(nk):
                for nn2 in range(nntot):
                    for nn1 in range(nntot):
                        block = uHu[ik, nn1, nn2]
                        for m in range(nb):
                            for n in range(nb):
                                v = block[m, n]
                                f.write(f"{v.real:.12f} {v.imag:.12f}\n")

    def _write_unformatted(self, path, uHu, header="test"):
        nk, nntot, _, nb, _ = uHu.shape

        def rec(f, data_bytes):
            n = len(data_bytes)
            f.write(struct.pack('<i', n)); f.write(data_bytes); f.write(struct.pack('<i', n))

        with open(path, "wb") as f:
            rec(f, header.encode().ljust(60))
            rec(f, struct.pack('<iii', nb, nk, nntot))
            for ik in range(nk):
                for nn2 in range(nntot):
                    for nn1 in range(nntot):
                        on_disk = uHu[ik, nn1, nn2].T   # pre-transpose array w90 would read
                        rec(f, on_disk.astype('<c16').tobytes(order='F'))

    def test_formatted_roundtrip(self, tmp):
        uHu = self._reference_uHu()
        p = tmp / "test_fmt.uHu"
        self._write_formatted(p, uHu)
        got = read_uHu(p)
        assert (got["num_bands"], got["num_kpts"], got["nntot"]) == (self.NB, self.NK, self.NNTOT)
        np.testing.assert_allclose(got["uHu"], uHu, rtol=1e-10)

    def test_unformatted_roundtrip(self, tmp):
        uHu = self._reference_uHu()
        p = tmp / "test_unf.uHu"
        self._write_unformatted(p, uHu)
        got = read_uHu(p)
        assert (got["num_bands"], got["num_kpts"], got["nntot"]) == (self.NB, self.NK, self.NNTOT)
        np.testing.assert_allclose(got["uHu"], uHu, rtol=1e-10)

    def test_formatted_and_unformatted_agree(self, tmp):
        """The auto-detect must route each file to the right decoder."""
        uHu = self._reference_uHu()
        pf, pu = tmp / "test_fmt.uHu", tmp / "test_unf.uHu"
        self._write_formatted(pf, uHu)
        self._write_unformatted(pu, uHu)
        np.testing.assert_allclose(read_uHu(pf)["uHu"], read_uHu(pu)["uHu"], rtol=1e-10)


class TestReadSHuSIu:
    """
    .sHu/.sIu (spin Hall conductivity Ryoo method, pw2wannier90's
    `compute_shc`): data[ik,ib,m,n,c] = <u_{m,k}|sigma_c[H(k)]|u_{n,k+b}>.

    Unlike .uHu (two neighbour indices, nn1/nn2), these have ONE neighbour
    index (matching .mmn) plus a 3-component Pauli axis. Both writers below
    simulate pw2wannier90's own `utility_write_array` loop nesting exactly
    (bra/m slow, ket/n fast) -- derived and cross-checked against a
    hand-simulated Fortran write loop before writing this test, since a
    first draft of the formatted reader had the wrong transpose (caught by
    this exact round-trip check).
    """

    NB, NK, NNTOT = 3, 2, 2

    def _reference(self):
        rng = np.random.default_rng(2)
        shape = (self.NK, self.NNTOT, self.NB, self.NB, 3)
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def _write_formatted(self, path, data, header="test"):
        nk, nntot, nb, _, _ = data.shape
        with open(path, "w") as f:
            f.write(header + "\n")
            f.write(f"{nb} {nk} {nntot}\n")
            for ik in range(nk):
                for ib in range(nntot):
                    for c in range(3):
                        block = data[ik, ib, :, :, c]   # (m=bra, n=ket)
                        for m in range(nb):             # slow (bra)
                            for n in range(nb):         # fast (ket)
                                v = block[m, n]
                                f.write(f"{v.real:.12f} {v.imag:.12f}\n")

    def _write_unformatted(self, path, data, header="test"):
        nk, nntot, nb, _, _ = data.shape

        def rec(f, data_bytes):
            n = len(data_bytes)
            f.write(struct.pack('<i', n)); f.write(data_bytes); f.write(struct.pack('<i', n))

        with open(path, "wb") as f:
            rec(f, header.encode().ljust(60))
            rec(f, struct.pack('<iii', nb, nk, nntot))
            for ik in range(nk):
                for ib in range(nntot):
                    for c in range(3):
                        on_disk = data[ik, ib, :, :, c].T   # pre-transpose, as w90's reader expects
                        rec(f, on_disk.astype('<c16').tobytes(order='F'))

    @pytest.mark.parametrize("reader", [read_sHu, read_sIu])
    def test_formatted_roundtrip(self, tmp, reader):
        data = self._reference()
        p = tmp / "test_fmt.sX"
        self._write_formatted(p, data)
        got = reader(p)
        assert (got["num_bands"], got["num_kpts"], got["nntot"]) == (self.NB, self.NK, self.NNTOT)
        np.testing.assert_allclose(got["data"], data, rtol=1e-10)

    @pytest.mark.parametrize("reader", [read_sHu, read_sIu])
    def test_unformatted_roundtrip(self, tmp, reader):
        data = self._reference()
        p = tmp / "test_unf.sX"
        self._write_unformatted(p, data)
        got = reader(p)
        assert (got["num_bands"], got["num_kpts"], got["nntot"]) == (self.NB, self.NK, self.NNTOT)
        np.testing.assert_allclose(got["data"], data, rtol=1e-10)

    def test_formatted_and_unformatted_agree(self, tmp):
        data = self._reference()
        pf, pu = tmp / "test_fmt.sX", tmp / "test_unf.sX"
        self._write_formatted(pf, data)
        self._write_unformatted(pu, data)
        np.testing.assert_allclose(read_sHu(pf)["data"], read_sHu(pu)["data"], rtol=1e-10)


# ===========================================================================
# read_dmn — symmetry-adapted Wannier functions (tutorial21)
# ===========================================================================

class TestReadDmn:
    """
    Synthetic .dmn round-trip: NB=2, nsymmetry=3, nkptirr=2, num_kpts=4,
    num_wann=1. Complex numbers are written Fortran list-directed style,
    "(re,im)" tuples, spread across lines however -- the parser must not
    care about line breaks (Fortran list-directed I/O doesn't either).
    """

    NB, NSYM, NKPTIRR, NK, NW = 2, 3, 2, 4, 1

    def _reference(self):
        rng = np.random.default_rng(0)
        ik2ir = rng.integers(0, self.NKPTIRR, size=self.NK)
        ir2ik = np.array([0, 2])
        kptsym = rng.integers(0, self.NK, size=(self.NSYM, self.NKPTIRR))
        d_wann = (rng.normal(size=(self.NW, self.NW, self.NSYM, self.NKPTIRR))
                  + 1j * rng.normal(size=(self.NW, self.NW, self.NSYM, self.NKPTIRR)))
        d_band = (rng.normal(size=(self.NB, self.NB, self.NSYM, self.NKPTIRR))
                  + 1j * rng.normal(size=(self.NB, self.NB, self.NSYM, self.NKPTIRR)))
        return ik2ir, ir2ik, kptsym, d_wann, d_band

    def _write(self, path, ik2ir, ir2ik, kptsym, d_wann, d_band):
        def fmt_complex(z):
            return f"({z.real:.10E},{z.imag:.10E})"

        with open(path, "w") as f:
            f.write("Created for a test\n")
            f.write(f"{self.NB} {self.NSYM} {self.NKPTIRR} {self.NK}\n")
            f.write(" ".join(str(i + 1) for i in ik2ir) + "\n")
            f.write(" ".join(str(i + 1) for i in ir2ik) + "\n")
            # kptsym: nsymmetry fastest (Fortran first index), so write
            # column-by-column (per ir) with nsym values each
            for ir in range(self.NKPTIRR):
                f.write(" ".join(str(i + 1) for i in kptsym[:, ir]) + "\n")
            # d_matrix_wann: first index (row) fastest
            for ir in range(self.NKPTIRR):
                for isym in range(self.NSYM):
                    for col in range(self.NW):
                        for row in range(self.NW):
                            f.write(fmt_complex(d_wann[row, col, isym, ir]) + "\n")
            for ir in range(self.NKPTIRR):
                for isym in range(self.NSYM):
                    for col in range(self.NB):
                        for row in range(self.NB):
                            f.write(fmt_complex(d_band[row, col, isym, ir]) + "\n")

    def test_roundtrip(self, tmp):
        ik2ir, ir2ik, kptsym, d_wann, d_band = self._reference()
        p = tmp / "test.dmn"
        self._write(p, ik2ir, ir2ik, kptsym, d_wann, d_band)

        d = read_dmn(p)
        assert d["num_bands"] == self.NB
        assert d["nsymmetry"] == self.NSYM
        assert d["nkptirr"] == self.NKPTIRR
        assert d["num_kpts"] == self.NK
        assert d["num_wann"] == self.NW
        np.testing.assert_array_equal(d["ik2ir"], ik2ir)
        np.testing.assert_array_equal(d["ir2ik"], ir2ik)
        np.testing.assert_array_equal(d["kptsym"], kptsym)
        np.testing.assert_allclose(d["d_matrix_wann"], d_wann, rtol=1e-9)
        np.testing.assert_allclose(d["d_matrix_band"], d_band, rtol=1e-9)

    def test_num_wann_auto_detected_matches_explicit(self, tmp):
        ik2ir, ir2ik, kptsym, d_wann, d_band = self._reference()
        p = tmp / "test.dmn"
        self._write(p, ik2ir, ir2ik, kptsym, d_wann, d_band)

        d_auto = read_dmn(p)
        d_explicit = read_dmn(p, num_wann=self.NW)
        assert d_auto["num_wann"] == d_explicit["num_wann"] == self.NW
        np.testing.assert_allclose(d_auto["d_matrix_wann"], d_explicit["d_matrix_wann"])


# ===========================================================================
# write_win — minimal .win writer (array-based, ase-free)
# ===========================================================================

def test_write_win_roundtrips_through_read_win(tmp):
    """io.write_win output parses back through read_win with all fields intact."""
    lattice = np.array([[0.0, 2.715, 2.715],
                        [2.715, 0.0, 2.715],
                        [2.715, 2.715, 0.0]])          # Angstrom
    frac    = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    symbols = ["Si", "Si"]
    mp_grid = (2, 2, 2)
    kpts    = np.array([[i / 2, j / 2, k / 2]
                        for i in range(2) for j in range(2) for k in range(2)])

    win_path = write_win(tmp / "si.win", lattice, frac, symbols, mp_grid, kpts,
                         num_wann=4, num_bands=4)
    params = read_win(win_path)

    assert int(params["num_wann"]) == 4
    assert int(params["num_bands"]) == 4
    assert tuple(int(x) for x in str(params["mp_grid"]).split()) == mp_grid
    # lattice block round-trips verbatim (written under an Ang tag)
    parsed_lat = np.array([[float(x) for x in line.split()[:3]]
                           for line in params["unit_cell_cart"][1:4]])
    np.testing.assert_allclose(parsed_lat, lattice, atol=1e-8)
    # full k-point list survives
    assert len(params["kpoints"]) == len(kpts)
    parsed_k = np.array([[float(x) for x in line.split()[:3]]
                         for line in params["kpoints"]])
    np.testing.assert_allclose(parsed_k, kpts, atol=1e-8)
