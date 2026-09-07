"""Band unfolding (Popescu-Zunger) and the WAVECAR reader it runs on.

WHAT IS AND IS NOT COVERED HERE. The G-vector ENUMERATION ORDER cannot be
checked without a real VASP file -- it is not stored, and any test that
recomputes it with the same rule is circular. It was validated against the real
thing: on a 71-atom CuI(111) 2x2 WAVECAR the reader's plane-wave count matches
VASP's at every one of 28 k-points, and the unfolded weights agree with
easyunfold to 2.9e-7. What these tests cover is everything else, on a WAVECAR
this module writes itself: the record layout and offsets, both rtag precisions,
spinor splitting, and the projection algebra (k-point matching including time
reversal, the primitive mask, PAW-norm normalisation, completeness over the
folded images).

Two traps are pinned down by assertions rather than comments, because both
produce plausible wrong numbers instead of errors: VASP selects G with its OWN
hbar^2/2m built from RYTOEV and AUTOA rather than CODATA, and PAW pseudo-norms
are not 1 so the weights must be divided by them.
"""
import numpy as np
import pytest

from waw.analysis.unfolding import (
    match_supercell_kpoint, primitive_mask, spectral_function, spectral_weights,
)
from waw.interfaces.vasp.wavecar import HSQDTM, RYTOEV, AUTOA, Wavecar


# --------------------------------------------------------------------------
# A WAVECAR written from scratch, so the reader's record arithmetic is tested
# --------------------------------------------------------------------------

def gcount(cell, encut, kvec):
    """Plane waves inside ENCUT at `kvec`, counted independently of the reader.

    Monotonic enumeration, deliberately NOT the reader's FFT order: the SET is
    order-independent, so this is a real check on the cutoff arithmetic (it is
    what caught the CODATA-vs-VASP hbar^2/2m difference), while saying nothing
    about the order.
    """
    b = np.linalg.inv(cell).T
    n = 6
    rng = np.arange(-n, n + 1)
    G = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    kg = (G + np.asarray(kvec)) @ (2 * np.pi * b)
    return int((HSQDTM * (kg ** 2).sum(1) < encut).sum())


def write_wavecar(path, cell, encut, kpts, eig, coeffs, rtag=45200, nspin=1):
    """Minimal but format-exact WAVECAR. `coeffs[s][k][b]` is a 1-D array."""
    nk, nb = len(kpts), eig.shape[-1]
    itemsize = 8 if rtag == 45200 else 16
    nplw = [len(coeffs[0][k][0]) for k in range(nk)]
    recl = max(3 * 8, 12 * 8, (4 + 3 * nb) * 8, max(nplw) * itemsize)
    recl = int(np.ceil(recl / 8) * 8)
    dt = np.complex64 if rtag == 45200 else np.complex128
    with open(path, "wb") as f:
        def rec(i, buf):
            f.seek(i * recl)
            f.write(buf.tobytes())
        rec(0, np.array([recl, nspin, rtag], dtype=np.float64))
        rec(1, np.concatenate([[nk, nb, encut], np.asarray(cell).ravel()]
                              ).astype(np.float64))
        for s in range(nspin):
            for k in range(nk):
                base = 2 + s * nk * (1 + nb) + k * (1 + nb)
                tri = np.zeros((nb, 3))
                tri[:, 0] = eig[s, k]
                tri[:, 2] = 1.0
                rec(base, np.concatenate([[nplw[k]], kpts[k], tri.ravel()]
                                         ).astype(np.float64))
                for ib in range(nb):
                    rec(base + 1 + ib, coeffs[s][k][ib].astype(dt))
        f.seek((2 + nspin * nk * (1 + nb)) * recl - 1)
        f.write(b"\0")


CELL = np.diag([4.0, 4.0, 4.0])          # a simple cubic 2x2x2 supercell
ENCUT = 120.0
KPTS = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.25, 0.0]])


@pytest.fixture
def wavecar(tmp_path):
    """One band per k, its coefficients a single plane wave of known index."""
    nb = 2
    npw = [gcount(CELL, ENCUT, k) for k in KPTS]
    coeffs = [[[np.zeros(npw[k], complex) for _ in range(nb)]
               for k in range(len(KPTS))]]
    for k in range(len(KPTS)):
        coeffs[0][k][0][0] = 1.0            # band 0: the first plane wave
        coeffs[0][k][1][3] = 0.5            # band 1: the fourth, norm 0.25
    eig = np.tile(np.array([-1.0, 2.0]), (1, len(KPTS), 1))
    p = tmp_path / "WAVECAR"
    write_wavecar(p, CELL, ENCUT, KPTS, eig, coeffs)
    return p


def test_header_and_metadata_round_trip(wavecar):
    with Wavecar(wavecar) as w:
        assert (w.header.nkpts, w.header.nbands, w.header.nspin) == (3, 2, 1)
        assert w.header.encut == pytest.approx(ENCUT)
        assert np.allclose(w.header.cell, CELL)
        assert np.allclose(w.kpoints[0], KPTS)
        assert np.allclose(w.eigenvalues[0, :, 0], -1.0)
        assert np.allclose(w.eigenvalues[0, :, 1], 2.0)
        assert not w.header.noncollinear


def test_plane_wave_count_matches_at_every_kpoint(wavecar):
    """The strongest single check on the reader: the enumeration must agree with
    the stored count at EVERY k, not on average."""
    with Wavecar(wavecar) as w:
        chk = w.check_plane_wave_count()
        assert chk["ok"], chk["mismatches"]
        assert chk["checked"] == 3
        for k in range(3):
            assert len(w.gvectors(k)) == gcount(CELL, ENCUT, KPTS[k])


def test_vasps_own_hbar_squared_over_2m_not_codata():
    """VASP decides which G fall inside ENCUT with RYTOEV*AUTOA^2. The CODATA
    value differs by 1.4e-5 relative, which is enough to move the sphere
    boundary and lose the plane-wave count at some k-points."""
    assert HSQDTM == pytest.approx(RYTOEV * AUTOA ** 2)
    assert HSQDTM != pytest.approx(3.8100340, abs=1e-6)
    assert abs(HSQDTM - 3.8100340) / 3.8100340 < 1e-4      # close, not equal


def test_coefficients_and_pseudo_norm(wavecar):
    with Wavecar(wavecar) as w:
        c0 = w.coefficients(0, 0)
        assert c0.ndim == 1 and c0[0] == pytest.approx(1.0)
        assert w.pseudo_norm(0, 0) == pytest.approx(1.0, abs=1e-6)
        assert w.pseudo_norm(0, 1) == pytest.approx(0.25, abs=1e-6)


def test_double_precision_rtag(tmp_path):
    npw = [gcount(CELL, ENCUT, KPTS[0])]
    coeffs = [[[np.full(npw[0], 0.1 + 0.2j)]]]
    eig = np.zeros((1, 1, 1))
    p = tmp_path / "WAVECAR"
    write_wavecar(p, CELL, ENCUT, KPTS[:1], eig, coeffs, rtag=45210)
    with Wavecar(p) as w:
        assert w.header.coeff_dtype is np.complex128
        assert w.coefficients(0, 0)[0] == pytest.approx(0.1 + 0.2j)


def test_spinor_record_is_split_in_two(tmp_path):
    """With LSORBIT the record holds 2 x npw coefficients. Missing that factor
    is the easiest thing in this format to get wrong."""
    npw = gcount(CELL, ENCUT, KPTS[0])
    c = np.concatenate([np.full(npw, 1.0), np.full(npw, 2.0)]).astype(complex)
    p = tmp_path / "WAVECAR"
    write_wavecar(p, CELL, ENCUT, KPTS[:1], np.zeros((1, 1, 1)), [[[c]]])
    with Wavecar(p) as w:
        assert w.header.noncollinear
        got = w.coefficients(0, 0)
        assert got.shape == (2, npw)
        assert np.allclose(got[0], 1.0) and np.allclose(got[1], 2.0)


def test_mismatched_count_refuses_rather_than_mispairing(tmp_path):
    """A G-set that does not match the stored count must raise: pairing
    coefficients with the wrong G scrambles the wavefunction silently."""
    npw = gcount(CELL, ENCUT, KPTS[0])
    coeffs = [[[np.ones(npw + 7, complex)]]]
    p = tmp_path / "WAVECAR"
    write_wavecar(p, CELL, ENCUT, KPTS[:1], np.zeros((1, 1, 1)), coeffs)
    with pytest.raises(ValueError, match="match neither the plane-wave count"):
        Wavecar(p)


# --------------------------------------------------------------------------
# The projection: kappa_sc = kappa_pc M^T, modulo 1, with time reversal
# --------------------------------------------------------------------------

M2 = np.diag([2, 2, 2])


def test_supercell_kpoint_matching_folds_modulo_one():
    kset = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.25, 0.0]])
    # (0.5, 0, 0) x M^T = (1, 0, 0) == Gamma
    assert match_supercell_kpoint([0.5, 0.0, 0.0], kset, M2) == (0, False)
    # (0.125, 0, 0) -> (0.25, 0, 0)
    assert match_supercell_kpoint([0.125, 0.0, 0.0], kset, M2) == (1, False)


def test_time_reversal_is_used_and_reported():
    kset = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    idx, rev = match_supercell_kpoint([-0.125, 0.0, 0.0], kset, M2)
    assert (idx, rev) == (1, True)


def test_absent_kpoint_raises_instead_of_snapping_to_the_nearest():
    kset = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="neither directly nor by time reversal"):
        match_supercell_kpoint([0.1, 0.0, 0.0], kset, M2)


def test_primitive_mask_selects_one_in_det_M_plane_waves():
    """For a diagonal M and kappa_pc = kappa_sc = 0 the mask keeps the G whose
    components are all even -- exactly 1 of det(M) = 8."""
    rng = np.arange(-4, 5)
    G = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    m = primitive_mask(G, np.zeros(3), np.zeros(3), M2)
    assert np.array_equal(m, (G % 2 == 0).all(axis=1))


def test_primitive_masks_partition_the_plane_waves():
    """Every plane wave belongs to exactly one of the det(M) folded images, so
    the masks must tile the set. If they overlapped, weights would exceed 1."""
    rng = np.arange(-4, 5)
    G = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    images = np.stack(np.meshgrid(*[[0, 0.5]] * 3, indexing="ij"), -1).reshape(-1, 3)
    counts = np.zeros(len(G), dtype=int)
    for kpc in images:
        counts += primitive_mask(G, np.zeros(3), kpc, M2)
    assert np.array_equal(counts, np.ones(len(G), dtype=int))


# --------------------------------------------------------------------------
# End to end on the synthetic file
# --------------------------------------------------------------------------

def test_weights_sum_to_one_over_the_folded_images(wavecar):
    """Completeness: a supercell state's weight distributed over the det(M)
    primitive k-points it can project onto must add up to 1, whatever the
    coefficients are."""
    images = np.stack(np.meshgrid(*[[0, 0.5]] * 3, indexing="ij"), -1).reshape(-1, 3)
    with Wavecar(wavecar) as w:
        sp = spectral_weights(w, images, M2, bands=(0, 2))
        assert sp.weights.shape == (8, 2)
        assert sp.weights.sum(axis=0) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_weights_are_normalised_by_the_pseudo_norm(wavecar):
    """Band 1's coefficients have norm 0.25, not 1. Dividing by 1 instead would
    report the PAW deficiency as if it were unfolding physics."""
    images = np.stack(np.meshgrid(*[[0, 0.5]] * 3, indexing="ij"), -1).reshape(-1, 3)
    with Wavecar(wavecar) as w:
        sp = spectral_weights(w, images, M2, bands=(0, 2))
        # each band sits entirely in ONE image (a single plane wave)
        for b in range(2):
            col = np.sort(sp.weights[:, b])
            assert col[-1] == pytest.approx(1.0, abs=1e-6)
            assert col[:-1] == pytest.approx(np.zeros(7), abs=1e-6)


def test_energy_window_selects_bands_and_records_the_range(wavecar):
    with Wavecar(wavecar) as w:
        sp = spectral_weights(w, [[0.0, 0.0, 0.0]], M2,
                              energy_window=(-2.0, 0.0), fermi=0.0)
        assert sp.band_range == (0, 1)
        assert sp.energies.shape == (1, 1)
        assert sp.energies[0, 0] == pytest.approx(-1.0)
        with pytest.raises(ValueError, match="no band falls inside"):
            spectral_weights(w, [[0.0, 0.0, 0.0]], M2,
                             energy_window=(50.0, 60.0), fermi=0.0)


def test_spectrum_records_which_route_each_kpoint_took(wavecar):
    with Wavecar(wavecar) as w:
        sp = spectral_weights(w, [[0.0, 0.0, 0.0], [-0.125, 0.0, 0.0]], M2,
                              bands=(0, 1))
        assert sp.time_reversed.tolist() == [False, True]
        assert sp.sc_index.tolist() == [0, 1]


def test_spectral_function_conserves_weight():
    """Gaussian broadening redistributes weight in energy; it must not create or
    destroy any."""
    from waw.analysis.unfolding import UnfoldedSpectrum

    sp = UnfoldedSpectrum(kpoints_pc=np.zeros((1, 3)),
                          energies=np.array([[-1.0, 2.0]]),
                          weights=np.array([[0.3, 0.7]]),
                          sc_index=np.zeros(1, int),
                          time_reversed=np.zeros(1, bool), band_range=(0, 2))
    grid = np.linspace(-8, 9, 4001)
    A = spectral_function(sp, grid, sigma=0.1)
    assert A.shape == (1, len(grid))
    assert A.sum() * (grid[1] - grid[0]) == pytest.approx(1.0, abs=1e-6)
