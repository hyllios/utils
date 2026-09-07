"""
The collinear-spin path of `interfaces.quantum_espresso.dvscf_io.read_dvscf`.

QE writes dvscf with record length ``lrdrho = 2*nnr*nspin_mag`` REAL(8) words --
i.e. ``nnr*nspin_mag`` complex numbers, spin blocks contiguous WITHIN a record.
So on an ``nspin=2`` ground state the record stride doubles, and reading with the
non-magnetic stride fails SILENTLY: record 0 spin 0 still lands correctly and
every later mode is misaligned. That is the failure this module guards against,
so the offsets are tested here directly on a synthesised file rather than
inferred from a calculation.

VALIDATED AGAINST QE TOO, with a zero-moment control that needs no magnetic
reference data: non-magnetic Al run at ``nspin=2`` with zero starting
magnetization must give dV_up == dV_dn == dV(nspin=1). Measured on a 4^3 /
30 Ry Al run at q = Gamma and q = (0,1/2,1/2), 20^3 FFT grid:

    quantity                       q = Gamma     q = (0,1/2,1/2)
    up vs dn                        5.6e-06         5.5e-05
    up vs nspin=1                   5.0e-06         5.2e-04
    dn vs nspin=1                   2.2e-06         4.6e-04
    nspin-UNAWARE read, per mode    0.71/1.67/0.94  0.56/0.00046/1.02

i.e. the channels agree to SCF-convergence level, while the nspin-unaware read
is 100% wrong on the modes it misaligns (and, tellingly, exactly right on one of
them -- which is why this cannot be caught by spot-checking a single mode).
"""

import numpy as np
import pytest

from waw.interfaces.quantum_espresso.dvscf_io import (
    BYTES_PER_COMPLEX128,
    _read_dvscf_records,
)

GRID = (3, 4, 5)
NPTS = GRID[0] * GRID[1] * GRID[2]


def _synth(tmp_path, n_records, nspin):
    """A dvscf-layout file whose every element encodes (record, spin, point):
    value = record + 1000*spin + 1j*point. Any stride or offset slip therefore
    shows up as a wrong record or spin label, not as a subtle numerical drift."""
    buf = np.empty(n_records * nspin * NPTS, dtype=np.complex128)
    at = 0
    for r in range(n_records):
        for s in range(nspin):
            buf[at:at + NPTS] = (r + 1000 * s) + 1j * np.arange(NPTS)
            at += NPTS
    p = tmp_path / "dvscf1"
    p.write_bytes(buf.tobytes())
    return p


def _label(block):
    """Recover (record + 1000*spin) from a returned grid block."""
    flat = block.reshape(-1, order="F")
    assert np.allclose(flat.imag, np.arange(NPTS)), "grid ordering scrambled"
    assert np.allclose(flat.real, flat.real[0]), "block mixes records"
    return float(flat.real[0])


class TestRecordOffsets:
    def test_nonmagnetic_layout_unchanged(self, tmp_path):
        p = _synth(tmp_path, n_records=4, nspin=1)
        got = _read_dvscf_records(p, 0, 4, GRID)
        assert got.shape == (4, *GRID)
        assert [_label(got[r]) for r in range(4)] == [0.0, 1.0, 2.0, 3.0]

    @pytest.mark.parametrize("ispin,base", [(0, 0.0), (1, 1000.0)])
    def test_spin_polarised_layout_picks_the_right_block(self, tmp_path, ispin, base):
        """Each record holds [spin0 | spin1]; both channels must come back with
        the record index intact."""
        p = _synth(tmp_path, n_records=4, nspin=2)
        got = _read_dvscf_records(p, 0, 4, GRID, nspin=2, ispin=ispin)
        assert [_label(got[r]) for r in range(4)] == [base + r for r in range(4)]

    def test_an_offset_irrep_still_lands_correctly(self, tmp_path):
        """Irreps after the first start at a running mode offset -- the place a
        stride error does its damage."""
        p = _synth(tmp_path, n_records=6, nspin=2)
        got = _read_dvscf_records(p, 2, 3, GRID, nspin=2, ispin=1)
        assert [_label(got[r]) for r in range(3)] == [1002.0, 1003.0, 1004.0]

    def test_reading_a_magnetic_file_with_the_nonmagnetic_stride_is_wrong(self, tmp_path):
        """The silent failure, made explicit: record 0 is right, the rest are
        not. This is why nspin must be stated and cannot be sniffed."""
        p = _synth(tmp_path, n_records=4, nspin=2)
        naive = _read_dvscf_records(p, 0, 4, GRID)              # nspin defaulted to 1
        right = _read_dvscf_records(p, 0, 4, GRID, nspin=2, ispin=0)
        assert _label(naive[0]) == _label(right[0]) == 0.0      # mode 0 survives
        assert [_label(naive[r]) for r in (1, 2, 3)] == [1000.0, 1.0, 1001.0]
        assert [_label(right[r]) for r in (1, 2, 3)] == [1.0, 2.0, 3.0]


class TestGuards:
    def test_ispin_must_be_inside_nspin(self, tmp_path):
        p = _synth(tmp_path, 2, 2)
        with pytest.raises(ValueError, match="outside nspin"):
            _read_dvscf_records(p, 0, 1, GRID, nspin=2, ispin=2)
        with pytest.raises(ValueError, match="outside nspin"):
            _read_dvscf_records(p, 0, 1, GRID, nspin=1, ispin=1)

    def test_short_file_names_the_factor(self, tmp_path):
        """Asking for nspin=2 on a non-magnetic file is exactly a factor-2
        shortfall, and the message says so -- that is the likeliest mistake."""
        p = _synth(tmp_path, n_records=4, nspin=1)
        with pytest.raises(ValueError, match="nspin is wrong"):
            _read_dvscf_records(p, 0, 4, GRID, nspin=2, ispin=0)

    def test_byte_accounting_matches_qes_record_length(self, tmp_path):
        """lrdrho = 2*nnr*nspin_mag REAL(8) = nnr*nspin_mag complex128."""
        for nspin in (1, 2):
            p = _synth(tmp_path, n_records=3, nspin=nspin)
            assert p.stat().st_size == 3 * nspin * NPTS * BYTES_PER_COMPLEX128
