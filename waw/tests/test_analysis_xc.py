"""
Tests for `analysis.xc` -- the GGA (PBE) exchange-correlation potential and its
linear-response kernel action, used by the el-ph NLCC term.

The central check: `xc_kernel_action` must be the exact finite-difference
derivative of `xc_potential_gga` (a QE-independent self-consistency test of the
whole gradient-corrected kernel, including the FFT gradient/divergence).
"""

import numpy as np
import pytest

pytest.importorskip("pylibxc")
from waw.analysis import xc


def _smooth_density(grid, recip_lattice, seed=0):
    """A positive, smooth, periodic test density with real structure (a few
    low-G Fourier components) so the GGA gradient terms are exercised."""
    rng = np.random.default_rng(seed)
    rho = np.full(grid, 0.30)
    ints = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in grid]
    G = np.stack(np.meshgrid(*ints, indexing="ij"), axis=-1)
    for _ in range(6):
        gv = rng.integers(-1, 2, size=3)
        if not gv.any():
            continue
        phase = 2 * np.pi * (G @ gv) / np.array(grid).max()   # smooth wave over the cell
        rho += 0.05 * rng.uniform(-1, 1) * np.cos(phase)
    return np.clip(rho, 0.02, None)


class TestXCKernel:
    def test_kernel_is_fd_derivative_of_potential(self):
        grid = (12, 12, 12)
        B = 2 * np.pi * np.linalg.inv(np.eye(3) * 7.0).T   # cubic 7 Bohr cell
        rho = _smooth_density(grid, B, seed=3)
        rng = np.random.default_rng(11)
        drho = np.zeros(grid)
        ints = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in grid]
        G = np.stack(np.meshgrid(*ints, indexing="ij"), axis=-1)
        for _ in range(4):                                  # smooth real perturbation
            gv = rng.integers(-1, 2, size=3)
            if not gv.any():
                continue
            drho += 0.02 * rng.uniform(-1, 1) * np.cos(2 * np.pi * (G @ gv) / max(grid))

        dv_analytic = xc.xc_kernel_action(rho, drho.astype(complex), B).real
        eps = 1e-5
        vp = xc.xc_potential_gga(rho + eps * drho, B)
        vm = xc.xc_potential_gga(rho - eps * drho, B)
        dv_fd = (vp - vm) / (2 * eps)

        # compare where the density is well away from the floor
        num = np.linalg.norm(dv_analytic - dv_fd)
        den = np.linalg.norm(dv_fd)
        assert num / den < 1e-4, f"kernel vs FD mismatch: {num/den:.2e}"

    def test_potential_matches_libxc_lda_limit(self):
        """With a CONSTANT density (zero gradient) PBE's gradient terms vanish
        and V_xc reduces to the plain vrho from libxc."""
        grid = (8, 8, 8)
        B = 2 * np.pi * np.linalg.inv(np.eye(3) * 6.0).T
        rho = np.full(grid, 0.25)
        V = xc.xc_potential_gga(rho, B)
        vrho = xc.pbe_derivatives(rho.ravel(), np.zeros(rho.size))["vrho"]
        assert np.allclose(V.ravel(), vrho, atol=1e-9)

    def test_kernel_hermiticity_real_for_real_perturbation(self):
        """A real q=0 perturbation on a real density gives a real dV_xc."""
        grid = (8, 8, 8)
        B = 2 * np.pi * np.linalg.inv(np.eye(3) * 6.0).T
        rho = _smooth_density(grid, B, seed=1)
        drho = _smooth_density(grid, B, seed=2) - 0.30
        dv = xc.xc_kernel_action(rho, drho.astype(complex), B)
        assert np.max(np.abs(dv.imag)) < 1e-9
