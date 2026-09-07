"""
Statistical-mechanics and numerical-smearing building blocks shared
across `analysis/`: Fermi-Dirac occupation (and its energy derivative),
Bose-Einstein occupation, and a Gaussian delta-function kernel.

Pure numpy, atomic units throughout (energies in Hartree, matching the
rest of `core`/`analysis`) -- callers convert to/from SI via
`waw.units.to_si_units`.
"""

from __future__ import annotations

import numpy as np

_MAX_EXP = 36.0   # overflow guard, matches wannier90's MinusFermiDerivative


def fermi_dirac(E: np.ndarray, mu: float, kT: float) -> np.ndarray:
    """f(E) = 1 / (exp((E-mu)/kT) + 1). E, mu, kT: same energy unit (Hartree
    elsewhere here). ``kT <= 0`` gives the exact zero-temperature step
    function (1 below mu, 0 above, 1/2 at mu) rather than a 0/0."""
    E = np.asarray(E)
    if kT <= 0.0:
        return np.where(E < mu, 1.0, np.where(E > mu, 0.0, 0.5))
    x = np.clip((E - mu) / kT, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def minus_fermi_deriv(E: np.ndarray, mu: float, kT: float) -> np.ndarray:
    """-df/dE (>= 0, peaked at E=mu), in 1/[energy unit]."""
    E = np.asarray(E)
    x = np.clip((E - mu) / kT, -_MAX_EXP, _MAX_EXP)
    ex = np.exp(x)
    out = (1.0 / kT) * ex / (ex + 1.0) ** 2
    out = np.where(np.abs((E - mu) / kT) > _MAX_EXP, 0.0, out)
    return out


def dfermi_dE(E: np.ndarray, mu: float, kT: float) -> np.ndarray:
    """df/dE (<= 0, peaked at E=mu) -- the signed derivative, i.e. -minus_fermi_deriv."""
    return -minus_fermi_deriv(E, mu, kT)


def bose_einstein(omega: np.ndarray, kT: float) -> np.ndarray:
    """n(omega) = 1 / (exp(omega/kT) - 1). omega, kT: same energy unit,
    omega > 0 expected. ``kT <= 0`` gives exactly zero -- no thermal bosons at
    T = 0, which is a limit rather than the 0/0 a bare division would hit.
    Mirrors `fermi_dirac`'s treatment of the same case."""
    omega = np.asarray(omega)
    if kT <= 0.0:
        return np.zeros_like(omega, dtype=np.float64)
    x = np.clip(omega / kT, 1e-12, _MAX_EXP)
    return 1.0 / np.expm1(x)


def gaussian_smearing(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian delta-function approximation: exp(-x^2/(2*sigma^2)) / (sigma*sqrt(2*pi))."""
    return np.exp(-0.5 * (np.asarray(x) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


#: QE/EPW's ``degauss`` is the width of QE's ``w0gauss(x, 0)`` =
#: exp(-x^2)/sqrt(pi), i.e. delta(e) ~ exp(-(e/degauss)^2)/(degauss sqrt(pi)).
#: `gaussian_smearing` above uses the statistician's normalization
#: exp(-e^2/2 sigma^2)/(sigma sqrt(2 pi)). Both are normalized deltas, but
#: their widths differ by sqrt(2):
#:
#:     sigma = degauss / sqrt(2)
#:
#: makes the two functions IDENTICAL pointwise. Quoting a QE/EPW ``degauss``
#: straight into a ``sigma`` argument silently broadens every delta by 41% --
#: use `epw_degauss_to_sigma`.
EPW_DEGAUSS_TO_SIGMA = 1.0 / np.sqrt(2.0)


def epw_degauss_to_sigma(degauss: float) -> float:
    """QE/EPW ``degauss`` (``degaussw``/``degaussq``) -> this project's
    Gaussian ``sigma``. See `EPW_DEGAUSS_TO_SIGMA`."""
    return float(degauss) * EPW_DEGAUSS_TO_SIGMA


def sigma_to_epw_degauss(sigma: float) -> float:
    """Inverse of `epw_degauss_to_sigma`."""
    return float(sigma) / EPW_DEGAUSS_TO_SIGMA


def w0gauss(x: np.ndarray, ngauss: int = 0) -> np.ndarray:
    """
    QE's ``w0gauss(x, n)`` (``Modules/w0gauss.f90``), the delta-function
    approximant QE and EPW smear with -- ``ngauss=0`` is the plain Gaussian
    ``exp(-x^2)/sqrt(pi)``, ``ngauss=1`` is Methfessel-Paxton order 1,
    ``exp(-x^2)(3/2 - x^2)/sqrt(pi)`` (which goes NEGATIVE for |x| > sqrt(3/2)).

    Note ``x`` is already ``(e - ef)/degauss``; divide the result by
    ``degauss`` for a normalized delta. Pointwise identical to
    `gaussian_smearing` at ``ngauss=0`` when ``sigma = degauss/sqrt(2)``.
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.exp(-np.minimum(x ** 2, 200.0)) / np.sqrt(np.pi)
    if ngauss == 0:
        return g
    if ngauss == 1:
        return g * (1.5 - x ** 2)
    raise ValueError(f"w0gauss: only ngauss 0 and 1 are implemented, got {ngauss}")


def wgauss(x: np.ndarray, ngauss: int = 0) -> np.ndarray:
    """
    QE's ``wgauss(x, n)`` (``Modules/wgauss.f90``), the smeared step /
    occupation function complementary to `w0gauss` -- ``x`` is
    ``(eF - e)/degauss``:

      * ``ngauss=0``: plain Gaussian, ``erfc(-x)/2``;
      * ``ngauss=1``: Methfessel-Paxton order 1,
        ``erfc(-x)/2 + x exp(-x^2)/(2 sqrt(pi))`` (NOT monotonic in x --
        it overshoots 1 and undershoots 0 around |x| ~ 1, which is why
        QE's own ``efermig`` needs the Newton dance replicated in
        `waw.analysis.dos.epw_fermi_level`);
      * ``ngauss=-99``: Fermi-Dirac, ``1/(1 + exp(-x))``.
    """
    from scipy.special import erfc

    x = np.asarray(x, dtype=np.float64)
    if ngauss == -99:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -200.0, 200.0)))
    w = 0.5 * erfc(-x)
    if ngauss == 0:
        return w
    if ngauss == 1:
        return w + x * np.exp(-np.minimum(x ** 2, 200.0)) / (2.0 * np.sqrt(np.pi))
    raise ValueError(f"wgauss: only ngauss 0, 1 and -99 are implemented, got {ngauss}")
