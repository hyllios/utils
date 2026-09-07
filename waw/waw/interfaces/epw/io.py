"""
Readers for EPW's Wannier-representation checkpoint files -- ``crystal.fmt``,
``epwdata.fmt`` and ``prefix.epmatwp`` (formats of EPW v5.8.1, the QE 7.3.1
module on this cluster; pinned against ``EPW/src/io_epw.f90::epw_write``).

Purpose: cross-code MACHINERY tests. Loading EPW's own H(R), dynamical
matrices D(R) and el-ph vertex g(R_e, R_g) into `analysis.elph.alpha2f`
removes every data difference between the codes, so any residual in
lambda/alpha2F is a summation-machinery discrepancy -- the sharpest test
that exists. Validated on Al 12^3 (2026-07-27): with all inputs from EPW's
files, waw reproduces EPW's phonselfen lambda to 7 digits in the
occupation-difference form (0.0003188) and the remaining double-delta
residual tracks the phonon source exactly (al.fc phonons: +0.08%; EPW's own
``rdw``: see the parity scripts). EPW's fine-mesh Fermi level and MP1
``dosef`` are reproduced to 2 microeV / 5 digits via
`analysis.dos.epw_fermi_level` / `analysis.dos.fermi_surface_dos(ngauss=1)`.

Conventions worth pinning (each verified in EPW 5.8.1 source):

* All three Wigner-Seitz R-sets are the ``use_ws = .FALSE.`` (default) ones:
  plain zone-centred WS cells of the coarse k-mesh (electrons) and q-mesh
  (phonons AND el-ph), enumerated lexicographically over n1, n2, n3 --
  bit-identical to `core.hamiltonian._wigner_seitz` including the +-2 image
  search and tie degeneracies (checked: nrr_k/nrr_q/nrr_g match on Al).
* EPW interpolates with ``e^{+2 pi i k.R}/ndegen`` -- the SAME sign
  `core.hamiltonian.operator_k` and `analysis.elph.interpolate_elph_fixed_q`
  use, so ``chw``/``rdw``/``epmatwp`` feed waw with ``+R`` directly.
* ``epmatwp(m, n, irk, imode, irg)`` has m = k+q-side, n = k-side Wannier
  index (``ephwan2bloch`` rotates ``cufkq . epmat . cufkk^dagger`` with
  ``cuf = U^dagger``), matching `alpha2f`'s ``g_R[..., row, col]``.
* Units are Rydberg throughout the files; everything returned here is
  converted to this project's Hartree atomic units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "read_crystal_fmt",
    "read_epwdata",
    "read_epmatwp",
    "epw_phonons",
    "load_epw_model",
]


def _tokens(path):
    with open(path) as fh:
        for line in fh:
            for t in line.split():
                yield t


def _read_complex(tk, n: int) -> np.ndarray:
    """Fortran list-directed complex output: ``(re,im)``, possibly with the
    parts split across whitespace."""
    out = np.empty(n, dtype=np.complex128)
    i = 0
    while i < n:
        s = next(tk)
        if not s.startswith("("):
            continue
        while ")" not in s:
            s += next(tk)
        re_s, im_s = s.strip("()").split(",")
        out[i] = float(re_s) + 1j * float(im_s)
        i += 1
    return out


def read_crystal_fmt(path: str | Path) -> dict:
    """EPW's ``crystal.fmt``. Returns nat, nmodes, nelec, the real lattice
    (rows = a_i, Bohr), reciprocal lattice (rows = b_i, 2*pi/Bohr), volume
    (Bohr^3), alat (Bohr), tau (fractional, (nat, 3)) and amass (amu, the
    raw QE array -- index it with ityp)."""
    tk = _tokens(path)
    nat = int(next(tk))
    nmodes = int(next(tk))
    nelec = float(next(tk))
    at = np.array([float(next(tk)) for _ in range(9)]).reshape(3, 3, order="F")
    bg = np.array([float(next(tk)) for _ in range(9)]).reshape(3, 3, order="F")
    omega = float(next(tk))
    alat = float(next(tk))
    tau_cart = np.array([float(next(tk)) for _ in range(3 * nat)]
                        ).reshape(3, nat, order="F").T * alat        # Bohr
    rest = [t for t in tk]
    real_lattice = at.T * alat                                       # rows a_i
    recip_lattice = bg.T * (2.0 * np.pi / alat)                      # rows b_i
    tau_frac = tau_cart @ np.linalg.inv(real_lattice)
    # amass ... ityp ... noncolin ...: amass is the fixed-size QE array; take
    # the floats up to the nat integers that follow (ityp), robustly: the
    # trailing entries are logicals/w_centers -- only amass/ityp are needed.
    floats = []
    for t in rest:
        try:
            floats.append(float(t))
        except ValueError:
            break
    ityp = None
    for i in range(len(floats) - nat + 1):
        cand = floats[i:i + nat]
        if all(v == int(v) and 1 <= v <= 10 for v in cand):
            ityp = np.array([int(v) - 1 for v in cand])              # 0-based
            amass = np.array(floats[:i])
    if ityp is None:
        raise ValueError(f"read_crystal_fmt: could not locate ityp in {path}")
    # EPW stores amass in RYDBERG mass units: epw_readin.f90:1220 does
    # ``amass = AMU_RY * amass`` before crystal.fmt is ever written (Al reads
    # 24592.13, not 26.98). Convert back to genuine amu here.
    amass = amass[amass > 0] / AMU_RY
    return dict(nat=nat, nmodes=nmodes, nelec=nelec,
                real_lattice=real_lattice, recip_lattice=recip_lattice,
                volume=omega, alat=alat, tau_frac=tau_frac,
                masses_amu=amass, types=ityp)


def read_epwdata(path: str | Path) -> dict:
    """EPW's ``epwdata.fmt``: the coarse Fermi level, array dimensions, and
    the Wannier-representation Hamiltonian ``chw`` (Hartree) and dynamical
    matrix ``rdw`` (Ry^2-mass-weighted force constants, LEFT IN Ry -- see
    `epw_phonons`), shaped ``(nbnd, nbnd, nrr_k)`` / ``(nmodes, nmodes,
    nrr_q)`` in EPW's own R ordering."""
    tk = _tokens(path)
    ef_ry = float(next(tk))
    nbnd, nrr_k, nmodes, nrr_q, nrr_g = (int(next(tk)) for _ in range(5))
    # zstar(3,3,nat)+epsi(3,3): nat is not stored here; infer from nmodes
    nat = nmodes // 3
    _ = [float(next(tk)) for _ in range(9 * nat + 9)]
    chw = _read_complex(tk, nbnd * nbnd * nrr_k).reshape(nbnd, nbnd, nrr_k)
    rdw = _read_complex(tk, nmodes * nmodes * nrr_q).reshape(nmodes, nmodes, nrr_q)
    return dict(ef=ef_ry * 0.5, nbnd=nbnd, nrr_k=nrr_k, nmodes=nmodes,
                nrr_q=nrr_q, nrr_g=nrr_g, chw=chw * 0.5, rdw=rdw)


def read_epmatwp(path: str | Path, nbnd: int, nrr_k: int, nmodes: int,
                 nrr_g: int) -> np.ndarray:
    """EPW's ``prefix.epmatwp`` -- the el-ph vertex in the double Wannier
    representation, direct-access complex128 records ``(nbnd, nbnd, nrr_k,
    nmodes)`` (Fortran order), one per ``irg``.

    Returns `analysis.elph.alpha2f`'s ``g_R`` layout ``(nrr_k, nrr_g,
    nmodes, nbnd_kq_row, nbnd_k_col)`` in Hartree/Bohr; feed it with
    ``R_e = +irvec_k``, ``R_q = +irvec_g`` and the matching degeneracies
    (`core.hamiltonian._wigner_seitz` on the coarse meshes reproduces both
    sets exactly -- assert the counts against `read_epwdata`'s)."""
    ep = np.fromfile(path, dtype=np.complex128)
    expect = nbnd * nbnd * nrr_k * nmodes * nrr_g
    if ep.size != expect:
        raise ValueError(
            f"read_epmatwp: {path} holds {ep.size} complex values, expected "
            f"{expect} = {nbnd}^2 * {nrr_k} * {nmodes} * {nrr_g} -- dims "
            "disagree with epwdata.fmt (different run?).")
    ep = ep.reshape(nrr_g, nmodes, nrr_k, nbnd, nbnd)   # C-view of Fortran dims
    return np.ascontiguousarray(ep.transpose(2, 0, 1, 4, 3)) * 0.5


#: QE's amu -> Rydberg-atomic-unit mass factor (``amu_ry``); rdw carries it.
AMU_RY = 911.4442424792128


def epw_phonons(rdw: np.ndarray, R_q: np.ndarray, degen_q: np.ndarray,
                masses_amu: np.ndarray, types: np.ndarray,
                qpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EPW's own fine-mesh phonons from ``rdw`` (``dynwan2blochf``):
    ``D(q) = sum_R e^{+2 pi i q.R} rdw / ndegen / sqrt(M_a M_b)`` with the
    masses in Ry units, diagonalised -> (omega (nq, nmodes) HARTREE, eigvecs
    (nq, nmodes, nmodes), the same mass-weighted convention as
    `analysis.phonon.interpolate_phonons`). Negative omega^2 comes back as
    negative omega, EPW's own sign convention."""
    import torch

    from ...core.hamiltonian import operator_k

    m = np.repeat(masses_amu[types] * AMU_RY, 3)
    rdw_m = rdw / np.sqrt(m[:, None] * m[None, :])[..., None]
    Dq = operator_k(torch.from_numpy(np.ascontiguousarray(
        rdw_m.transpose(2, 0, 1))), np.asarray(R_q), np.asarray(degen_q),
        np.asarray(qpts, dtype=np.float64))
    Dq = Dq.numpy()
    Dq = 0.5 * (Dq + np.conj(Dq.transpose(0, 2, 1)))
    w2, ev = np.linalg.eigh(Dq)
    omega = np.sqrt(np.abs(w2)) * np.sign(w2) * 0.5    # Ry -> Ha
    return omega, ev


def load_epw_model(directory: str | Path, k_grid: tuple[int, int, int],
                   q_grid: tuple[int, int, int]) -> dict:
    """One-call assembly of everything `analysis.elph.alpha2f` needs from an
    EPW run directory (``epwwrite = .true.`` output): ``hr`` (HamiltonianR
    from ``chw``), ``g_R``/``R_e``/``degen_e``/``R_q``/``degen_q`` (from
    ``epmatwp``), a ``phonon_fn(qpts) -> (omega, eigvecs)`` closure over
    ``rdw``, plus the ``crystal`` dict. ``k_grid``/``q_grid`` are the COARSE
    meshes of the EPW run (its ``nk1..``/``nq1..``)."""
    import torch

    from ...core.hamiltonian import HamiltonianR, _wigner_seitz

    directory = Path(directory)
    crystal = read_crystal_fmt(directory / "crystal.fmt")
    data = read_epwdata(directory / "epwdata.fmt")
    R_k, deg_k = _wigner_seitz(tuple(k_grid), crystal["real_lattice"])
    R_q, deg_q = _wigner_seitz(tuple(q_grid), crystal["real_lattice"])
    if len(R_k) != data["nrr_k"] or len(R_q) != data["nrr_q"] \
            or len(R_q) != data["nrr_g"]:
        raise ValueError(
            f"load_epw_model: Wigner-Seitz counts (k: {len(R_k)} vs "
            f"{data['nrr_k']}, q: {len(R_q)} vs {data['nrr_q']}/"
            f"{data['nrr_g']}) disagree with epwdata.fmt -- wrong coarse "
            "grids, or a use_ws=.true. run (not supported).")
    epmatwp = next(directory.glob("*.epmatwp"))
    g_R = read_epmatwp(epmatwp, data["nbnd"], data["nrr_k"],
                       data["nmodes"], data["nrr_g"])
    hr = HamiltonianR(
        H_R=torch.from_numpy(np.ascontiguousarray(data["chw"].transpose(2, 0, 1))),
        R_vectors=R_k, degen=deg_k, nw=data["nbnd"])

    def phonon_fn(qpts):
        return epw_phonons(data["rdw"], R_q, deg_q, crystal["masses_amu"],
                           crystal["types"], qpts)

    return dict(hr=hr, g_R=g_R, R_e=R_k, degen_e=deg_k, R_q=R_q,
                degen_q=deg_q, phonon_fn=phonon_fn, crystal=crystal,
                ef_coarse=data["ef"], nmodes=data["nmodes"])
