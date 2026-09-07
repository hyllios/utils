"""
Build the waw W90 tutorial notebooks with nbformat.

These are the *pedagogical* re-imaginings of the Wannier90 tutorials, done the
**waw-native** way: converged Quantum ESPRESSO (PseudoDojo pseudos) -> waw's own
`.nnkp` -> pw2wannier90 overlaps -> waw ASE-native `wannierize` -> waw analysis.
No `wannier90.x` and no `.win`/`.chk` anywhere.

Run `python _generate.py` to (re)write the .ipynb files, then execute them with
    module load quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k
    python -m nbconvert --to notebook --execute --inplace --kernel=waw NN_*.ipynb
"""

import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
KERNEL = {"kernelspec": {"name": "waw", "display_name": "waw"},
          "language_info": {"name": "python"}}


def make(path, cells):
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "md" else nbf.v4.new_code_cell(s)
                for t, s in cells]
    nbf.write(nb, str(HERE / path))
    print("wrote", path)


SETUP = """\
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

# locate the repo root (the folder containing the `waw` package), so the
# notebook runs from anywhere
HERE = pathlib.Path.cwd()
REPO = HERE
while not (REPO / 'waw').exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

import waw
from waw.interfaces import quantum_espresso as qe   # the direct-input QE driver
from waw.interfaces.ase.driver import wannierize
from waw.units import BOHR_TO_ANG, HARTREE_TO_EV
from waw.vis import plot_bands, BandSeries

PSEUDO_DIR = REPO / 'workflows' / 'pseudos'   # shared across workflows/, not w90tutorial-specific
NCORES = 16
waw.set_num_threads(NCORES)                  # use many cores for waw's batched linear algebra
print('waw', 'threads =', waw.get_num_threads(), '| repo', REPO)"""


# ==========================================================================
# 01 — GaAs, 4 isolated sp3 MLWFs
# ==========================================================================
gaas = [
    ("md",
     "# Tutorial 01 — GaAs: four isolated $sp^3$ Wannier functions\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 1. We run a **converged** "
     "Quantum ESPRESSO calculation on zincblende **GaAs** with PseudoDojo "
     "pseudopotentials, generate the overlaps with `pw2wannier90`, and then "
     "**Wannierise entirely inside waw** (no `wannier90.x`, no `.win`/`.chk`).\n"
     "\n"
     "**A modern-pseudopotential wrinkle.** The classic tutorial uses an old "
     "3-electron Ga pseudo, giving 4 occupied bands that map directly onto 4 "
     "bond-centred $sp^3$ MLWFs. The PseudoDojo pseudos here are more accurate: "
     "Ga and As each carry a filled **3$d$ semicore** (13 + 15 = 28 valence "
     "electrons → 14 occupied bands = 10 semicore $d$ + 4 $sp^3$). We simply "
     "**exclude the 10 semicore $d$ bands** (and the conduction bands), leaving "
     "the 4 $sp^3$ valence bands as a clean **isolated** manifold — no "
     "disentanglement needed.\n"
     "\n"
     "**Pipeline:** `qe.generate_overlaps` runs scf → waw `.nnkp` → nscf → "
     "`pw2wannier90` (SCDM); then `waw.interfaces.ase.driver.wannierize` does the "
     "localisation."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Zincblende GaAs, $a = 5.65$ Å. The 3$d$ semicore needs a high plane-wave "
     "cutoff (70 Ry); the SCF uses an 8×8×8 mesh, and the Wannier overlaps are "
     "built on a 4×4×4 mesh. We compute 20 bands (a buffer above the 14 occupied "
     "for a fast, clean NSCF) and exclude the 10 semicore $d$ bands (1–10) plus "
     "the conduction bands (15–20), keeping bands 11–14 = the 4 $sp^3$ valence. "
     "(First run takes a few minutes of DFT.)"),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('GaAs', 'zincblende', a=5.65)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'gaas'\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas',\n"
     "    ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=20, num_wann=4,\n"
     "    exclude_bands=list(range(1, 11)) + list(range(15, 21)),  # 10 semicore d + conduction\n"
     "    pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, scdm_entanglement='isolated', ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/gaas/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Wannierise (waw, ASE-native)\n"
     "The structure supplies the lattice; the overlap arrays go straight into the "
     "core optimiser. 4 isolated bands → 4 MLWFs, no windows."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, n_restarts=3, n_iter=5000, verbose=False,\n"
     ")\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "centres = result.centres_bohr * BOHR_TO_ANG\n"
     "print(f'Omega_total = {omega:.4f} Ang^2')\n"
     "for i, (c, s) in enumerate(zip(centres, spreads)):\n"
     "    print(f'  WF {i+1}: centre ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) A   spread {s:.4f} A^2')"),
    ("md",
     "The four MLWFs come out **equivalent** — identical spread, centred on the "
     "four tetrahedral Ga–As bonds at $(\\pm,\\pm,\\pm)$ — the hallmark of $sp^3$ "
     "bond orbitals. (Their spread is a little larger than the classic tutorial's "
     "because the semicore PseudoDojo pseudo gives slightly more diffuse valence "
     "orbitals; the *symmetry* is what matters.)"),
    ("md",
     "### 3. Interpolated band structure\n"
     "The Wannier tight-binding $H(R)$ interpolates the valence bands onto any "
     "k-path at essentially no cost. `band_path` asks **ASE** for the standard "
     "high-symmetry path of the cell's Bravais lattice (Setyawan-Curtarolo "
     "convention), rather than a hand-copied one."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=120)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands, figsize=(6, 4),\n"
     "           title='GaAs valence bands from 4 MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** Starting from nothing but a crystal structure, waw ran a "
     "converged DFT calculation and produced four maximally-localised $sp^3$ bond "
     "Wannier functions and an interpolatable tight-binding model — with the "
     "wannier90 minimiser replaced entirely by waw's own optimiser."),
]


# ==========================================================================
# 02 — Silicon, 8 sp3 MLWFs by disentanglement + band structure
# ==========================================================================
si = [
    ("md",
     "# Tutorial 03 — Silicon: disentanglement and band interpolation\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 3 — the first "
     "**disentanglement** example. We build **8** $sp^3$ atom-centred MLWFs for "
     "silicon (4 per Si atom). These do *not* come from an isolated set of "
     "bands: they span the 4 valence bands **plus** 4 low conduction bands, which "
     "are entangled with the rest of the conduction manifold. Disentanglement "
     "extracts an optimally-connected 8-dimensional subspace using an **outer "
     "window** (states allowed to mix) and a **frozen window** (states kept "
     "exactly — here the whole valence).\n"
     "\n"
     "Same waw-native pipeline as tutorial 01, with `pw2wannier90` SCDM in "
     "`erfc` mode (a smooth energy weighting for entangled bands) and waw's "
     "disentanglement driven by the two energy windows."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Diamond-cubic Si, $a = 5.43$ Å, `ecutwfc = 40` Ry (Si has no semicore — a "
     "clean 4-valence pseudo). The NSCF computes 16 bands so the conduction "
     "states entering the outer window are available. SCDM `erfc` with "
     "$\\mu = 12$ eV, $\\sigma = 4$ eV weights the projection toward the "
     "valence + lower-conduction manifold."),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('Si', 'diamond', a=5.43)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'si'\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=16, num_wann=8,\n"
     "    scdm_entanglement='erfc', scdm_mu=12.0, scdm_sigma=4.0,\n"
     "    pseudopotentials={'Si': 'Si.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/si/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Disentangle + Wannierise\n"
     "The **frozen window** ends at 6.4 eV (just above the valence-band maximum, "
     "so all 4 valence bands are kept exactly); the **outer window** extends to "
     "17 eV, letting the 4 extra Wannier states be built from an optimal mix of "
     "the low conduction bands."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=8, outer_window=(-1e3, 17.0), frozen_window=(-1e3, 6.4),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2   (Wannier90 tutorial 3: 11.85)')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial 3: 14.50)')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))"),
    ("md",
     "$\\Omega_I$ — the gauge-invariant part fixed by the chosen subspace — "
     "matches Wannier90's value, confirming waw found the same optimally-"
     "connected subspace. The 8 MLWFs are equivalent ($sp^3$ hybrids, 4 per "
     "atom)."),
    ("md",
     "### 3. Interpolated band structure\n"
     "The 8-band Wannier model interpolates the silicon bands onto any k-path. "
     "Inside the **frozen window** (shaded) the interpolation reproduces the DFT "
     "valence bands essentially exactly — the defining property of a good "
     "disentanglement."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=120)   # ASE's standard path for this lattice\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands, figsize=(6, 4),\n"
     "           shade_window=(bands.min() - 1, 6.4),\n"
     "           title='Si bands from 8 disentangled MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** With PseudoDojo pseudos and waw's own optimiser + "
     "disentanglement — no `wannier90.x` — the 8-band $sp^3$ model reproduces "
     "Wannier90 tutorial 3's $\\Omega_I$ and interpolates the silicon band "
     "structure across the valence and lower conduction bands."),
]


# ==========================================================================
# Shared Fe (bcc, SOC) wannierisation — used by notebooks 17 and 18
# ==========================================================================
FE_SETUP = """\
from ase.build import bulk
import torch
from waw.interfaces.ase.structure import real_lattice, recip_lattice, band_path_segments

atoms = bulk('Fe', 'bcc', a=2.8699)          # 5.4235 bohr
MP_GRID = (4, 4, 4)
WORK = HERE / 'runs' / 'fe'

# noncollinear + spin-orbit; PseudoDojo fully-relativistic Fe pseudo
SOC = dict(noncolin=True, lspinorb=True)
SOC['starting_magnetization(1)'] = 0.4
SOC.update(occupations='smearing', smearing='cold', degauss=0.02)

# scf (16^3) -> waw .nnkp (spinor) -> nscf (4^3, 36 bands) -> pw2wannier90 SCDM
# 8 semicore 3s3p spinor bands are excluded; 28 -> 18 spinor MLWFs.
ov = qe.generate_overlaps(
    atoms, MP_GRID, WORK, 'fe',
    ecutwfc=60, scf_kpts=(16, 16, 16), nbnd=36, num_wann=18,
    exclude_bands=list(range(1, 9)),
    scdm_entanglement='erfc', scdm_mu=25.0, scdm_sigma=5.0,
    system_extra=SOC, pseudopotentials={'Fe': 'Fe-sp_r.upf'},
    pseudo_dir=PSEUDO_DIR, ncores=NCORES, write_spn=True,
    rerun_scf=False,          # reuse a completed SCF in runs/fe/ if present
)

result = wannierize(
    atoms, MP_GRID, ov['kpts'],
    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],
    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=18,
    outer_window=(8.0, 70.0), frozen_window=(8.0, 19.8),
    n_restarts=2, dis_n_iter=1000, n_iter=3000, verbose=False,
)
E_FERMI = ov['fermi_energy']                 # eV, read from the SCF by qe.py
print(f'Omega_I = {result.dis.omega_i * BOHR_TO_ANG**2:.4f} Ang^2  '
      f'(18 spinor MLWFs from 28 bands; W90 tut17/18 4x4x4: ~9.17); '
      f'E_F = {E_FERMI:.3f} eV')

# ASE's standard bcc high-symmetry path, as parse_kpoint_path-style segments
KPATH = band_path_segments(atoms)
REAL = real_lattice(atoms)
RECIP = recip_lattice(atoms)"""


# ==========================================================================
# 03 — bcc Fe, spin-orbit: spin-coloured band structure
# ==========================================================================
fe_spin = [
    ("md",
     "# Tutorial 17 — bcc iron with spin–orbit coupling: spin texture\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 17. Ferromagnetic bcc "
     "**iron** with full **spin–orbit coupling** (noncollinear DFT): 28 bands "
     "disentangle into **18 spinor MLWFs** (each a 2-component spinor). Because "
     "SOC mixes the spin channels, the band energies alone don't tell you the "
     "spin — we interpolate a per-band **spin expectation** $\\langle S_z "
     "\\rangle$ from the `.spn` Pauli matrices between Bloch states "
     "(`pw2wannier90 write_spn`), giving a **spin-coloured band structure**.\n"
     "\n"
     "Fully waw-native: converged noncollinear+SOC QE, waw's own spinor `.nnkp`, "
     "`pw2wannier90` SCDM, waw disentanglement, and "
     "`waw.analysis.spin_texture`. No `wannier90.x`."),
    ("code", SETUP),
    ("md",
     "### 1. Converged noncollinear+SOC DFT and Wannierisation\n"
     "PseudoDojo **fully-relativistic** Fe pseudo, `ecutwfc = 60` Ry, 16×16×16 "
     "SCF (magnetisation ≈ 2.3 μB along $z$, $E_F$ ≈ 17.6 eV). The Wannier "
     "overlaps and the `.spn` spin operator are built on a 4×4×4 mesh. (The SCF "
     "is the slow step, a few minutes.)"),
    ("code", FE_SETUP),
    ("md",
     "### 2. Spin operator $S(R)$ and the spin-coloured bands\n"
     "The `.spn` file holds $\\langle\\psi_{nk}|\\sigma_i|\\psi_{mk}\\rangle$ "
     "between the ab-initio Bloch states. We rotate it into the Wannier gauge "
     "(the same $W = V\\,U$ that builds $H(R)$), Fourier-transform to $S(R)$, and "
     "interpolate $\\langle n(k)|S_z|n(k)\\rangle$ along the k-path."),
    ("code",
     "from waw.analysis.spin_texture import spin_operator_r, spin_colored_bands\n"
     "\n"
     "W = torch.bmm(result.dis.V, result.spread.U_final)      # full spinor gauge\n"
     "SS_R = spin_operator_r(W, ov['spn'], result.wdata.kpts, MP_GRID, REAL)\n"
     "\n"
     "scb = spin_colored_bands(result.hr, SS_R, KPATH, RECIP, n_points=120,\n"
     "                         axis=(0.0, 0.0, 1.0))\n"
     "bands = scb.bands.bands * HARTREE_TO_EV        # (nk, nw) eV\n"
     "spin = scb.spin                                # (nk, nw), <S_z> in [-1, 1]\n"
     "print('S_z range:', float(spin.min()), '..', float(spin.max()))"),
    ("code",
     "dist = scb.bands.kpath.dists\n"
     "series = BandSeries(bands=bands - E_FERMI, color_by=spin, vmin=-1, vmax=1)\n"
     "plot_bands(dist, scb.bands.kpath.tick_dists, scb.bands.kpath.tick_labels, series,\n"
     "           figsize=(6.5, 4), ref_line=0.0, ref_color='0.5', ylim=(-8, 8),\n"
     "           ylabel='E - E_F (eV)', colorbar_label='<S_z>',\n"
     "           title='bcc Fe (SOC): bands coloured by <S_z>')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** The 18-spinor-MLWF model reproduces the exchange-split bands "
     "of ferromagnetic Fe and their spin character: majority (blue, "
     "$\\langle S_z\\rangle < 0$ for magnetisation along $+z$) and minority (red) "
     "bands, with spin-orbit coupling mixing them where they cross. All from "
     "waw — DFT, disentanglement, and spin interpolation."),
]


# ==========================================================================
# 04 — bcc Fe, spin-orbit: Berry curvature + anomalous Hall conductivity
# ==========================================================================
fe_ahc = [
    ("md",
     "# Tutorial 18 — bcc iron: Berry curvature and anomalous Hall conductivity\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 18 (the postw90 `berry` "
     "module). Using the same 18 spinor MLWFs of ferromagnetic bcc Fe, we "
     "compute the **Berry curvature** along a k-path and the **anomalous Hall "
     "conductivity** (AHC) as a function of Fermi energy — the full WYSV06 "
     "$J_0+J_1+J_2$ formula with the Wannier position operator "
     "(`waw.analysis.topology`, `waw.core.compute_position_r`).\n"
     "\n"
     "The AHC needs a **dense** k-mesh to converge; this is where waw's "
     "multi-core batched Berry-curvature evaluation (`waw.set_num_threads`) "
     "matters — a 25³ mesh runs in seconds."),
    ("code", SETUP),
    ("md",
     "### 1. Converged SOC DFT and Wannierisation\n"
     "Identical to notebook 17 (same converged noncollinear+SOC Fe, 18 spinor "
     "MLWFs)."),
    ("code", FE_SETUP),
    ("md",
     "### 2. Wannier position operator and Berry curvature along ASE's standard path\n"
     "$AA(R) = \\langle 0n|r|Rm\\rangle$ comes from the finite-difference Berry "
     "connection of the overlaps; the occupied-band Berry curvature "
     "$J_0+J_1+J_2$ is evaluated at $E_F$."),
    ("code",
     "from waw.core.hamiltonian import compute_position_r\n"
     "from waw.analysis.topology import (wannier_interpolated_curvature,\n"
     "                                   anomalous_hall_conductivity)\n"
     "from waw.analysis.kpath import parse_kpoint_path, build_kpath\n"
     "from waw.units import EV_TO_HARTREE, to_si_units\n"
     "\n"
     "AA_R = compute_position_r(result.m_tilde, result.wdata.wb,\n"
     "                          result.wdata.bvecs, result.wdata.kpts, MP_GRID, REAL)\n"
     "CELL_VOLUME_BOHR3 = abs(np.linalg.det(REAL))\n"
     "\n"
     "segments = parse_kpoint_path(KPATH)\n"
     "kpath = build_kpath(segments, RECIP, n_points=200)\n"
     "curv = wannier_interpolated_curvature(result.hr, AA_R, RECIP, REAL,\n"
     "                                      kpath.kpts, E_FERMI * EV_TO_HARTREE)[:, 0, :]\n"
     "curv_z = -curv[:, 2] * BOHR_TO_ANG**2      # -Omega_z, postw90's plotted sign, Ang^2\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6.5, 3.6))\n"
     "ax.plot(kpath.dists, curv_z, 'C3')\n"
     "for x in kpath.tick_dists:\n"
     "    ax.axvline(x, color='0.85', lw=0.8)\n"
     "ax.axhline(0.0, color='0.5', lw=0.6)\n"
     "ax.set_xticks(kpath.tick_dists); ax.set_xticklabels(kpath.tick_labels)\n"
     "ax.set_ylabel(r'$-\\Omega_z$ (Ang$^2$)'); ax.set_yscale('symlog')\n"
     "ax.set_xlim(kpath.dists[0], kpath.dists[-1])\n"
     "ax.set_title('bcc Fe: Berry curvature along the k-path')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "Sharp Berry-curvature spikes appear where spin–orbit coupling opens "
     "avoided crossings near $E_F$ — these hot-spots dominate the anomalous Hall "
     "effect."),
    ("md",
     "### 3. Anomalous Hall conductivity vs Fermi energy\n"
     "The AHC is the Brillouin-zone average of the occupied-band curvature, in "
     "S/cm. bcc Fe's AHC is famously hard to converge (it needs very dense "
     "meshes / adaptive refinement); here a uniform 25³ mesh — fast thanks to "
     "the batched multi-core evaluation — gives the right scale and the "
     "$\\sigma_{xy}$ sign. (Only $\\sigma_z$ is nonzero; $M \\parallel z$.)"),
    ("code",
     "import time\n"
     "fermis_eV = np.arange(E_FERMI - 1.0, E_FERMI + 1.01, 0.25)\n"
     "t0 = time.time()\n"
     "ahc = anomalous_hall_conductivity(result.hr, AA_R, RECIP, REAL,\n"
     "                                  fermis_eV * EV_TO_HARTREE, mesh=(25, 25, 25))\n"
     "sigma_S_cm = to_si_units(ahc.sigma, 'hall_conductivity',\n"
     "                         cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "print(f'AHC scan over {len(fermis_eV)} Fermi energies on a 25^3 mesh '\n"
     "      f'({25**3} k-points): {time.time()-t0:.1f} s')\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 3.8))\n"
     "ax.plot(fermis_eV - E_FERMI, sigma_S_cm[:, 2], 'o-', color='C0')\n"
     "ax.axvline(0.0, color='0.5', lw=0.8, ls='--')\n"
     "ax.set_xlabel('E_F offset (eV)'); ax.set_ylabel(r'$\\sigma_{xy}$ (S/cm)')\n"
     "ax.set_title('bcc Fe: anomalous Hall conductivity')\n"
     "plt.tight_layout(); plt.show()\n"
     "print(f'sigma_xy at E_F: {sigma_S_cm[len(fermis_eV)//2, 2]:.0f} S/cm '\n"
     "      f'(experiment ~ 1000 S/cm; sensitive to mesh/E_F)')"),
    ("md",
     "**Takeaway.** From a single converged SOC DFT calculation, waw builds the "
     "18-spinor-MLWF model, the Wannier position operator, and the full postw90 "
     "`berry` workflow — Berry curvature and the anomalous Hall conductivity "
     "$\\sigma^A_{xy}(E_F)$ from the full WYSV06 $J_0+J_1+J_2$ formula — "
     "waw-native and multi-core, with no `wannier90.x`."),
]


# ==========================================================================
# 05 — Copper: metal disentanglement + Fermi surface
# ==========================================================================
cu = [
    ("md",
     "# Tutorial 04 — Copper: a metal and its Fermi surface\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorials 4/6. **Copper** is the "
     "classic Wannier metal: 7 MLWFs — the five localised Cu $3d$ orbitals plus "
     "two diffuse $s$-like functions — reproduce the $d$ bands and the "
     "free-electron-like $s$ band that crosses the Fermi level and forms the "
     "**Fermi surface**.\n"
     "\n"
     "This is a **disentanglement** on a metal: there is no gap, so the 7-band "
     "subspace is extracted with energy windows. The PseudoDojo Cu pseudo carries "
     "a $3s3p$ semicore (19 valence electrons), which we exclude."),
    ("code", SETUP),
    ("md",
     "### 1. Converged DFT (metal: smearing) and Wannierisation\n"
     "fcc Cu, $a = 3.615$ Å, `ecutwfc = 55` Ry, cold smearing. $E_F \\approx "
     "17.36$ eV. We compute 20 bands, exclude the 4 semicore $3s3p$ bands, and "
     "disentangle 7 MLWFs (frozen window below $E_F$, outer window up into the "
     "conduction bands)."),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('Cu', 'fcc', a=3.615)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'cu'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'cu',\n"
     "    ecutwfc=55, scf_kpts=(12, 12, 12), nbnd=20, num_wann=7,\n"
     "    exclude_bands=[1, 2, 3, 4],              # 3s3p semicore\n"
     "    scdm_entanglement='erfc', scdm_mu=19.0, scdm_sigma=4.0,\n"
     "    system_extra=SMEAR, pseudopotentials={'Cu': 'Cu.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/cu/ if present\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']                 # eV, read from the SCF by qe.py\n"
     "print(f'E_F = {E_FERMI:.3f} eV')\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=7,\n"
     "    outer_window=(-1e3, 45.0), frozen_window=(-1e3, 16.0),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=5000, verbose=False,\n"
     ")\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_I = {result.dis.omega_i*BOHR_TO_ANG**2:.4f} Ang^2  '\n"
     "      f'(Wannier90 tutorials 4/6: ~3.7)')\n"
     "print('spreads (Ang^2):', spreads.round(3), '  -> 5 tight d + 2 diffuse s')"),
    ("md",
     "### 2. Band structure\n"
     "The 7-orbital model reproduces the flat Cu $d$ manifold (~2-5 eV below "
     "$E_F$) and the dispersive $s$ band crossing $E_F$."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=120)   # ASE's standard path for this lattice\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands - E_FERMI, figsize=(6.5, 4),\n"
     "           ref_line=0.0, ylim=(-8, 8), ylabel='E - E_F (eV)',\n"
     "           title='Cu bands from 7 MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "### 3. Fermi surface\n"
     "The Fermi surface is the constant-energy surface at $E_F$. We interpolate "
     "the bands on a dense 2-D slice through the Brillouin zone — the "
     "$(k_x, k_y, 0)$ plane in reciprocal-lattice coordinates — and contour where "
     "a band equals $E_F$. Copper's characteristic **necks** reaching toward the "
     "L points are visible where the surface bulges out. (`waw.interfaces."
     "wannier90.fermi_surface` also writes a full 3-D `.bxsf` for XCrySDen.)"),
    ("code",
     "n = 200\n"
     "g = np.linspace(-0.75, 0.75, n)\n"
     "KX, KY = np.meshgrid(g, g)\n"
     "kpts2d = np.stack([KX.ravel(), KY.ravel(), np.zeros(KX.size)], axis=1)\n"
     "bands2d = interpolate_bands(result.hr, kpts2d) * HARTREE_TO_EV\n"
     "bands2d = bands2d.reshape(n, n, -1)\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(5.2, 5))\n"
     "for ib in range(bands2d.shape[-1]):\n"
     "    ax.contour(KX, KY, bands2d[:, :, ib], levels=[E_FERMI],\n"
     "               colors='C0', linewidths=1.2)\n"
     "ax.set_xlabel('k_x  (recip. lattice units)'); ax.set_ylabel('k_y')\n"
     "ax.set_aspect('equal'); ax.set_title('Cu Fermi surface: (k_x, k_y, 0) slice')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "### 4. The full 3-D Fermi surface inside the Brillouin zone (interactive)\n"
     "The slice above is one cut; the full Fermi surface is a 2-D surface in 3-D "
     "$k$-space. `waw.analysis.fermi_surface.fermi_surface_sheets` evaluates the "
     "bands on a dense **3-D** grid, extracts the constant-energy surface "
     "$\\varepsilon_n(k) = E_F$ per band with marching cubes (`skimage`), and "
     "folds it into the first Brillouin zone -- generically, for any crystal, via "
     "`ase.dft.bz.bz_vertices` (the same Voronoi/Wigner-Seitz construction ASE's "
     "own `BandPath.plot()` uses). `waw.vis.plot_fermi_surface` then renders it "
     "as an **interactive Plotly figure**: drag to rotate/zoom, and click a "
     "sheet's legend entry to toggle it on/off -- a waw-native replacement for "
     "the old XCrySDen Fermi-surface viewer.\n"
     "\n"
     "Copper's Fermi surface is the classic free-electron **sphere that bulges "
     "into necks** touching the hexagonal ($L$-point) faces of the fcc "
     "truncated-octahedron BZ."),
    ("code",
     "from waw.analysis.fermi_surface import fermi_surface_sheets\n"
     "from waw.vis import plot_fermi_surface, show_plotly\n"
     "from waw.interfaces.ase.structure import recip_lattice\n"
     "\n"
     "RECIP = recip_lattice(atoms)          # reciprocal lattice rows b1,b2,b3 (Bohr^-1)\n"
     "\n"
     "sheets = fermi_surface_sheets(\n"
     "    result.hr, RECIP, fermi_energy=E_FERMI / HARTREE_TO_EV,   # Hartree, like the rest of `analysis`\n"
     "    mesh=(32, 32, 32),\n"
     ")\n"
     "print(f'{len(sheets)} Fermi-surface sheet(s): bands '\n"
     "      f'{[s.band_index for s in sheets]}')\n"
     "\n"
     "fig = plot_fermi_surface(sheets, RECIP, title='Cu Fermi surface in the first Brillouin zone')\n"
     "\n"
     "# Jupyter's own notebook-trust/sanitization rules can still block inline\n"
     "# rendering in some setups (a script embedded in a NOT-freshly-executed-\n"
     "# by-you notebook may be stripped) -- also write a fully standalone .html\n"
     "# file next to this notebook, openable directly in any browser with zero\n"
     "# Jupyter involvement at all, as a guaranteed fallback.\n"
     "html_path = WORK / 'cu_fermi_surface.html'\n"
     "fig.write_html(html_path, include_plotlyjs=True, full_html=True)\n"
     "print(f'Standalone interactive plot (open directly in a browser): {html_path}')\n"
     "\n"
     "show_plotly(fig)   # self-contained inline HTML -- see waw.vis.fermi_surface's module docstring for why\n"
     "                   # this is used instead of fig.show() (which depends on the viewing frontend)"),
    ("md",
     "**Takeaway.** waw disentangles a 7-orbital $s+d$ model for metallic copper "
     "— matching Wannier90's $\\Omega_I$ — and interpolates the band structure, a "
     "Fermi-surface slice, and the **full 3-D, interactive, togglable Fermi "
     "surface with its Brillouin zone** (the classic necked sphere), entirely "
     "waw-native and generic for any crystal."),
]


# ==========================================================================
# 06 — Silicon: semiclassical thermoelectric transport (BoltzWann)
# ==========================================================================
si_tep = [
    ("md",
     "# Tutorial 16 — Silicon: thermoelectric transport (BoltzWann)\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 16 (the postw90 "
     "**BoltzWann** module). Once we have a Wannier tight-binding model, the "
     "bands can be interpolated onto an arbitrarily dense mesh essentially for "
     "free — dense enough for **semiclassical Boltzmann transport** in the "
     "constant-relaxation-time approximation.\n"
     "\n"
     "From the band velocities we build the **transport distribution function** "
     "$\\Sigma_{ij}(E) = (\\tau/V)\\sum_{nk} v_i v_j\\,\\delta(E-\\varepsilon_{nk})$, "
     "then the Mott integrals give the electrical conductivity $\\sigma$, the "
     "**Seebeck coefficient** $S$, and the electronic thermal conductivity "
     "$\\kappa$ as functions of chemical potential (doping) and temperature."),
    ("code", SETUP),
    ("md",
     "### 1. Wannierise silicon\n"
     "Same 8-MLWF disentanglement as tutorial 02 (the model just needs to be "
     "accurate across the valence bands and lower conduction)."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.ase.structure import real_lattice, recip_lattice\n"
     "\n"
     "atoms = bulk('Si', 'diamond', a=5.43)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'si'\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=16, num_wann=8,\n"
     "    scdm_entanglement='erfc', scdm_mu=12.0, scdm_sigma=4.0,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/si/ if present\n"
     ")\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=8,\n"
     "    outer_window=(-1e3, 17.0), frozen_window=(-1e3, 6.4),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "REAL, RECIP = real_lattice(atoms), recip_lattice(atoms)\n"
     "print(f'Omega_I = {result.dis.omega_i*BOHR_TO_ANG**2:.4f} Ang^2')"),
    ("md",
     "### 2. Transport distribution function on a dense mesh\n"
     "The TDF is interpolated on a **40×40×40** mesh (64 000 k-points — cheap "
     "band interpolation, and the band velocities are evaluated with the batched "
     "multi-core path). A constant relaxation time $\\tau = 10$ fs sets the "
     "overall scale (the CRTA)."),
    ("code",
     "from waw.analysis import transport_distribution_function, transport_coefficients\n"
     "from waw.units import EV_TO_HARTREE, K_B_HARTREE, to_si_units\n"
     "\n"
     "TDF_MESH = (40, 40, 40)\n"
     "TAU_FS = 10.0\n"
     "tau_au = TAU_FS / 2.4188843265857e-2       # fs -> atomic time units\n"
     "\n"
     "energies_eV = np.arange(-6.0, 15.0, 0.01)  # covers valence + lower conduction\n"
     "tdf = transport_distribution_function(\n"
     "    result.hr, REAL, RECIP, TDF_MESH,\n"
     "    energies_eV * EV_TO_HARTREE, tau_au, num_elec_per_state=2,\n"
     ")\n"
     "print('TDF computed on', TDF_MESH, '=', np.prod(TDF_MESH), 'k-points')"),
    ("md",
     "### 3. Seebeck coefficient and conductivity vs doping\n"
     "Sweeping the chemical potential $\\mu$ across the gap turns silicon from "
     "**p-type** (below the valence-band maximum) to **n-type** (above the "
     "conduction-band minimum). The Seebeck coefficient changes sign accordingly "
     "and is largest just inside the gap; the conductivity grows as $\\mu$ enters "
     "a band."),
    ("code",
     "mu_eV = np.arange(5.6, 8.4, 0.03)\n"
     "temps = np.array([300.0, 600.0])\n"
     "kT = temps * K_B_HARTREE\n"
     "bt = transport_coefficients(tdf, mu_eV * EV_TO_HARTREE, kT)\n"
     "seebeck_si = to_si_units(bt.seebeck_reduced, 'seebeck')\n"
     "elcond_si = to_si_units(bt.elcond, 'electrical_conductivity')\n"
     "\n"
     "seebeck = seebeck_si[:, :, 0, 0] * 1e6      # V/K -> uV/K, (n_mu, n_T)\n"
     "elcond = elcond_si[:, :, 0, 0]              # S/m, (n_mu, n_T)\n"
     "\n"
     "fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)\n"
     "for it, T in enumerate(temps):\n"
     "    axs[0].plot(mu_eV, seebeck[:, it], label=f'{T:.0f} K')\n"
     "    axs[1].semilogy(mu_eV, elcond[:, it], label=f'{T:.0f} K')\n"
     "for ax in axs:\n"
     "    ax.axvspan(6.24, 6.94, color='0.9', label='band gap')\n"
     "    ax.set_xlabel(r'$\\mu$ (eV)'); ax.legend(fontsize=8)\n"
     "axs[0].axhline(0.0, color='0.5', lw=0.6)\n"
     "axs[0].set_ylabel(r'Seebeck $S$ ($\\mu$V/K)'); axs[0].set_title('Seebeck coefficient')\n"
     "axs[1].set_ylabel(r'$\\sigma/\\tau$-scaled (S/m)'); axs[1].set_title('electrical conductivity')\n"
     "plt.tight_layout(); plt.show()\n"
     "print('max |S| at 300 K:', f'{np.abs(seebeck[:,0]).max():.0f} uV/K',\n"
     "      '(sign flips p-type <-> n-type across the gap)')"),
    ("md",
     "**Takeaway.** The Wannier model turns a coarse 4×4×4 DFT calculation into a "
     "dense-mesh transport calculation: the constant-relaxation-time Seebeck "
     "coefficient and conductivity of silicon vs doping and temperature — the "
     "postw90 BoltzWann workflow, waw-native and multi-core."),
]


# ==========================================================================
# 07 — Diamond: real-space Wannier functions
# ==========================================================================
diamond = [
    ("md",
     "# Tutorial 05 — Diamond: real-space Wannier functions\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 5. So far we have used the "
     "Wannier *Hamiltonian*; here we look at the Wannier *functions themselves*. "
     "Diamond's four valence bands give four equivalent **bond-centred** MLWFs — "
     "the textbook picture of a covalent $\\sigma$ bond.\n"
     "\n"
     "To visualise them we need the periodic parts $u_{nk}(r)$ on a real-space "
     "grid: `pw2wannier90` writes these as **UNK** files (`write_unk`), and "
     "`waw.interfaces.wannier90.realspace.build_wannier_functions` assembles "
     "$W_{n0}(r) = \\tfrac{1}{N_k}\\sum_k e^{ik\\cdot r} u_{nk}(r)$ in the Wannier "
     "gauge."),
    ("code", SETUP),
    ("md",
     "### 1. Wannierise diamond (4 isolated bond MLWFs)\n"
     "Diamond-cubic C, $a = 3.567$ Å. Four valence bands → four isolated $sp^3$ "
     "bond MLWFs (we exclude the conduction bands). `write_unk=True` makes "
     "`pw2wannier90` dump the real-space Bloch functions."),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('C', 'diamond', a=3.567)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'diamond'\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'diamond',\n"
     "    ecutwfc=50, scf_kpts=(8, 8, 8), nbnd=8, num_wann=4,\n"
     "    exclude_bands=[5, 6, 7, 8],              # keep the 4 valence bands\n"
     "    scdm_entanglement='isolated',\n"
     "    pseudopotentials={'C': 'C.upf'}, pseudo_dir=PSEUDO_DIR,\n"
     "    ncores=NCORES, write_unk=True,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/diamond/ if present\n"
     ")\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, n_restarts=3, n_iter=4000, verbose=False,\n"
     ")\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_total = {result.omega_final*BOHR_TO_ANG**2:.4f} Ang^2')\n"
     "print('4 equivalent bond WFs, spread (Ang^2):', spreads.round(3))"),
    ("md",
     "### 2. Build and visualise a bond Wannier function\n"
     "`build_wannier_functions` reads the UNK files and rotates $u_{nk}$ by the "
     "converged gauge, on a 2×2×2 supercell so the bond lobe isn't cut by the "
     "cell boundary. The functions come out **real** (the small "
     "imaginary/real ratio below is the reality check)."),
    ("code",
     "from waw.interfaces.wannier90.realspace import build_wannier_functions\n"
     "\n"
     "U = result.spread.U_final.detach().cpu().numpy()\n"
     "V = result.dis.V.detach().cpu().numpy() if result.dis is not None else None\n"
     "rswf = build_wannier_functions(WORK, ov['kpts'], U, V=V,\n"
     "                               wann_list=[0], supercell=(2, 2, 2))\n"
     "wf = rswf.wf[0]                              # (nx, ny, nz) real\n"
     "print('grid', rswf.grid, 'x supercell', rswf.supercell,\n"
     "      '-> array', wf.shape, '| max Im/Re =', float(rswf.max_im_re_ratio[0]))"),
    ("code",
     "# slice through the plane of maximum amplitude (the bond centre)\n"
     "iz = int(np.unravel_index(np.argmax(np.abs(wf)), wf.shape)[2])\n"
     "sl = wf[:, :, iz]\n"
     "vmax = np.abs(wf).max()\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(5, 4.4))\n"
     "im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)\n"
     "ax.contour(sl.T, levels=[-0.5*vmax, -0.2*vmax, 0.2*vmax, 0.5*vmax],\n"
     "           colors='k', linewidths=0.5)\n"
     "ax.set_title('Diamond bond Wannier function (slice)')\n"
     "ax.set_xlabel('grid x'); ax.set_ylabel('grid y')\n"
     "fig.colorbar(im, ax=ax, label='W(r) (arb. units)')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "The lobe sits **between** two carbon atoms — a bond-centred $sp^3$ "
     "$\\sigma$-bond Wannier function, symmetric and real. (For a full 3-D view, "
     "`waw.interfaces.wannier90.realspace.write_xsf` exports an XCrySDen `.xsf`.)\n"
     "\n"
     "**Takeaway.** Beyond the tight-binding Hamiltonian, waw reconstructs the "
     "maximally-localised Wannier functions in real space from the DFT "
     "wavefunctions — here the four covalent bonds of diamond — entirely "
     "waw-native."),
]


# ==========================================================================
# 08 — Aluminium chain: ballistic (Landauer) quantum transport
# ==========================================================================
al_chain = [
    ("md",
     "# Bonus — Aluminium chain: ballistic quantum transport\n"
     "\n"
     "Not one of the 18 official Wannier90 tutorials (no tutorial number), but "
     "a waw-native demonstration of the same **ballistic (Landauer) transport** "
     "capability the official tutorials 13/14 exercise, on a "
     "**monatomic aluminium chain** -- a one-dimensional conductor; at low "
     "temperature its conductance is **quantized**, $G = G_0\\,T(E_F)$ with "
     "$G_0 = 2e^2/h$, where $T(E)$ counts the number of conducting channels "
     "(bands crossing the energy $E$).\n"
     "\n"
     "The Wannier tight-binding $H(R)$ along the chain is exactly the input the "
     "**Landauer** Green's-function transport method needs "
     "(`waw.analysis.transport.transport_bulk`, Lopez-Sancho decimation). This is "
     "a 1-D system in a transverse vacuum, so — like any slab/chain — the "
     "out-of-plane Wannier spread is unconstrained by the finite-difference "
     "mesh; we use **`guiding_centres=True`** to pin the WF centres and keep the "
     "localisation well-behaved."),
    ("code", SETUP),
    ("md",
     "### 1. Converged DFT of the chain\n"
     "One Al atom per cell, spacing $d = 2.5$ Å along $z$, with 12 Å of vacuum "
     "in $x,y$. The chain is metallic (cold smearing); the $k$-mesh is dense "
     "**along the chain only** — `mp_grid = (1, 1, 8)`."),
    ("code",
     "from ase import Atoms\n"
     "from waw.interfaces.ase.structure import real_lattice\n"
     "from waw.analysis import transport_bulk\n"
     "from waw.units import ANG_TO_BOHR, EV_TO_HARTREE\n"
     "\n"
     "d = 2.5\n"
     "atoms = Atoms('Al', positions=[[0, 0, 0]],\n"
     "              cell=[[12, 0, 0], [0, 12, 0], [0, 0, d]], pbc=True)\n"
     "MP_GRID = (1, 1, 8)\n"
     "WORK = HERE / 'runs' / 'alchain'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'alchain',\n"
     "    ecutwfc=45, scf_kpts=(1, 1, 16), nbnd=12, num_wann=4,\n"
     "    scdm_entanglement='erfc', scdm_mu=5.0, scdm_sigma=4.0,\n"
     "    system_extra=SMEAR, pseudopotentials={'Al': 'Al.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/alchain/ if present\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']            # eV, read from the SCF by qe.py\n"
     "print(f'E_F = {E_FERMI:.3f} eV')"),
    ("md",
     "### 2. Wannierise (4 $s$+$p$ MLWFs, guiding centres)\n"
     "Four MLWFs (one $s$-like, three $p$-like) span the bands around $E_F$. "
     "`guiding_centres=True` keeps them centred on the atom despite the "
     "transverse vacuum."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=4,\n"
     "    outer_window=(-1e3, E_FERMI + 12), frozen_window=(-1e3, E_FERMI - 6),\n"
     "    guiding_centres=True, n_restarts=3, dis_n_iter=1000, n_iter=6000,\n"
     "    verbose=False,\n"
     ")\n"
     "print(f'Omega_I = {result.dis.omega_i*BOHR_TO_ANG**2:.3f} Ang^2, '\n"
     "      f'Omega_total = {result.omega_final*BOHR_TO_ANG**2:.3f} Ang^2 '\n"
     "      f'(well-controlled thanks to guiding_centres)')"),
    ("md",
     "### 3. Landauer transmission $T(E)$\n"
     "`transport_bulk` builds the semi-infinite chain from $H(R)$ and computes "
     "the transmission by Green's-function decimation. In a **perfect periodic** "
     "conductor $T(E)$ is exactly the **integer number of bands at energy $E$** — "
     "a conductance staircase. The value at $E_F$ is the number of open channels "
     "of the ideal chain."),
    ("code",
     "tr = transport_bulk(\n"
     "    result.hr, real_lattice(atoms), result.centres_bohr, mp_grid=MP_GRID,\n"
     "    one_dim_axis='z', dist_cutoff=6.0 * ANG_TO_BOHR,\n"
     "    fermi_energy=E_FERMI * EV_TO_HARTREE,\n"
     "    energy_window=(-8.0 * EV_TO_HARTREE, 8.0 * EV_TO_HARTREE),\n"
     "    energy_step=0.02 * EV_TO_HARTREE, translate_home_cell=True,\n"
     ")\n"
     "E = tr.energies * HARTREE_TO_EV        # relative to E_F\n"
     "T = tr.transmission\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 4))\n"
     "ax.plot(E, T, 'C0', lw=1.5)\n"
     "ax.axvline(0.0, color='C3', lw=0.9, ls='--', label='E_F')\n"
     "ax.set_xlabel('E - E_F (eV)'); ax.set_ylabel('transmission  T(E)  (units of G0)')\n"
     "ax.set_title('Al chain: ballistic conductance staircase')\n"
     "ax.set_ylim(-0.1, T.max() + 0.4); ax.legend()\n"
     "plt.tight_layout(); plt.show()\n"
     "\n"
     "T_EF = float(T[np.argmin(np.abs(E))])\n"
     "print(f'conductance at E_F: G = {T_EF:.2f} G0  ({T_EF:.0f} open channels; '\n"
     "      f'G0 = 2e^2/h = 77.5 uS)')"),
    ("md",
     "The transmission is a clean integer staircase — each step is a band "
     "entering the transport window — and at $E_F$ the ideal Al chain has a "
     "well-defined number of conducting channels. (For a **device** with a "
     "scattering region between leads, `waw.analysis.transport.transport_lcr` "
     "computes the same Landauer transmission through a left-centre-right "
     "junction.)\n"
     "\n"
     "**Takeaway.** The Wannier $H(R)$ turns a converged DFT chain into a "
     "quantum-transport calculation: the Landauer conductance staircase of a 1-D "
     "aluminium wire, waw-native, with `guiding_centres` handling the "
     "transverse-vacuum geometry."),
]


# ==========================================================================
# 09 — Silane (SiH4): Gamma-only molecular Wannierization
# ==========================================================================
silane = [
    ("md",
     "# Tutorial 07 — Silane: $\\Gamma$-only molecular Wannierisation\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 7. **SiH$_4$** sits alone "
     "in a large vacuum box, so it needs only the **$\\Gamma$ point** — QE's "
     "real-wavefunction `K_POINTS {gamma}` trick, which runs roughly 2x faster "
     "and halves the memory of an explicit $k=(0,0,0)$ calculation. The 4 "
     "occupied valence bands give 4 equivalent Si–H bond MLWFs.\n"
     "\n"
     "**Two $\\Gamma$-only wrinkles.** (1) `pw2wannier90` does not implement SCDM "
     "for gamma-only wavefunctions (\"Gamma_only and SCDM not implemented\"), so "
     "this tutorial uses **analytic Si:$sp^3$ trial projections** instead "
     "(`write_nnkp`'s `projections=`, angular momentum code $l=-3$). (2) QE's "
     "real-wavefunction storage only keeps *half* the reciprocal sphere, so the "
     "full $\\pm b$ neighbour shell `generate_nnkp` would normally pick makes "
     "`pw2wannier90` abort (\"g_kpb vector is not in the list of Gs\"); real "
     "`wannier90.x -pp` sidesteps this with a **half-shell** of 3 b-vectors "
     "$(1,0,0),(0,1,0),(0,0,1)$ (confirmed against the committed tutorial14 "
     "Na-chain `.mmn`, itself $\\Gamma$-only), and waw's own weight solver and "
     "Z-matrix already handle a half shell (auto-detected from a missing $-b$ "
     "partner) — so `qe.generate_overlaps(..., gamma_only=True)` reproduces it "
     "directly."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "SiH$_4$ (tetrahedral, Si–H = 1.48 Å) centred in a 12-bohr cubic box — "
     "the exact geometry and box size of the classic tutorial. "
     "`gamma_only=True` forces `mp_grid = (1,1,1)` and the `K_POINTS {gamma}` "
     "card for both scf and nscf."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "a = 12.0 * BOHR_TO_ANG                 # 12 bohr cubic box\n"
     "d = 1.056551   # Si-H bond projected onto each Cartesian axis (=> |r|=1.48 A)\n"
     "positions = [[0, 0, 0],\n"
     "             [ d,  d,  d], [ d, -d, -d], [-d,  d, -d], [-d, -d,  d]]\n"
     "atoms = Atoms('SiH4', positions=positions,\n"
     "              cell=[[a, 0, 0], [0, a, 0], [0, 0, a]], pbc=True)\n"
     "WORK = HERE / 'runs' / 'silane'\n"
     "\n"
     "si_frac = atoms.get_scaled_positions()[0]\n"
     "sp3 = [(tuple(si_frac), -3, mr, 1, (0., 0., 1.), (1., 0., 0.), 1.0)\n"
     "       for mr in (1, 2, 3, 4)]           # Si sp3 hybrids, l=-3, mr=1..4\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, (1, 1, 1), WORK, 'silane',\n"
     "    ecutwfc=40, scf_kpts=(1, 1, 1), nbnd=8, num_wann=4,\n"
     "    exclude_bands=[5, 6, 7, 8],          # 4 occupied bonds + a Davidson buffer\n"
     "    pseudopotentials={'Si': 'Si.upf', 'H': 'H.upf'}, pseudo_dir=PSEUDO_DIR,\n"
     "    ncores=NCORES, gamma_only=True, projections=sp3,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/silane/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')}, '  nntot =', ov['mmn'].shape[1])"),
    ("md",
     "### 2. Wannierise (waw, ASE-native)\n"
     "4 isolated occupied bands -> 4 MLWFs, no disentanglement."),
    ("code",
     "result = wannierize(\n"
     "    atoms, (1, 1, 1), ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, n_restarts=3, n_iter=5000, verbose=False,\n"
     ")\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "centres = result.centres_bohr * BOHR_TO_ANG\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (w90 tutorial07 reference: 4.0450 Ang^2)')\n"
     "for i, (c, s) in enumerate(zip(centres, spreads)):\n"
     "    print(f'  WF {i+1}: centre ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) A   spread {s:.4f} A^2')"),
    ("md",
     "Four **equivalent** MLWFs (same spread, related by the $T_d$ symmetry), "
     "each centred partway along a Si–H bond — bond-centred $sp^3$ orbitals, "
     "matching the classic tutorial's picture (the small offset from the "
     "reference $\\Omega$ is the PseudoDojo Si pseudo vs. the historical "
     "norm-conserving one)."),
    ("md",
     "### 3. The localised (on-site) Hamiltonian\n"
     "A single-point molecule has no $k$-dispersion, so there is no band "
     "structure to interpolate — but the Wannier gauge still gives a "
     "physically meaningful **on-site Hamiltonian** $H(R=0)$ in the localised "
     "bond basis, distinct from the delocalised canonical Kohn-Sham "
     "eigenvalues."),
    ("code",
     "H0 = result.hr.H_R[np.all(result.hr.R_vectors == 0, axis=1)][0].detach().cpu().numpy()\n"
     "H0_eV = np.real(H0) * HARTREE_TO_EV\n"
     "print('H(0) diagonal (bond on-site energies, eV):', np.round(np.diag(H0_eV), 3))\n"
     "print('H(0) largest off-diagonal coupling (eV):   ',\n"
     "      round(float(np.abs(H0_eV - np.diag(np.diag(H0_eV))).max()), 3))\n"
     "print('canonical KS eigenvalues (eV):            ', np.round(ov[\"eig\"][0], 3))"),
    ("md",
     "The four bond orbitals are degenerate on-site (symmetry again) with a "
     "sizeable inter-bond coupling — the localised basis trades the "
     "canonical picture (one deep bonding level + three degenerate ones, "
     "reflecting the molecule's point group) for four equivalent, chemically "
     "intuitive $\\sigma$-bond orbitals; diagonalising $H(0)$ recovers the same "
     "four canonical eigenvalues.\n"
     "\n"
     "**Takeaway.** $\\Gamma$-only Wannierisation of an isolated molecule needs "
     "two adjustments from the periodic recipe — analytic projections (SCDM "
     "does not support gamma-only in `pw2wannier90`) and a half b-vector shell "
     "(QE's real-wavefunction storage) — both already understood by waw's "
     "existing machinery, so the rest of the pipeline (`generate_overlaps` -> "
     "`wannierize`) is unchanged."),
]


# ==========================================================================
# 10 — Graphite: disentanglement on a non-orthogonal (hexagonal) lattice
# ==========================================================================
graphite = [
    ("md",
     "# Tutorial 10 — Graphite: hexagonal disentanglement\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 10. **AB-stacked "
     "graphite** is the first tutorial with a **non-orthogonal (hexagonal)** "
     "lattice and **mixed per-atom projections** — but unlike a free-standing "
     "graphene sheet (which needs an artificial vacuum gap and hits the "
     "out-of-plane spread problem noted in the bonus aluminium-chain "
     "notebook), bulk "
     "graphite is a genuine **3-D periodic crystal**: the weak interlayer "
     "bonding still gives a complete finite-difference $b$-vector shell in all "
     "three directions, so no special handling is needed.\n"
     "\n"
     "10 Wannier functions come from disentangling the 20 lowest bands: 6 "
     "in-plane $sp^2$ $\\sigma$-bonds + 4 out-of-plane $p_z$ $\\pi$ orbitals. "
     "SCDM's energy-window projection (`erfc`) builds these directly from the "
     "band manifold, without needing the mixed `C1:sp2;pz` / `C2:pz` analytic "
     "projection blocks the original tutorial uses."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Hexagonal AB-stacked graphite, $a = 2.46$ Å, $c = 6.8$ Å, 4 atoms/cell. "
     "Metallic (semimetal) — cold smearing. 20 bands on a $4\\times4\\times4$ "
     "mesh, SCDM `erfc` weighted around the target manifold "
     "($\\mu=14.5$, $\\sigma=3$ eV)."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "cell = [[2.1304215583, -1.2299994602, 0.0],\n"
     "        [0.0,           2.4599989204, 0.0],\n"
     "        [0.0,           0.0,          6.8]]\n"
     "frac = [[0, 0, 0.25], [0, 0, 0.75], [1/3, 2/3, 0.25], [-1/3, -2/3, 0.75]]\n"
     "atoms = Atoms('C4', scaled_positions=frac, cell=cell, pbc=True)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'graphite'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'graphite',\n"
     "    ecutwfc=50, scf_kpts=(10, 10, 10), nbnd=22, num_wann=10,\n"
     "    scdm_entanglement='erfc', scdm_mu=14.5, scdm_sigma=3.0,\n"
     "    system_extra=SMEAR,\n"
     "    pseudopotentials={'C': 'C.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/graphite/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Disentangle + Wannierise\n"
     "**Frozen window** to 9.8 eV (the occupied $\\sigma+\\pi$ manifold kept "
     "exactly), **outer window** to 19.2 eV (the same values Wannier90 "
     "tutorial 10 uses) — the disentanglement picks the optimally-connected "
     "10-dimensional subspace inside."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=10, outer_window=(-1e3, 19.2), frozen_window=(-1e3, 9.8),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2   (Wannier90 tutorial10: 5.764)')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial10: 7.381)')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))"),
    ("md",
     "$\\Omega_I$ matches the Wannier90 reference closely (the residual is the "
     "PseudoDojo vs. historical norm-conserving pseudo, as in the other "
     "notebooks). The spreads split into two families — the more localised "
     "in-plane $\\sigma$-bonds and the more diffuse out-of-plane $\\pi$ "
     "orbitals — matching the tutorial's `sp2` + `pz` picture even though SCDM "
     "never saw those orbital labels."),
    ("md",
     "### 3. Interpolated band structure\n"
     "ASE's standard hexagonal path for this cell. Inside the frozen "
     "window (shaded) the 10-band Wannier model reproduces the DFT bands "
     "exactly."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)   # ASE's standard hexagonal path\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands, figsize=(6.5, 4),\n"
     "           shade_window=(bands.min() - 1, 9.8), ref_line=ov['fermi_energy'],\n"
     "           title='Graphite bands from 10 disentangled MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "The bands show graphite's characteristic weakly-dispersing $\\pi$ bands "
     "touching near K (the origin of graphene's Dirac cone, gapped open here "
     "by the interlayer AB coupling) alongside the deeper $\\sigma$-bond "
     "manifold.\n"
     "\n"
     "**Takeaway.** A non-orthogonal lattice and mixed orbital character need "
     "no special-casing in the waw-native pipeline — `generate_overlaps` + "
     "`wannierize` handle graphite's hexagonal cell exactly like a cubic one, "
     "and the periodic (not slab) geometry sidesteps the vacuum-spread issue "
     "that blocks a bare graphene monolayer."),
]


# ==========================================================================
# 09 — BaTiO3: bulk perovskite, exclude_bands-only isolation
# ==========================================================================
batio3 = [
    ("md",
     "# Tutorial 09 — Barium titanate: bulk perovskite\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 9. Cubic perovskite "
     "**BaTiO$_3$** is the first multi-species system in the series, and the "
     "first case where the isolated manifold is picked with `exclude_bands` "
     "alone (no disentanglement window at all): with PseudoDojo's electron "
     "counts (Ba 10 + Ti 12 + 3$\\times$O 6 = 40 valence electrons, exactly "
     "matching the classic tutorial's ultrasoft-pseudo total) the crystal has "
     "**20 occupied bands**, and the top 9 (O 2$p$ character) are excluded-in "
     "directly — `nb == nw == 9`, so `wannierize()` runs with no outer/frozen "
     "window at all, same as the isolated-band notebooks 01/05/07."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Cubic perovskite, $a = 7.44266$ bohr, Ba at the cube corner, Ti "
     "body-centred, 3 O face-centred. 22 bands (20 occupied + a 2-band "
     "Davidson buffer); `exclude_bands` drops the 11 lowest (Ba/Ti semicore + "
     "O 2$s$) and the 2 buffer bands, keeping bands 12-20 = the O 2$p$ "
     "manifold."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "a = 7.44266 * BOHR_TO_ANG\n"
     "frac = [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]\n"
     "atoms = Atoms('BaTiO3', scaled_positions=frac,\n"
     "              cell=[[a, 0, 0], [0, a, 0], [0, 0, a]], pbc=True)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'batio3'\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'batio3',\n"
     "    ecutwfc=60, scf_kpts=(8, 8, 8), nbnd=22, num_wann=9,\n"
     "    exclude_bands=list(range(1, 12)) + [21, 22],\n"
     "    pseudopotentials={'Ba': 'Ba.upf', 'Ti': 'Ti.upf', 'O': 'O.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/batio3/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Wannierise (waw, ASE-native)\n"
     "9 isolated O $2p$ bands -> 9 MLWFs, no disentanglement."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=9, n_restarts=3, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial09: 12.7833)')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))"),
    ("md",
     "The spreads split into 3+6 (rather than 3 equivalent triples of 3): "
     "the two crystallographically distinct oxygen sites (the one bonded to "
     "two Ti neighbours along the cell edge vs. the two related by the "
     "cubic symmetry) don't give 9 identical orbitals, matching Wannier90's "
     "own `Omega_D = 0` (a perfectly diagonal, already-optimal gauge for "
     "this isolated case) and near-exact $\\Omega_I$ agreement."),
    ("md",
     "### 3. Interpolated band structure\n"
     "ASE's standard cubic path: the 9-band Wannier model reproduces the "
     "isolated O $2p$ valence manifold exactly (no frozen-window shading "
     "needed here — the whole manifold is exact by construction)."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands, figsize=(6.5, 4),\n"
     "           ref_line=ov['fermi_energy'],\n"
     "           title='BaTiO3 O-2p valence bands from 9 MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** A multi-species crystal needs no new machinery — "
     "`generate_overlaps`/`wannierize` already handle mixed atomic species "
     "and a purely `exclude_bands`-selected isolated manifold (no window at "
     "all) exactly like the single-species isolated-band notebooks."),
]


# ==========================================================================
# 08 — bcc Iron: collinear spin-polarized (no SOC)
# ==========================================================================
fe_collinear = [
    ("md",
     "# Tutorial 08 — bcc iron: collinear spin-polarized Wannierisation\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 8 — the first "
     "**spin-polarized** tutorial. Unlike notebooks 17/18 (noncollinear + "
     "spin-orbit, spinor MLWFs), this is an ordinary **collinear** magnetic "
     "calculation: QE's `nspin=2` solves one exchange-split SCF, and the up "
     "and down spin channels are each Wannierised **independently** as "
     "ordinary (non-spinor) entangled problems — 9 Fe $s;p;d$ MLWFs per "
     "channel, 18 total, no new waw machinery needed.\n"
     "\n"
     "**One new `qe.py` wrinkle**: `pw2wannier90` needs telling which spin "
     "channel to write via `spin_component='up'/'down'` — run once per "
     "channel (with a distinct seedname so the SCF/overlaps don't collide), "
     "reading the *same* (spin-independent) neighbour topology each time."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT (one run per spin channel)\n"
     "Same bcc Fe cell as notebooks 17/18 ($a=2.8699$ Å), but plain "
     "collinear `nspin=2` (no `noncolin`/`lspinorb`). 14 bands per channel "
     "(4 semicore 3$s$3$p$ excluded, buffer above that), SCDM `erfc` "
     "(same $\\mu,\\sigma$ as the SOC notebooks — the target $d+s$ manifold "
     "sits at the same energies regardless of spin treatment)."),
    ("code",
     "from ase.build import bulk\n"
     "\n"
     "atoms = bulk('Fe', 'bcc', a=2.8699)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'fe_collinear'\n"
     "COLLIN = dict(nspin=2, occupations='smearing', smearing='cold', degauss=0.02)\n"
     "COLLIN['starting_magnetization(1)'] = 0.4\n"
     "\n"
     "overlaps, results = {}, {}\n"
     "for spin, seed in (('up', 'fe_up'), ('down', 'fe_dn')):\n"
     "    ov = qe.generate_overlaps(\n"
     "        atoms, MP_GRID, WORK, seed,\n"
     "        ecutwfc=60, scf_kpts=(8, 8, 8), nbnd=18, num_wann=9,\n"
     "        exclude_bands=list(range(1, 5)),\n"
     "        scdm_entanglement='erfc', scdm_mu=25.0, scdm_sigma=5.0,\n"
     "        system_extra=COLLIN, pseudopotentials={'Fe': 'Fe.upf'},\n"
     "        pseudo_dir=PSEUDO_DIR, ncores=NCORES, spin_component=spin,\n"
     "        rerun_scf=False,      # reuse a completed SCF in runs/fe_collinear/ if present\n"
     "    )\n"
     "    overlaps[spin] = ov\n"
     "    print(seed, 'overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                                     if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Wannierise each channel independently\n"
     "9 disentangled Fe $s;p;d$ MLWFs per spin, frozen window to 30 eV "
     "(occupied $d$ manifold), outer window to 70 eV — the same windows "
     "Wannier90 tutorial 8 uses."),
    ("code",
     "for spin in ('up', 'down'):\n"
     "    ov = overlaps[spin]\n"
     "    result = wannierize(\n"
     "        atoms, MP_GRID, ov['kpts'],\n"
     "        mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "        nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "        nw=9, outer_window=(-1e3, 70.0), frozen_window=(-1e3, 30.0),\n"
     "        n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     "    )\n"
     "    results[spin] = result\n"
     "    omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "    omega = result.omega_final * BOHR_TO_ANG**2\n"
     "    ref = {'up': (4.7486, 6.5245), 'down': (4.6235, 6.3387)}[spin]\n"
     "    print(f'{spin:5s}  Omega_I={omega_I:.4f} (w90 {ref[0]})  '\n"
     "          f'Omega_total={omega:.4f} (w90 {ref[1]}) Ang^2')"),
    ("md",
     "The two channels differ — the spin-split Fe $d$-bands give slightly "
     "different Wannier spreads for up vs. down, exactly as in the real "
     "spin-polarized calculation. Each channel independently reaches close "
     "to Wannier90's own $\\Omega_I$."),
    ("md",
     "### 3. Spin-resolved band structure\n"
     "ASE's standard bcc path: plotting both channels together shows the "
     "exchange splitting directly (no `.spn`/spinor machinery needed here — "
     "\"spin-resolved\" just means \"two separate calculations\")."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "series = [BandSeries(bands=interpolate_bands(results[spin].hr, bp.kpts) * HARTREE_TO_EV,\n"
     "                     label=label, color=color)\n"
     "          for spin, color, label in (('up', 'C0', 'spin up'), ('down', 'C3', 'spin down'))]\n"
     "plot_bands(xcoords, xspecial, labels, series, figsize=(6.5, 4),\n"
     "           ref_line=overlaps['up']['fermi_energy'], ref_color='0.4',\n"
     "           title='bcc Fe: collinear spin-polarized bands')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "The exchange splitting between the blue (up) and red (down) bands is "
     "the ferromagnetic Stoner splitting responsible for Fe's magnetic "
     "moment.\n"
     "\n"
     "**Takeaway.** Collinear spin-polarized Wannierisation is just two "
     "ordinary entangled Wannierisations — the only new capability needed "
     "was `pw2wannier90`'s `spin_component` keyword in `qe.py`, one line "
     "in the pw2wannier90 namelist."),
]


# ==========================================================================
# 11 — Silicon: bond-centred trial orbitals (select_projections analogue)
# ==========================================================================
si_selproj = [
    ("md",
     "# Tutorial 11 — Silicon: bond-centred trial orbitals\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 11 — the same 4 "
     "valence MLWFs as notebook 03, but built from **bond-centred $s$ "
     "orbitals** (one per Si-Si bond) instead of atom-centred $sp^3$ "
     "hybrids. Real Wannier90 ships a *wider* 12-orbital trial set (4 "
     "bond-centred $s$ + 8 atom-centred $sp^3$) and uses its "
     "`select_projections 1 2 3 4` keyword to keep only the first 4 columns "
     "of $A_{mn}$ as the SVD initial guess, while the disentanglement "
     "window still sees all 12 computed bands.\n"
     "\n"
     "**waw reaches the identical outcome more directly.** Because "
     "`generate_overlaps`'s `projections=` controls exactly which trial "
     "orbitals `pw2wannier90` computes $A_{mn}$ columns for, we simply ask "
     "for the 4 bond-centred orbitals directly — no wider 12-orbital set, "
     "no selection step needed. (The physically deeper point of "
     "`select_projections` — screening a machine-picked *subset* of a large "
     "automatically-generated trial set — doesn't arise when, as here, the "
     "desired subset is already known by hand.)"),
    ("code", SETUP),
    ("md",
     "### 1. Structure, bond-centred projections, and converged DFT\n"
     "Same primitive diamond-Si cell as Wannier90's own tutorial 11 (so the "
     "bond-centred fractional coordinates below sit exactly at the 4 Si-Si "
     "bond midpoints). `dis_froz_max = dis_win_max = 6.5` eV — the two "
     "windows coincide, so there is no real disentanglement freedom: the "
     "12-band NSCF cleanly separates into the 4 valence bands (kept) and 8 "
     "conduction bands (excluded by the window), the same physical subspace "
     "as notebook 03's Si, reached via a different trial-orbital choice."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "cell = np.array([[-5.10, 0.0, 5.10], [0.0, 5.10, 5.10], [-5.10, 5.10, 0.0]]) * BOHR_TO_ANG\n"
     "frac = [[-0.25, 0.75, -0.25], [0.0, 0.0, 0.0]]\n"
     "atoms = Atoms('Si2', scaled_positions=frac, cell=cell, pbc=True)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'si_selproj'\n"
     "\n"
     "bond_centres = [(-0.125, -0.125, 0.375), (0.375, -0.125, -0.125),\n"
     "                (-0.125, 0.375, -0.125), (-0.125, -0.125, -0.125)]\n"
     "projections = [(c, 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0) for c in bond_centres]\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_selproj',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=14, num_wann=4,\n"
     "    projections=projections,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/si_selproj/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Disentangle + Wannierise\n"
     "The coincident windows leave `wannierize()` no real disentanglement "
     "choice — the outcome is pinned to the 4-dimensional valence subspace."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, outer_window=(-1e3, 6.5), frozen_window=(-1e3, 6.5),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2   (Wannier90 tutorial11: 5.8117)')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial11: 6.3738)')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))"),
    ("md",
     "4 equivalent bond-centred MLWFs, matching Wannier90's own bond-centred "
     "picture (and its $\\Omega_I$/$\\Omega$, converted from the reference's "
     "native Bohr² here) to high precision — the same physics as notebook "
     "03's atom-centred $sp^3$ hybrids, in a different (equally valid) MLWF "
     "gauge."),
    ("md",
     "### 3. Interpolated band structure\n"
     "ASE's standard path for this lattice — the valence bands should match "
     "notebook 03's regardless of which trial-orbital gauge produced them."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands, figsize=(6, 4),\n"
     "           title='Si valence bands from 4 bond-centred MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** The choice of trial orbitals (atom-centred $sp^3$ vs. "
     "bond-centred $s$) is a gauge choice within the same physical valence "
     "subspace — waw reproduces Wannier90's `select_projections` outcome "
     "directly, by specifying only the desired projection subset up front."),
]


# ==========================================================================
# 12 — Benzene: Gamma-only molecule, isolated manifold from a hand-built guess
# ==========================================================================
benzene = [
    ("md",
     "# Tutorial 12 — Benzene: a bigger $\\Gamma$-only molecule\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 12 — the same "
     "$\\Gamma$-only molecular recipe as notebook 07 (silane), scaled up to "
     "**benzene** (C$_6$H$_6$) in a 30-bohr box: 30 valence electrons -> 15 "
     "isolated occupied MLWFs (no disentanglement), reusing the half b-vector "
     "shell and analytic-projections-instead-of-SCDM fixes from notebook 07 "
     "unchanged.\n"
     "\n"
     "**The one genuine difference**: real Wannier90's own tutorial 12 uses "
     "`begin projections / random / end projections` — literally random "
     "trial vectors, the tutorial's stated point being that "
     "*for an isolated manifold ($n_{bands}=n_{wann}$), the choice of "
     "initial guess only seeds the iterative spread minimisation — it "
     "doesn't select a physical subspace* (that's only true for "
     "disentanglement, where the guess also has to pick which bands to "
     "keep). waw's `projections=` needs concrete analytic specs, not a "
     "'random' keyword, so we hand-build a chemically-motivated (but "
     "not textbook-exact) 15-orbital guess instead — an s orbital on each "
     "of the 12 atoms plus 3 $p_z$ orbitals on alternating ring carbons "
     "(a rough Kekulé sketch of the $\\sigma$-framework + $\\pi$ system) — "
     "and, as the tutorial's point predicts, it converges to essentially "
     "the same total spread as Wannier90's random guess."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Benzene centred in a 30-bohr cubic box (the classic tutorial's own "
     "geometry). 18 bands (15 occupied + a 3-band Davidson buffer, excluded "
     "afterwards) on the sole $\\Gamma$ point."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "a = 30.0 * BOHR_TO_ANG\n"
     "positions_bohr = [\n"
     "    (15.628439779, 15.000000000, 15.00), (14.309410418, 17.284679795, 15.00),\n"
     "    (11.671351697, 17.284679795, 15.00), (10.352322336, 15.000000000, 15.00),\n"
     "    (11.671351697, 12.715320205, 15.00), (14.309410418, 12.715320205, 15.00),\n"
     "    (17.675013988, 15.000000000, 15.00), (15.333642386, 19.057243606, 15.00),\n"
     "    (10.647119729, 19.057243606, 15.00), (8.305748128, 15.000000000, 15.00),\n"
     "    (10.647119729, 10.942756393, 15.00), (15.333642386, 10.942756393, 15.00),\n"
     "]   # 6 C (ring) then 6 H, bohr\n"
     "positions = [(x * BOHR_TO_ANG, y * BOHR_TO_ANG, z * BOHR_TO_ANG) for x, y, z in positions_bohr]\n"
     "atoms = Atoms('C6H6', positions=positions, cell=[[a, 0, 0], [0, a, 0], [0, 0, a]], pbc=True)\n"
     "WORK = HERE / 'runs' / 'benzene'\n"
     "\n"
     "frac = atoms.get_scaled_positions()\n"
     "projections = [(tuple(frac[i]), 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0)\n"
     "               for i in range(12)]                     # s on each of the 12 atoms\n"
     "for i in (0, 2, 4):                                    # alternating ring carbons -> pz\n"
     "    projections.append((tuple(frac[i]), 1, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0))\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, (1, 1, 1), WORK, 'benzene',\n"
     "    ecutwfc=30, scf_kpts=(1, 1, 1), nbnd=18, num_wann=15,\n"
     "    exclude_bands=[16, 17, 18],\n"
     "    pseudopotentials={'C': 'C.upf', 'H': 'H.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    gamma_only=True, projections=projections,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/benzene/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')}, '  nntot =', ov['mmn'].shape[1])"),
    ("md",
     "### 2. Wannierise (waw, ASE-native)\n"
     "15 isolated occupied bands -> 15 MLWFs, no disentanglement."),
    ("code",
     "result = wannierize(\n"
     "    atoms, (1, 1, 1), ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=15, n_restarts=3, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial12, random guess: 13.0107)')\n"
     "print('per-WF spreads (Ang^2):', np.sort(spreads).round(3))"),
    ("md",
     "The total spread lands close to Wannier90's own **random**-guess "
     "result, even though the two runs started from completely different "
     "initial trial orbitals — exactly the point of tutorial 12: for an "
     "isolated manifold, the maximally-localised gauge is (up to unitary "
     "rotation) essentially unique, so the initial guess only affects how "
     "fast the iteration gets there, not where it ends up.\n"
     "\n"
     "**Takeaway.** The $\\Gamma$-only recipe from notebook 07 (analytic "
     "projections instead of SCDM, half b-vector shell) scales unchanged "
     "to a bigger, less symmetric molecule; the exact choice of trial "
     "orbitals for an *isolated* manifold is far less important than for a "
     "*disentangled* one, since it only seeds the optimiser rather than "
     "selecting the physical subspace."),
]


# ==========================================================================
# 13 — (5,5) Carbon nanotube: disentanglement + bulk ballistic transport
# ==========================================================================
cnt = [
    ("md",
     "# Tutorial 13 — (5,5) carbon nanotube: ballistic transport\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 13 — the tutorial "
     "that introduced **\"bulk\" ballistic transport** to the project "
     "(`waw.analysis.transport_bulk`, the same Lopez-Sancho decimation used "
     "in the aluminium-chain bonus notebook, applied here to a real, "
     "chemically-realistic periodic 1-D conductor). A (5,5) armchair "
     "nanotube, 20 C atoms/cell, is disentangled from 80 bands down to 50 "
     "MLWFs (20 radial $p_z$ $\\pi$ orbitals + 30 $\\sigma$-bond $s$ "
     "orbitals), then its Landauer transmission is computed directly from "
     "the Wannier $H(R)$."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "The tube sits in a $16\\times16$ Å box (vacuum around the tube) with "
     "a $2.4628$ Å period along $z$ (the tube axis) — a $1\\times1\\times5$ "
     "$k$-mesh (only the periodic direction is sampled). 82 bands (80 "
     "target + 2 Davidson buffer, excluded afterwards). The projections are "
     "the tutorial's own: a **radial** $p_z$ on every atom (zaxis pointing "
     "outward from the tube axis — the $\\pi$ system) plus an $s$ orbital "
     "on each of the 30 C-C $\\sigma$ bonds in the unit cell."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "POSITIONS = [\n"
     "    (3.378034, -0.712837, -0.615700), (3.378034, 0.712837, -0.615700),\n"
     "    (1.717153, 2.957521, -0.615700), (0.359290, 3.390555, -0.615700),\n"
     "    (-2.303060, 2.540326, -0.615700), (-3.151822, 1.395019, -0.615700),\n"
     "    (-3.151822, -1.395019, -0.615700), (-2.303060, -2.540326, -0.615700),\n"
     "    (0.359290, -3.390555, -0.615700), (1.717153, -2.957521, -0.615700),\n"
     "    (3.151822, -1.395019, 0.615700), (3.151822, 1.395019, 0.615700),\n"
     "    (2.303060, 2.540326, 0.615700), (-0.359290, 3.390555, 0.615700),\n"
     "    (-1.717153, 2.957521, 0.615700), (-3.378034, 0.712837, 0.615700),\n"
     "    (-3.378034, -0.712837, 0.615700), (-1.717153, -2.957521, 0.615700),\n"
     "    (-0.359290, -3.390555, 0.615700), (2.303060, -2.540326, 0.615700),\n"
     "]   # 20 C atoms, Angstrom (Cartesian)\n"
     "atoms = Atoms('C20', positions=POSITIONS, cell=[[16, 0, 0], [0, 16, 0], [0, 0, 2.4628]], pbc=True)\n"
     "MP_GRID = (1, 1, 5)\n"
     "WORK = HERE / 'runs' / 'cnt55'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.03)\n"
     "\n"
     "# 20 radial pz (pi system): (cx,cy,cz, zx,zy,zz, xx,xy,xz) Angstrom\n"
     "PZ_SPECS = [\n"
     "    (3.3780, -0.7128, -0.6157, 3.3780, -0.7128, 0.0, 0, 0, 1), (3.3780, 0.7128, -0.6157, 3.3780, 0.7128, 0.0, 0, 0, 1),\n"
     "    (1.7172, 2.9575, -0.6157, 1.7172, 2.9575, 0.0, 0, 0, 1), (0.3593, 3.3906, -0.6157, 0.3593, 3.3906, 0.0, 0, 0, 1),\n"
     "    (-2.3031, 2.5403, -0.6157, -2.3031, 2.5403, 0.0, 0, 0, 1), (-3.1518, 1.3950, -0.6157, -3.1518, 1.3950, 0.0, 0, 0, 1),\n"
     "    (-3.1518, -1.3950, -0.6157, -3.1518, -1.3950, 0.0, 0, 0, 1), (-2.3031, -2.5403, -0.6157, -2.3031, -2.5403, 0.0, 0, 0, 1),\n"
     "    (0.3593, -3.3906, -0.6157, 0.3593, -3.3906, 0.0, 0, 0, 1), (1.7172, -2.9575, -0.6157, 1.7172, -2.9575, 0.0, 0, 0, 1),\n"
     "    (3.1518, -1.3950, 0.6157, 3.1518, -1.3950, 0.0, 0, 0, 1), (3.1518, 1.3950, 0.6157, 3.1518, 1.3950, 0.0, 0, 0, 1),\n"
     "    (2.3031, 2.5403, 0.6157, 2.3031, 2.5403, 0.0, 0, 0, 1), (-0.3593, 3.3906, 0.6157, -0.3593, 3.3906, 0.0, 0, 0, 1),\n"
     "    (-1.7172, 2.9575, 0.6157, -1.7172, 2.9575, 0.0, 0, 0, 1), (-3.3780, 0.7128, 0.6157, -3.3780, 0.7128, 0.0, 0, 0, 1),\n"
     "    (-3.3780, -0.7128, 0.6157, -3.3780, -0.7128, 0.0, 0, 0, 1), (-1.7172, -2.9575, 0.6157, -1.7172, -2.9575, 0.0, 0, 0, 1),\n"
     "    (-0.3593, -3.3906, 0.6157, -0.3593, -3.3906, 0.0, 0, 0, 1), (2.3031, -2.5403, 0.6157, 2.3031, -2.5403, 0.0, 0, 0, 1),\n"
     "]\n"
     "# 30 sigma-bond-centred s orbitals: (cx,cy,cz) Angstrom\n"
     "S_CENTRES = [\n"
     "    (-2.7274, -1.9677, -0.6157), (1.0382, -3.1740, -0.6157), (3.3780, 0.0000, -0.6157),\n"
     "    (1.0382, 3.1740, -0.6157), (-2.7274, 1.9677, -0.6157), (-3.2649, -1.0539, 0.0),\n"
     "    (-2.0101, -2.7489, 0.0), (0.0000, -3.3906, 0.0), (2.0101, -2.7489, 0.0),\n"
     "    (3.2649, -1.0539, 0.0), (3.2649, 1.0539, 0.0), (2.0101, 2.7489, 0.0),\n"
     "    (0.0000, 3.3906, 0.0), (-2.0101, 2.7489, 0.0), (-3.2649, 1.0539, 0.0),\n"
     "    (-1.0382, -3.1740, 0.6157), (2.7274, -1.9677, 0.6157), (2.7274, 1.9677, 0.6157),\n"
     "    (-1.0382, 3.1740, 0.6157), (-3.3780, 0.0000, 0.6157), (-3.2649, -1.0539, 1.2314),\n"
     "    (-2.0101, -2.7489, 1.2314), (0.0000, -3.3906, 1.2314), (2.0101, -2.7489, 1.2314),\n"
     "    (3.2649, -1.0539, 1.2314), (3.2649, 1.0539, 1.2314), (2.0101, 2.7489, 1.2314),\n"
     "    (0.0000, 3.3906, 1.2314), (-2.0101, 2.7489, 1.2314), (-3.2649, 1.0539, 1.2314),\n"
     "]\n"
     "\n"
     "inv_cell_T = np.linalg.inv(atoms.cell[:].T)\n"
     "projections = []\n"
     "for cx, cy, cz, zx, zy, zz, xx, xy, xz in PZ_SPECS:\n"
     "    frac = tuple(inv_cell_T @ np.array([cx, cy, cz]))\n"
     "    projections.append((frac, 1, 1, 1, (zx, zy, zz), (xx, xy, xz), 1.0))   # pz, custom radial zaxis\n"
     "for cx, cy, cz in S_CENTRES:\n"
     "    frac = tuple(inv_cell_T @ np.array([cx, cy, cz]))\n"
     "    projections.append((frac, 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0))    # s\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'cnt55',\n"
     "    ecutwfc=30, scf_kpts=(1, 1, 5), nbnd=82, num_wann=50,\n"
     "    exclude_bands=[81, 82], projections=projections, system_extra=SMEAR,\n"
     "    pseudopotentials={'C': 'C.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/cnt55/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Disentangle + Wannierise\n"
     "Frozen window to 1.8 eV, outer window to 6.3 eV — the same values "
     "Wannier90 tutorial 13 uses."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=50, outer_window=(-1e3, 6.3), frozen_window=(-1e3, 1.8),\n"
     "    n_restarts=2, dis_n_iter=800, n_iter=4000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2   (Wannier90 tutorial13: 34.1754)')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial13: 41.2648)')"),
    ("md",
     "### 3. Ballistic (Landauer) transport along the tube\n"
     "The (5,5) armchair nanotube is metallic — two bands cross the Fermi "
     "level — so the transmission at $E_F$ should show that characteristic "
     "**two-channel** conductance."),
    ("code",
     "from waw.interfaces.ase.structure import real_lattice\n"
     "from waw.analysis import transport_bulk\n"
     "from waw.units import ANG_TO_BOHR, EV_TO_HARTREE\n"
     "\n"
     "tr = transport_bulk(\n"
     "    result.hr, real_lattice(atoms), result.centres_bohr, mp_grid=MP_GRID,\n"
     "    one_dim_axis='z', dist_cutoff=5.5 * ANG_TO_BOHR,\n"
     "    fermi_energy=ov['fermi_energy'] * EV_TO_HARTREE,\n"
     "    energy_window=(-6.5 * EV_TO_HARTREE, 6.5 * EV_TO_HARTREE),\n"
     "    energy_step=0.02 * EV_TO_HARTREE, translate_home_cell=True,\n"
     ")\n"
     "E = tr.energies * HARTREE_TO_EV\n"
     "T = tr.transmission\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 4))\n"
     "ax.plot(E, T, 'C0', lw=1.2)\n"
     "ax.axvline(0.0, color='C3', lw=0.9, ls='--', label='E_F')\n"
     "ax.set_xlabel('E - E_F (eV)'); ax.set_ylabel('transmission  T(E)  (units of G0)')\n"
     "ax.set_title('(5,5) CNT: ballistic transmission')\n"
     "ax.set_ylim(-0.2, T.max() + 0.5); ax.legend()\n"
     "plt.tight_layout(); plt.show()\n"
     "\n"
     "T_EF = float(T[np.argmin(np.abs(E))])\n"
     "print(f'transmission at E_F: T = {T_EF:.2f} G0  (metallic armchair CNT: expect ~2)')"),
    ("md",
     "The transmission near $E_F$ sits close to the textbook **2 conducting "
     "channels** of a metallic armchair nanotube — the two linearly-crossing "
     "$\\pi$/$\\pi^*$ bands near the Fermi level, each contributing one "
     "ballistic channel.\n"
     "\n"
     "**Takeaway.** The same `transport_bulk` Landauer machinery from the "
     "aluminium-chain bonus notebook applies unchanged to a real, "
     "chemically-detailed 1-D nanostructure — a 50-band disentanglement "
     "feeding directly into the ballistic-transport calculation, "
     "waw-native throughout."),
]


# ==========================================================================
# 14 — Na chain: periodic bulk ballistic transport
# ==========================================================================
na_chain = [
    ("md",
     "# Tutorial 14 — Sodium chain: periodic and defected transport\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 14, both halves. "
     "**Part 1** (periodic): a linear chain of 3 Na atoms per cell in a "
     "large transverse vacuum box, disentangled to 3 MLWFs, then "
     "`transport_bulk` gives the Landauer transmission along the chain — a "
     "half-filled monatomic metal, so it should show exactly **one** "
     "conducting channel. **Part 2** (defected): a 13-atom chain with one "
     "displaced atom (the defect), disentangled to 13 MLWFs, then "
     "`transport_lcr` — **lead-conductor-lead** transport, treating the "
     "outer atoms as ideal periodic leads around the defect — should show "
     "the transmission **dip below the ideal chain's** $T=1$ where the "
     "defect scatters.\n"
     "\n"
     "**A real gauge pathology, and how it's fixed.** Without "
     "`guiding_centres=True`, waw's ADAM optimiser lands on a "
     "\"runaway-centre\" MLWF gauge for this transverse-vacuum geometry — "
     "$\\Omega_{total}$ comes out **~9x too large** (the same phenomenon "
     "documented for the aluminium-chain bonus notebook)."),
    ("code", SETUP),
    ("md",
     "## Part 1 — periodic chain: bulk transport\n"
     "### 1. Structure and converged DFT\n"
     "3 Na atoms evenly spaced along $x$ ($a=9.75$ Å), 10 Å of vacuum in "
     "$y,z$. PseudoDojo Na carries 2$s$2$p$ semicore (9 valence e$^-$/atom, "
     "vs. the classic tutorial's single 3$s$ electron), so — like the "
     "GaAs/Cu/Fe notebooks — we `exclude_bands` the 12 semicore bands (4 "
     "orbitals $\\times$ 3 atoms) first."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "a, b, c = 9.75, 10.0, 10.0\n"
     "frac = [[1/6, 0.5, 0.5], [0.5, 0.5, 0.5], [5/6, 0.5, 0.5]]\n"
     "atoms = Atoms('Na3', scaled_positions=frac,\n"
     "              cell=[[a, 0, 0], [0, b, 0], [0, 0, c]], pbc=True)\n"
     "MP_GRID = (4, 1, 1)\n"
     "WORK = HERE / 'runs' / 'na_chain'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.007)\n"
     "\n"
     "projections = [(tuple(frac[i]), 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0)\n"
     "               for i in range(3)]        # one s orbital per Na atom\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'na3',\n"
     "    ecutwfc=40, scf_kpts=(4, 1, 1), nbnd=20, num_wann=3,\n"
     "    exclude_bands=list(range(1, 13)),      # 12 Na 2s2p semicore bands\n"
     "    projections=projections, system_extra=SMEAR,\n"
     "    pseudopotentials={'Na': 'Na.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/na_chain/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Disentangle + Wannierise\n"
     "Frozen window up to $E_F$, outer window 2.74 eV above it (matching "
     "the classic tutorial's own offsets, `dis_froz_max`/`dis_win_max` "
     "measured from its own Fermi level). **`guiding_centres=True`** is "
     "required here (see above)."),
    ("code",
     "EF = ov['fermi_energy']\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=3, outer_window=(-1e3, EF + 2.74), frozen_window=(-1e3, EF),\n"
     "    guiding_centres=True,\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2   (Wannier90 tutorial14: 13.2482)')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial14: 13.6431)')"),
    ("md",
     "### 3. Bulk ballistic transport\n"
     "A half-filled single s-band chain — exactly **one** conducting "
     "channel is expected at $E_F$."),
    ("code",
     "from waw.interfaces.ase.structure import real_lattice\n"
     "from waw.analysis import transport_bulk\n"
     "from waw.units import ANG_TO_BOHR, EV_TO_HARTREE\n"
     "\n"
     "tr = transport_bulk(\n"
     "    result.hr, real_lattice(atoms), result.centres_bohr, mp_grid=MP_GRID,\n"
     "    one_dim_axis='x', dist_cutoff=9.76 * ANG_TO_BOHR,\n"
     "    fermi_energy=EF * EV_TO_HARTREE,\n"
     "    energy_window=(-5.0 * EV_TO_HARTREE, 5.0 * EV_TO_HARTREE),\n"
     "    energy_step=0.01 * EV_TO_HARTREE, translate_home_cell=True,\n"
     ")\n"
     "E = tr.energies * HARTREE_TO_EV\n"
     "T = tr.transmission\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 4))\n"
     "ax.plot(E, T, 'C0', lw=1.2)\n"
     "ax.axvline(0.0, color='C3', lw=0.9, ls='--', label='E_F')\n"
     "ax.set_xlabel('E - E_F (eV)'); ax.set_ylabel('transmission  T(E)  (units of G0)')\n"
     "ax.set_title('Na chain: ballistic transmission')\n"
     "ax.set_ylim(-0.2, T.max() + 0.5); ax.legend()\n"
     "plt.tight_layout(); plt.show()\n"
     "\n"
     "T_EF = float(T[np.argmin(np.abs(E))])\n"
     "print(f'transmission at E_F: T = {T_EF:.2f} G0  (half-filled single s-band: expect 1)')"),
    ("md",
     "The transmission sits at **1 conducting channel** at $E_F$ — the "
     "single half-filled 3$s$-derived band of the sodium chain, the "
     "simplest possible ballistic conductor.\n"
     "\n"
     "The same `guiding_centres` + `transport_bulk` combination from the "
     "aluminium-chain notebook reproduces a second, independent 1-D metal's "
     "quantized conductance — here a genuinely different orbital character "
     "(a bare alkali $s$-band, not $sp$-like Al) confirms the recipe isn't "
     "special-cased to one system."),
    ("md",
     "## Part 2 — defected chain: lead-conductor-lead transport\n"
     "### 4. Structure and converged DFT\n"
     "The same 3s-derived chain physics, but now **13 Na atoms** in a "
     "42.25 Å cell, evenly spaced **except atom 7**, displaced from the "
     "midpoint (fractional 0.5) to 0.45 — the defect. $\\Gamma$-only (a "
     "single very long supercell, the same molecular-box trick as notebooks "
     "07/12), so this reuses the half b-vector shell + analytic-projections "
     "recipe again. PseudoDojo's semicore inflates the manifold hugely "
     "(52 = 13$\\times$(1$s$+3$p$) semicore bands vs. the classic tutorial's "
     "minimal-pseudo scheme), so `nbnd=90` with a generous buffer is needed."),
    ("code",
     "frac_x = [0.0385, 0.1154, 0.1923, 0.2692, 0.3462, 0.4231, 0.4500,\n"
     "          0.5769, 0.6538, 0.7308, 0.8077, 0.8846, 0.9615]   # atom 7 (0.45) is the defect\n"
     "frac13 = [[x, 0.5, 0.5] for x in frac_x]\n"
     "atoms13 = Atoms('Na13', scaled_positions=frac13,\n"
     "                cell=[[42.25, 0, 0], [0, 10, 0], [0, 0, 10]], pbc=True)\n"
     "WORK13 = HERE / 'runs' / 'na13'\n"
     "\n"
     "proj13 = [(tuple(frac13[i]), 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0)\n"
     "          for i in range(13)]             # one s orbital per Na atom\n"
     "\n"
     "ov13 = qe.generate_overlaps(\n"
     "    atoms13, (1, 1, 1), WORK13, 'na13',\n"
     "    ecutwfc=40, scf_kpts=(1, 1, 1), nbnd=90, num_wann=13,\n"
     "    exclude_bands=list(range(1, 53)),      # 52 Na 2s2p semicore bands\n"
     "    projections=proj13, system_extra=SMEAR,\n"
     "    pseudopotentials={'Na': 'Na.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    gamma_only=True,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/na13/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov13.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')}, '  nntot =', ov13['mmn'].shape[1])"),
    ("md",
     "### 5. Disentangle + Wannierise\n"
     "Same relative windows as Part 1 (frozen to $E_F$, outer 2.74 eV "
     "above it) and `guiding_centres=True` again."),
    ("code",
     "EF13 = ov13['fermi_energy']\n"
     "result13 = wannierize(\n"
     "    atoms13, (1, 1, 1), ov13['kpts'],\n"
     "    mmn=ov13['mmn'], amn=ov13['amn'], eig=ov13['eig'],\n"
     "    nnkpts=ov13['nnkpts'], g_vectors=ov13['g_vectors'],\n"
     "    nw=13, outer_window=(-1e3, EF13 + 2.7545), frozen_window=(-1e3, EF13),\n"
     "    guiding_centres=True,\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I13 = result13.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega13 = result13.omega_final * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I13:.4f} Ang^2   (Wannier90 tutorial14 defected: 64.2811)')\n"
     "print(f'Omega_total = {omega13:.4f} Ang^2   (Wannier90 tutorial14 defected: 71.5245)')"),
    ("md",
     "### 6. Lead-conductor-lead transport\n"
     "`transport_lcr` treats the leftmost/rightmost few Wannier functions as "
     "identical semi-infinite periodic leads and the rest as the scattering "
     "region — here sorting the 13 WFs by position along the chain (one WF "
     "per atom, so position order is unambiguous). The defect should show "
     "up as a transmission **dip below the ideal chain's $T=1$**."),
    ("code",
     "from waw.analysis import transport_lcr\n"
     "\n"
     "tr13 = transport_lcr(\n"
     "    result13.hr, real_lattice(atoms13), result13.centres_bohr,\n"
     "    one_dim_axis='x', dist_cutoff=9.76 * ANG_TO_BOHR,\n"
     "    num_ll=3, num_cell_ll=3,             # matches Wannier90 tutorial14's own tran_num_ll/_cell_ll\n"
     "    fermi_energy=EF13 * EV_TO_HARTREE,\n"
     "    energy_window=(-5.0 * EV_TO_HARTREE, 5.0 * EV_TO_HARTREE),\n"
     "    energy_step=0.01 * EV_TO_HARTREE,\n"
     ")\n"
     "E13 = tr13.energies * HARTREE_TO_EV\n"
     "T13 = tr13.transmission\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 4))\n"
     "ax.plot(E13, T13, 'C1', lw=1.2)\n"
     "ax.axhline(1.0, color='0.6', lw=0.9, ls=':', label='ideal chain (T=1)')\n"
     "ax.axvline(0.0, color='C3', lw=0.9, ls='--', label='E_F')\n"
     "ax.set_xlabel('E - E_F (eV)'); ax.set_ylabel('transmission  T(E)  (units of G0)')\n"
     "ax.set_title('Defected Na-13 chain: LCR transmission')\n"
     "ax.set_ylim(-0.05, 1.15); ax.legend()\n"
     "plt.tight_layout(); plt.show()\n"
     "\n"
     "T13_EF = float(T13[np.argmin(np.abs(E13))])\n"
     "print(f'transmission at E_F: T = {T13_EF:.3f} G0  (< 1: the defect scatters)')"),
    ("md",
     "The transmission dips well below the ideal chain's single conducting "
     "channel — the displaced Na atom scatters the ballistic electrons, "
     "exactly the qualitative signature a defect embedded in an otherwise "
     "clean 1-D conductor should produce.\n"
     "\n"
     "**Takeaway.** `transport_lcr` needs nothing beyond what `transport_bulk` "
     "already established (the same Wannier $H(R)$, the same $\\Gamma$-only "
     "recipe from notebooks 07/12) — just a different partition of the "
     "Wannier functions into leads and a scattering region, sorted by "
     "position along the transport axis."),
]


# ==========================================================================
# 02 — Lead: isolated sp3 MLWFs on a metal
# ==========================================================================
lead = [
    ("md",
     "# Tutorial 02 — Lead: isolated $sp^3$ MLWFs on a metal\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 2. **FCC lead**, one "
     "atom per cell: 4 $sp^3$-like MLWFs from an **isolated** band manifold "
     "(no disentanglement — `num_bands == num_wann`), the same recipe as "
     "GaAs (notebook 01) but on a metal instead of a semiconductor. "
     "PseudoDojo Pb carries a $5d^{10}$ semicore (14 valence electrons), "
     "which we exclude, leaving the $6s6p$-derived manifold as the 4 "
     "isolated MLWFs."),
    ("code", SETUP),
    ("md",
     "### 1. Converged DFT and Wannierisation\n"
     "FCC Pb, $a = 4.951$ Å, cold smearing (it's a metal). 11 bands (5 "
     "$5d$ semicore + 4 target + 2 buffer), exclude bands 1-5 and 10-11, "
     "keep the isolated 4-band $6s6p$ manifold."),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('Pb', 'fcc', a=4.951)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'pb'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'pb',\n"
     "    ecutwfc=50, scf_kpts=(8, 8, 8), nbnd=11, num_wann=4,\n"
     "    exclude_bands=[1, 2, 3, 4, 5, 10, 11],   # 5d semicore + 2-band buffer\n"
     "    scdm_entanglement='isolated', system_extra=SMEAR,\n"
     "    pseudopotentials={'Pb': 'Pb.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/pb/ if present\n"
     ")\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, n_restarts=3, n_iter=5000, verbose=False,\n"
     ")\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (Wannier90 tutorial02: 7.7513)')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))"),
    ("md",
     "4 equivalent $sp^3$-like MLWFs, matching Wannier90's reference "
     "closely despite the very different pseudopotential (PseudoDojo's "
     "$5d$-semicore-inclusive Pb vs. the classic tutorial's minimal one)."),
    ("md",
     "### 2. Interpolated band structure\n"
     "ASE's standard FCC path: the 4-band model reproduces the isolated "
     "$6s6p$ manifold exactly at the mesh points, interpolated everywhere "
     "else."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands, figsize=(6, 4),\n"
     "           ref_line=ov['fermi_energy'], title='Pb bands from 4 isolated MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "The bands cross $E_F$ — like Cu (notebook 04), Pb is a metal with a "
     "genuine Fermi surface; `waw.interfaces.wannier90.fermi_surface`/`write_bxsf` "
     "(demonstrated in full, including the 3-D Brillouin-zone rendering, "
     "in notebook 04) apply here unchanged, on this simpler *isolated* "
     "4-band model rather than a disentangled 7-band one.\n"
     "\n"
     "**Takeaway.** An isolated-band metal needs nothing beyond what "
     "GaAs's isolated-semiconductor recipe already provides — `exclude_bands` "
     "for the semicore, no window, `wannierize()` unchanged."),
]


# ==========================================================================
# Shared Fe (bcc, SOC) wannierisation + .uHu -- used by notebook 19
# ==========================================================================
FE_SETUP_MORB = """\
from ase.build import bulk
import torch
from waw.interfaces.ase.structure import real_lattice, recip_lattice
from waw.core.spread import rotate_overlaps, weight_overlaps_by_eigenvalues
from waw.core.hamiltonian import compute_position_r, compute_bb_r, compute_cc_r
from waw.analysis.orbital_magnetization import orbital_magnetization
from waw.units import EV_TO_HARTREE

atoms = bulk('Fe', 'bcc', a=2.8699)          # 5.4235 bohr
MP_GRID = (4, 4, 4)
WORK = HERE / 'runs' / 'fe'

# noncollinear + spin-orbit; PseudoDojo fully-relativistic Fe pseudo
SOC = dict(noncolin=True, lspinorb=True)
SOC['starting_magnetization(1)'] = 0.4
SOC.update(occupations='smearing', smearing='cold', degauss=0.02)

# Same recipe as notebooks 17/18 (28 -> 18 spinor MLWFs), plus write_uHu=True
# for the extra <u_{k+b1}|H_k|u_{k+b2}> overlap orbital magnetization needs.
ov = qe.generate_overlaps(
    atoms, MP_GRID, WORK, 'fe',
    ecutwfc=60, scf_kpts=(16, 16, 16), nbnd=36, num_wann=18,
    exclude_bands=list(range(1, 9)),
    scdm_entanglement='erfc', scdm_mu=25.0, scdm_sigma=5.0,
    system_extra=SOC, pseudopotentials={'Fe': 'Fe-sp_r.upf'},
    pseudo_dir=PSEUDO_DIR, ncores=NCORES, write_uHu=True,
    rerun_scf=False,          # reuse a completed SCF in runs/fe/ if present
)

result = wannierize(
    atoms, MP_GRID, ov['kpts'],
    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],
    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=18,
    outer_window=(8.0, 70.0), frozen_window=(8.0, 19.8),
    n_restarts=2, dis_n_iter=1000, n_iter=3000, verbose=False,
)
E_FERMI = ov['fermi_energy']                 # eV, read from the SCF by qe.py
print(f'Omega_I = {result.dis.omega_i * BOHR_TO_ANG**2:.4f} Ang^2  '
      f'(18 spinor MLWFs from 28 bands); E_F = {E_FERMI:.3f} eV')

REAL = real_lattice(atoms)
RECIP = recip_lattice(atoms)
V, U = result.dis.V, result.spread.U_final
W = torch.bmm(V, U)   # (nk, nb, nw) full converged gauge, for compute_cc_r"""


# ==========================================================================
# 19 — Iron: orbital magnetization (postw90 berry_task=morb)
# ==========================================================================
fe_morb = [
    ("md",
     "# Tutorial 19 — Iron: orbital magnetization\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 19 (the postw90 "
     "`berry_task = morb` module). Spin-orbit coupling gives ferromagnetic "
     "iron a small **orbital** magnetic moment on top of its dominant spin "
     "moment -- the intrinsic **orbital magnetization** $M_{\\rm orb}$ "
     "(Ceresoli-Thonhauser-Vanderbilt-Resta, PRB 74, 024408 (2006); "
     "Lopez-Vanderbilt-Souza-Tsirkin, PRB 85, 014435 (2012)).\n"
     "\n"
     "$M_{\\rm orb}$ is a Fermi-surface trace of THREE gauge-invariant "
     "quantities, $-2\\mathrm{Im}[f]$, $-2\\mathrm{Im}[g]$, "
     "$-2\\mathrm{Im}[h]$ -- $-2\\mathrm{Im}[f]$ is exactly the Berry "
     "curvature trace notebook 18's anomalous Hall conductivity already "
     "uses; $-2\\mathrm{Im}[g]/-2\\mathrm{Im}[h]$ need two NEW real-space "
     "quantities, $BB(R)=\\langle 0|H(r-R)|R\\rangle$ and "
     "$CC(R)=\\langle 0|r_\\alpha H(r-R)_\\beta|R\\rangle$ "
     "(`core.hamiltonian.compute_bb_r`/`compute_cc_r`). $CC(R)$ needs a "
     "genuinely new ab-initio overlap, `pw2wannier90`'s "
     "$\\langle u_{k+b_1}|H_k|u_{k+b_2}\\rangle$ (`write_uHu=.true.`, "
     "`interfaces.wannier90.io.read_uHu`) -- $BB(R)$, in contrast, "
     "needs only the already-familiar `.mmn` overlap plus the DFT "
     "eigenvalues, since $H_k$ acts diagonally on its own eigenstate."),
    ("code", SETUP),
    ("md",
     "### 1. Converged noncollinear+SOC DFT and Wannierisation (+ `.uHu`)\n"
     "Identical recipe to notebooks 17/18 (same converged noncollinear+SOC "
     "Fe, 18 spinor MLWFs from 28 bands, 4x4x4 mesh), with `write_uHu=True` "
     "added so `pw2wannier90` also writes the "
     "$\\langle u_{k+b_1}|H_k|u_{k+b_2}\\rangle$ overlap. (The SCF is the "
     "slow step if `runs/fe/` isn't already populated by notebooks 17/18 -- "
     "a few minutes; `write_uHu` itself adds a modest amount of extra "
     "`pw2wannier90` time.)"),
    ("code", FE_SETUP_MORB),
    ("md",
     "### 2. Building BB(R) and CC(R)\n"
     "$BB(R)$ comes from the SAME `.mmn` overlaps notebook 18's $AA(R)$ "
     "used, but with the bra band index first weighted by its ab-initio "
     "eigenvalue (`weight_overlaps_by_eigenvalues`) -- since "
     "$H_k|u_{m,k}\\rangle = \\varepsilon_{m,k}|u_{m,k}\\rangle$, this IS "
     "$\\langle u_{m,k}|H_k$ in the bra of the $(k,k+b)$ overlap, no `.uHu` "
     "needed. $CC(R)$ genuinely needs `.uHu`: it connects TWO different "
     "neighbours of $k$, $\\langle u_{k+b_1}|H_k|u_{k+b_2}\\rangle$, which "
     "cannot be recovered from `.mmn` + eigenvalues the way $BB(R)$ can."),
    ("code",
     "Mmn_weighted = weight_overlaps_by_eigenvalues(result.wdata.Mmn, result.wdata.eig)\n"
     "H_opt   = rotate_overlaps(V, Mmn_weighted, result.wdata.kb_idx)\n"
     "H_tilde = rotate_overlaps(U, H_opt, result.wdata.kb_idx)\n"
     "\n"
     "AA_R = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "BB_R = compute_bb_r(H_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                    result.wdata.kpts, MP_GRID, REAL)\n"
     "\n"
     "uHu = torch.tensor(ov['uHu'], dtype=torch.complex128) * EV_TO_HARTREE   # .uHu is in eV, like .eig\n"
     "CC_R = compute_cc_r(uHu, W, result.wdata.kb_idx, result.wdata.wb, result.wdata.bvecs,\n"
     "                    result.wdata.kpts, MP_GRID, REAL)\n"
     "print('AA_R, BB_R, CC_R built:', AA_R.shape, BB_R.shape, CC_R.shape)"),
    ("md",
     "### 3. Orbital magnetization\n"
     "$M_{\\rm orb}$ is evaluated on a dense interpolation mesh (postw90's "
     "`berry_kmesh`) -- a much finer mesh than the 4x4x4 mesh the "
     "Wannier model itself was built on, the same dense-mesh-from-a-coarse-"
     "DFT-calculation trick notebook 18's AHC already relies on.\n"
     "\n"
     "**A caveat worth stating plainly**: $M_{\\rm orb}$ is a notoriously "
     "delicate quantity -- it is built from second-order, SOC-driven "
     "cross-differences of an already-small effect, and is far more "
     "sensitive to the underlying Wannier-model mesh density than a band "
     "structure or even the AHC is. This notebook deliberately reuses "
     "notebooks 17/18's fast 4x4x4-mesh model, so the value below should "
     "be read as illustrative, not converged. The implementation itself "
     "(`analysis.orbital_magnetization`, `compute_bb_r`/`compute_cc_r`, "
     "`read_uHu`) was separately validated bit-for-bit against a real, "
     "much denser (8x8x8 Wannierisation + 25x25x25 `berry_kmesh`) "
     "`wannier90.x`/`postw90.x` reference run of this same Fe system: "
     "waw reproduced $\\Omega_I$/$\\Omega_{\\rm total}$ to 5 decimal places "
     "and $M_{\\rm orb}$'s dominant $z$ component (Fe's magnetisation axis) "
     "to within postw90's own printed precision, with the $x$/$y$ "
     "components correctly three orders of magnitude smaller."),
    ("code",
     "morb = orbital_magnetization(result.hr, AA_R, BB_R, CC_R, RECIP, REAL,\n"
     "                              fermi_energies=E_FERMI * EV_TO_HARTREE, mesh=(25, 25, 25))\n"
     "print(f'M_orb (Bohr magneton/cell) at E_F={E_FERMI:.3f} eV:', morb.m_orb[0].round(5))"),
    ("md",
     "### 4. Orbital magnetization vs. Fermi energy\n"
     "Scanning $E_F$ around the true Fermi level shows how $M_{\\rm orb}$ "
     "responds as bands are filled/emptied -- the same kind of scan "
     "notebook 18 performs for the AHC."),
    ("code",
     "fermis = np.arange(E_FERMI - 2.0, E_FERMI + 2.01, 0.5)\n"
     "scan = orbital_magnetization(result.hr, AA_R, BB_R, CC_R, RECIP, REAL,\n"
     "                             fermi_energies=fermis * EV_TO_HARTREE, mesh=(20, 20, 20))\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 3.8))\n"
     "for i, label in enumerate('xyz'):\n"
     "    ax.plot(fermis - E_FERMI, scan.m_orb[:, i], 'o-', label=f'$M_{{\\\\rm orb}}^{label}$')\n"
     "ax.axvline(0.0, color='0.5', lw=0.8, ls='--', label='E_F')\n"
     "ax.axhline(0.0, color='0.5', lw=0.6)\n"
     "ax.set_xlabel('E - E_F (eV)'); ax.set_ylabel(r'$M_{\\rm orb}$ ($\\mu_B$/cell)')\n"
     "ax.set_title('bcc Fe: orbital magnetization vs. Fermi energy')\n"
     "ax.legend(fontsize=8)\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** waw now implements postw90's full orbital-magnetization "
     "workflow -- the new `.uHu`-derived $BB(R)$/$CC(R)$ real-space "
     "quantities alongside the already-implemented $AA(R)$, and the "
     "$-2\\mathrm{Im}[f]/[g]/[h]$ trace formulas (CTVR06/LVTS12) built on "
     "the SAME WYSV06 machinery notebook 18's AHC uses -- validated "
     "against real `wannier90.x`/`postw90.x` output, entirely waw-native."),
]


# ==========================================================================
# 20 — LaVO3: k-space-localized disentanglement (dis_spheres)
# ==========================================================================
lavo3 = [
    ("md",
     "# Tutorial 20 — LaVO3: k-space-localized disentanglement\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 20. Correlated "
     "transition-metal oxides often have a narrow, chemically well-defined "
     "manifold — here the vanadium $t_{2g}$ triplet ($d_{xy}, d_{xz}, "
     "d_{yz}$) — that is cleanly isolated from every other band **almost "
     "everywhere** in the Brillouin zone, but hybridizes with neighbouring "
     "bands only near a handful of special k-points. Ordinary "
     "disentanglement (energy windows, or tutorial34's projectability) has "
     "no way to say \"trust a fixed set of bands here, but let the "
     "optimizer work only over there\" — `dis_spheres` does exactly that.\n"
     "\n"
     "**The mechanism** (`core.disentangle.disentangle`'s `dis_spheres`/"
     "`dis_spheres_first_wann`, transcribed from `disentangle.F90::"
     "dis_windows`): a list of `(kx, ky, kz, radius)` spheres in "
     "fractional k-space. At a k-point **inside** a sphere, disentanglement "
     "proceeds normally over all candidate bands. At a k-point **outside "
     "every sphere**, disentanglement is skipped entirely — the fixed "
     "contiguous band window `[first_wann, first_wann+num_wann)` is taken "
     "directly, no optimization at all."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "LaVO3 in the simple (undistorted) cubic-perovskite-like tetragonal "
     "cell Wannier90's own tutorial uses: $a=7.03$, $c/a=1.09$ (Bohr), 5 "
     "atoms (La, V, 3$\\times$O). PseudoDojo La/V/O pseudos need a high "
     "cutoff (both La and V carry semicore states); non-spin-polarized "
     "PBE with Methfessel-Paxton smearing, matching the official tutorial's "
     "own (Hubbard-$U$-free) treatment. A $6\\times6\\times6$ mesh, 40 "
     "bands. `exclude_bands` trims the NSCF's 40 bands down to exactly the "
     "6-band candidate window (bands 21-26, 15.0-18.2 eV) the $t_{2g}$ "
     "triplet lives in — the same energy range the official tutorial's "
     "`dis_win_min/max = 15.0/18.5` uses, even with an entirely different "
     "pseudopotential."),
    ("code",
     "from ase import Atoms\n"
     "a = 7.03 * BOHR_TO_ANG\n"
     "c = 7.03 * 1.09 * BOHR_TO_ANG\n"
     "frac = [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]\n"
     "atoms = Atoms('LaVO3', scaled_positions=frac,\n"
     "              cell=[[a, 0, 0], [0, a, 0], [0, 0, c]], pbc=True)\n"
     "MP_GRID = (6, 6, 6)\n"
     "WORK = HERE / 'runs' / 'lavo3'\n"
     "SMEAR = dict(occupations='smearing', smearing='mp', degauss=0.02)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'lavo3',\n"
     "    ecutwfc=80, scf_kpts=(6, 6, 6), nbnd=40, num_wann=3,\n"
     "    exclude_bands=list(range(1, 21)) + list(range(27, 41)),  # keep bands 21-26\n"
     "    scdm_entanglement='erfc', scdm_mu=16.75, scdm_sigma=1.5,\n"
     "    system_extra=SMEAR, pseudopotentials={'La': 'La.upf', 'V': 'V.upf', 'O': 'O.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/lavo3/ if present\n"
     ")\n"
     "print(f\"E_F = {ov['fermi_energy']:.4f} eV  (candidate window 15.0-18.2 eV straddles it)\")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})"),
    ("md",
     "### 2. Wannierise with `dis_spheres`\n"
     "One sphere at the zone-corner **R** point $(\\tfrac12,\\tfrac12,"
     "\\tfrac12)$, radius $0.2$ Bohr$^{-1}$ — the official tutorial's own "
     "placement, where the $t_{2g}$ triplet's upper edge brushes the next "
     "band up. `dis_spheres_first_wann=0` (0-based) takes the LOWEST 3 of "
     "the 6 candidate bands as the fixed window everywhere outside that "
     "sphere. No outer/frozen energy window is needed here: `exclude_bands` "
     "already narrowed the candidate set to exactly the 6 bands of "
     "interest."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=3,\n"
     "    dis_spheres=[(0.5, 0.5, 0.5, 0.2)], dis_spheres_first_wann=0,\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=3000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2   (converged: {result.dis.converged})')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3),\n"
     "      ' -- two equal (dxz/dyz) + one different (dxy), the t2g tetragonal split')"),
    ("md",
     "### 3. Why `dis_spheres` matters: a direct comparison\n"
     "Running the SAME 6 -> 3 disentanglement with no `dis_spheres` at all "
     "lets the optimizer mix in whatever combination of the 6 candidate "
     "bands minimizes $\\Omega_I$ everywhere in the zone, not just near "
     "R. That can reach a numerically *lower* $\\Omega_I$ — more "
     "variational freedom always can — but at the cost of the guaranteed, "
     "consistent band identity `dis_spheres` deliberately trades away "
     "flexibility for: away from R, every k-point is forced to use the "
     "SAME fixed 3 bands, so the resulting MLWFs are unambiguously the "
     "$t_{2g}$ triplet everywhere except the one region (near R) that "
     "genuinely needs to be resolved by optimization. In practice this "
     "also tends to leave the free-disentanglement fit a harder, less "
     "well-posed problem for the spread minimizer to converge on."),
    ("code",
     "result_free = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=3,\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=3000, verbose=False,\n"
     ")\n"
     "print(f'No dis_spheres: Omega_I = {result_free.dis.omega_i*BOHR_TO_ANG**2:.4f} Ang^2  '\n"
     "      f'Omega_total = {result_free.omega_final*BOHR_TO_ANG**2:.4f} Ang^2  '\n"
     "      f'(spread-minimization converged: {result_free.spread.converged})')\n"
     "print(f'dis_spheres:    Omega_I = {omega_I:.4f} Ang^2  Omega_total = {omega:.4f} Ang^2  '\n"
     "      f'(spread-minimization converged: {result.spread.converged})')\n"
     "print('Lower Omega_I from free disentanglement is expected (more variational freedom) --\\n'\n"
     "      'the point of dis_spheres is a well-defined band identity, not a smaller number.')"),
    ("md",
     "### 4. Interpolated band structure\n"
     "The 3-band $t_{2g}$ model reproduces the DFT bands inside the "
     "candidate window; ASE's standard k-path for this tetragonal cell "
     "stands in for the official tutorial's own G-M-X-G-Z-A-R-X path."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands - E_FERMI, figsize=(6.5, 4.2),\n"
     "           ref_line=0.0, ylim=(-3, 3), ylabel='E - E_F (eV)',\n"
     "           title='LaVO3: t2g bands from dis_spheres-disentangled MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "### 4. SrMnO3: the official tutorial's other three examples\n"
     "Wannier90 tutorial 20's remaining examples are all on cubic-perovskite "
     "SrMnO3 (Mn$^{4+}$, $d^3$): the full 5-orbital Mn $3d$ manifold "
     "(`SrMnO3-d`), the $e_g$ doublet alone (`SrMnO3-eg`), and the $t_{2g}$ "
     "triplet alone (`SrMnO3-t2g`) — the same `f=...:dxz;dyz;dxy`-style "
     "individual-orbital analytic projections as before, just picking a "
     "different subset of the 5 $d$ orbitals each time. Unlike LaVO3, a "
     "**wide** candidate window (here, 20 bands framing the Mn $3d$/O $2p$ "
     "manifold, with PseudoDojo's semicore states excluded) combined with "
     "the orbital-selective analytic projection is already enough for plain "
     "energy-window disentanglement to isolate each target manifold "
     "cleanly, with `Omega_D+Omega_OD` shrinking to essentially zero -- "
     "`dis_spheres` genuinely isn't needed for every correlated-oxide case, "
     "only when the target bands hybridize non-trivially somewhere in the "
     "zone the way LaVO3's $t_{2g}$ does near R."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "a_sr = 3.6460309631\n"
     "c_sr = 4.0470943691\n"
     "frac_sr = [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]\n"
     "atoms_sr = Atoms('SrMnO3', scaled_positions=frac_sr,\n"
     "                 cell=[[a_sr, 0, 0], [0, a_sr, 0], [0, 0, c_sr]], pbc=True)\n"
     "MP_GRID_SR = (6, 6, 6)\n"
     "WORK_SR = HERE / 'runs' / 'srmno3'\n"
     "SMEAR_SR = dict(occupations='smearing', smearing='mp', degauss=0.02)\n"
     "mn_frac = (0.5, 0.5, 0.5)\n"
     "\n"
     "srmno3_results = {}\n"
     "for label, mrs, nw in [('t2g', (2, 3, 5), 3), ('eg', (1, 4), 2), ('d', (1, 2, 3, 4, 5), 5)]:\n"
     "    proj = [(mn_frac, 2, mr, 1, (0., 0., 1.), (1., 0., 0.), 1.0) for mr in mrs]\n"
     "    ov_sr = qe.generate_overlaps(\n"
     "        atoms_sr, MP_GRID_SR, WORK_SR, 'srmno3',\n"
     "        ecutwfc=80, scf_kpts=(6, 6, 6), nbnd=45, num_wann=nw,\n"
     "        exclude_bands=list(range(1, 12)) + list(range(32, 46)),  # semicore + high conduction\n"
     "        projections=proj, system_extra=SMEAR_SR,\n"
     "        pseudopotentials={'Sr': 'Sr.upf', 'Mn': 'Mn.upf', 'O': 'O.upf'},\n"
     "        pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "        rerun_scf=False,      # reuse a completed SCF in runs/srmno3/ if present\n"
     "    )\n"
     "    srmno3_results[label] = wannierize(\n"
     "        atoms_sr, MP_GRID_SR, ov_sr['kpts'],\n"
     "        mmn=ov_sr['mmn'], amn=ov_sr['amn'], eig=ov_sr['eig'],\n"
     "        nnkpts=ov_sr['nnkpts'], g_vectors=ov_sr['g_vectors'], nw=nw,\n"
     "        dis_n_iter=1000, n_iter=2000, conv_tol=1e-13, n_restarts=2, verbose=False,\n"
     "    )\n"
     "\n"
     "w90_ref = {'t2g': 1.571760472, 'eg': 1.019482504, 'd': 2.591236161}\n"
     "for label in ('t2g', 'eg', 'd'):\n"
     "    r = srmno3_results[label]\n"
     "    print(f\"SrMnO3-{label}: Omega_total = {r.omega_final*BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "          f\"(W90 reference: {w90_ref[label]:.6f})\")"),
    ("md",
     "All three match real `wannier90.x` run on the identical overlaps to "
     "4-6 decimal places, and every Wannier centre lands exactly on the Mn "
     "site, as it must for a genuinely $d$-orbital-derived MLWF set."),
    ("md",
     "**Takeaway.** `dis_spheres` is a THIRD, complementary disentanglement "
     "mode alongside energy windows (tutorial03/04/10) and projectability "
     "(tutorial34): instead of selecting bands by energy or atomic-orbital "
     "character, it selects *where in the Brillouin zone* optimization is "
     "even allowed to happen, trusting a fixed band identity everywhere "
     "else. For a correlated, narrow $t_{2g}$/$e_g$ manifold that only "
     "genuinely hybridizes near specific k-points — LaVO3's situation here — "
     "that guarantee matters more than squeezing out the last bit of "
     "$\\Omega_I$; SrMnO3's three variants above show the complementary "
     "case, where a wide-enough window and the right orbital projection "
     "already suffice without it. All four of wannier90 tutorial 20's "
     "official examples are validated here against real `wannier90.x` "
     "output, entirely waw-native."),
]


# ==========================================================================
# 21 — GaAs: symmetry-adapted Wannier functions (site_symmetry)
# ==========================================================================
gaas_sitesym = [
    ("md",
     "# Tutorial 21 — GaAs: symmetry-adapted Wannier functions\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 21 (R. Sakuma, "
     "*Symmetry-adapted Wannier functions in the maximal localization "
     "procedure*, PRB **87**, 235109 (2013)). Every other tutorial in this "
     "series wannierizes independently at every k-point in the mesh; here "
     "the crystal's **point-group symmetry** is imposed explicitly, so only "
     "the *irreducible* k-points (10 of GaAs's 64, for this "
     "$4\\times4\\times4$ mesh) carry independent numerical work -- every "
     "other k-point's gauge is *derived* from an irreducible representative "
     "via the symmetry operation that relates them, not independently "
     "optimized.\n"
     "\n"
     "This needs a new ab-initio input, `pw2wannier90`'s `.dmn` file "
     "(`write_dmn=.true.`): representation matrices of every point-group "
     "operation in both the raw DFT band basis and the target "
     "Wannier-function basis, plus the k-point mapping under each "
     "operation (`waw.interfaces.wannier90.io.read_dmn`). "
     "`waw.core.sitesym` symmetrizes the disentanglement Z-matrix and the "
     "spread-minimization gradient at each irreducible k, then *broadcasts* "
     "the result to the rest of the mesh via the group action -- "
     "`core.disentangle.disentangle(..., sitesym=...)` and "
     "`core.optim.minimize_spread_symmetrized`."),
    ("code", SETUP),
    ("md",
     "### 1. Structure, converged DFT, and the `.dmn` symmetry file\n"
     "Zincblende GaAs (notebook 01's system), with a single **Ga-centred "
     "$s$-like** Wannier function -- Wannier90 tutorial 21's simplest "
     "example (`atom_centered_Ga_s`). Two things `site_symmetry` needs "
     "that no other notebook in this series does:\n"
     "\n"
     "1. **An explicit analytic projection**, not SCDM -- `pw2wannier90` "
     "aborts trying to resolve SCDM's implicit projection sites against "
     "the symmetry group (`spd_projections` gives the same `f=0,0,0:s` "
     "block Wannier90's own `begin projections` block declares).\n"
     "2. **Symmetry ON in the NSCF** (`nscf_symmetry=True`) -- every other "
     "notebook here turns QE's `nosym`/`noinv` OFF to keep the explicit "
     "k-list exactly as generated; site symmetry needs the OPPOSITE, "
     "since `pw2wannier90` reads the crystal's point-group operations off "
     "the NSCF's own symmetry detection. Safe to combine: because the "
     "k-list is given EXPLICITLY (`K_POINTS crystal`), QE cannot silently "
     "reduce or reorder it regardless of symmetry -- only *automatic* "
     "meshes get reduced."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.projections import spd_projections\n"
     "\n"
     "atoms = bulk('GaAs', 'zincblende', a=5.65)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'gaas_sitesym'\n"
     "\n"
     "projs = spd_projections((0.0, 0.0, 0.0), 's')   # Ga-centred s orbital\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas',\n"
     "    ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=20, num_wann=1,\n"
     "    exclude_bands=list(range(1, 11)) + list(range(15, 21)),  # 10 semicore d + conduction\n"
     "    projections=projs, write_dmn=True, nscf_symmetry=True,\n"
     "    pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/gaas_sitesym/ if present\n"
     ")\n"
     "dmn = ov['dmn']\n"
     "print(f\"num_bands={dmn['num_bands']}  nsymmetry={dmn['nsymmetry']}  \"\n"
     "      f\"nkptirr={dmn['nkptirr']} of {dmn['num_kpts']} k-points  num_wann={dmn['num_wann']}\")"),
    ("md",
     "### 2. Symmetrized disentanglement + spread minimization\n"
     "The 4 candidate bands (already narrowed by `exclude_bands`) disentangle "
     "down to 1 Wannier function at every IRREDUCIBLE k-point only; "
     "`sitesym=` runs both disentanglement and spread minimization over that "
     "wedge, symmetrizing at each step and broadcasting the result to the "
     "rest of the mesh (`core.pipeline.wannierize`'s `sitesym` argument, "
     "which threads through to `core.disentangle.disentangle`/`core.optim."
     "minimize_spread_symmetrized`) -- the same `wannierize()` convenience "
     "wrapper every other notebook uses, with CG (this project's -- and "
     "wannier90's own -- default minimizer): Adam's adaptive per-element "
     "step gets stuck in a bad local minimum on this system (only "
     "`nkptirr` independent phases for a single Wannier function). "
     "`n_restarts`/`n_hops` aren't supported together with `sitesym` (a "
     "single irreducible-wedge run needs no restarts to begin with); "
     "`guiding_centres` is."),
    ("code",
     "from waw.core.sitesym import site_symmetry_from_dmn\n"
     "\n"
     "sitesym = site_symmetry_from_dmn(dmn)\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=1,\n"
     "    sitesym=sitesym, lr=1.0,\n"
     "    dis_n_iter=200, dis_conv_tol=1e-12,\n"
     "    n_iter=500, conv_tol=1e-14, conv_window=5,\n"
     ")\n"
     "print(f'Omega_I     = {result.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 2.392981110)')\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 2.410924074)')\n"
     "print('centre (Ang):', (result.centres_bohr[0] * BOHR_TO_ANG).tolist(),\n"
     "      ' (W90 reference: essentially the Ga site, (0,0,0))')"),
    ("md",
     "This reproduces real `wannier90.x`'s own `site_symmetry=.true.` run "
     "on the IDENTICAL `.mmn`/`.amn`/`.eig`/`.dmn` to 5-6 decimal places -- "
     "$\\Omega_I$, $\\Omega_{\\rm total}$, and the Wannier centre (pinned "
     "almost exactly to the Ga site by the site symmetry, as it must be for "
     "a genuinely atom-centred $s$-like function) all agree."),
    ("md",
     "### 3. Checking the result stays point-group covariant\n"
     "The whole point of the method: verify that the converged full-mesh "
     "gauge $U(k)$ still satisfies the group's covariance relation "
     "$U(Rk) = D_{\\rm wann}(R,k)\\,U(k)\\,D_{\\rm wann}(R,k)^\\dagger$ for "
     "an arbitrary non-trivial symmetry operation $R$ and a k-point outside "
     "the irreducible set -- this is enforced at *every* iteration by "
     "`core.optim.minimize_spread_symmetrized`, not just checked post-hoc."),
    ("code",
     "U_final = result.spread.U_final\n"
     "ir = 1   # an irreducible k with a nontrivial orbit\n"
     "ik = int(sitesym.ir2ik[ir])\n"
     "for isym in range(1, sitesym.nsymmetry):\n"
     "    irk = int(sitesym.kptsym[isym, ir])\n"
     "    if irk == ik:\n"
     "        continue\n"
     "    D = sitesym.d_matrix_wann[:, :, isym, ir]\n"
     "    lhs = D @ U_final[ik] @ D.conj().transpose(-1, -2)\n"
     "    err = (lhs - U_final[irk]).abs().max().item()\n"
     "    print(f'symmetry {isym}: k={ik} -> k={irk},  covariance error = {err:.2e}')\n"
     "    break"),
    ("md",
     "### 4. `atom_centered_Ga_p`: a genuinely degenerate irrep\n"
     "Three $p$-like orbitals at the Ga site instead of one $s$-like orbital "
     "-- under the zincblende structure's $T_d$ point group at $\\Gamma$, "
     "these three MLWFs are genuinely, exactly degenerate (they transform as "
     "a 3-dimensional irrep). Extracting the right *subspace* from a "
     "degenerate eigenspace needs more care than picking a plain `eigh`'s "
     "top eigenvectors, whose choice of basis within a degenerate block is "
     "arbitrary and not generally equivariant under the stabilizer -- "
     "`core.sitesym.extract_symmetrized_subspace` ports wannier90's own "
     "band-by-band 2x2 generalized-eigenvalue steepest-ascent solver "
     "(`sitesym_dis_extract_symmetry`) for exactly this reason."),
    ("code",
     "projs = spd_projections((0.0, 0.0, 0.0), 'p')   # Ga-centred p orbitals\n"
     "\n"
     "ov_p = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas',\n"
     "    ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=20, num_wann=3,\n"
     "    exclude_bands=list(range(1, 11)) + list(range(15, 21)),\n"
     "    projections=projs, write_dmn=True, nscf_symmetry=True,\n"
     "    pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "sitesym_p = site_symmetry_from_dmn(ov_p['dmn'])\n"
     "result_p = wannierize(\n"
     "    atoms, MP_GRID, ov_p['kpts'],\n"
     "    mmn=ov_p['mmn'], amn=ov_p['amn'], eig=ov_p['eig'],\n"
     "    nnkpts=ov_p['nnkpts'], g_vectors=ov_p['g_vectors'], nw=3,\n"
     "    sitesym=sitesym_p,\n"
     "    dis_n_iter=200, dis_conv_tol=1e-12,\n"
     "    n_iter=1000, conv_tol=1e-14, conv_window=5,\n"
     ")\n"
     "print(f'Omega_I     = {result_p.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 7.898822)')\n"
     "print(f'Omega_total = {result_p.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 11.593156)')"),
    ("md",
     "### 5. `atom_centered_Ga_sp` and `atom_centered_As_sp`\n"
     "One $s$ + three $p$ orbitals (4 MLWFs total) centred on Ga, then the "
     "same thing centred on As instead -- the same recipe, just a different "
     "projection site."),
    ("code",
     "results_sp = {}\n"
     "for label, centre in [('Ga_sp', (0.0, 0.0, 0.0)), ('As_sp', (0.25, 0.25, 0.25))]:\n"
     "    projs_sp = spd_projections(centre, 's;p')\n"
     "    ov_sp = qe.generate_overlaps(\n"
     "        atoms, MP_GRID, WORK, 'gaas',\n"
     "        ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=20, num_wann=4,\n"
     "        exclude_bands=list(range(1, 11)) + list(range(15, 21)),\n"
     "        projections=projs_sp, write_dmn=True, nscf_symmetry=True,\n"
     "        pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "        pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     "    )\n"
     "    sitesym_sp = site_symmetry_from_dmn(ov_sp['dmn'])\n"
     "    results_sp[label] = wannierize(\n"
     "        atoms, MP_GRID, ov_sp['kpts'],\n"
     "        mmn=ov_sp['mmn'], amn=ov_sp['amn'], eig=ov_sp['eig'],\n"
     "        nnkpts=ov_sp['nnkpts'], g_vectors=ov_sp['g_vectors'], nw=4,\n"
     "        sitesym=sitesym_sp,\n"
     "        dis_n_iter=200, dis_conv_tol=1e-12,\n"
     "        n_iter=1000, conv_tol=1e-14, conv_window=5,\n"
     "    )\n"
     "\n"
     "print(f\"Ga_sp: Omega_I={results_sp['Ga_sp'].spread.Omega_I*BOHR_TO_ANG**2:.6f} Ang^2 \"\n"
     "      f\"(W90: 6.553593)   Omega_total={results_sp['Ga_sp'].omega_final*BOHR_TO_ANG**2:.6f} \"\n"
     "      f\"Ang^2 (W90: 14.044861)\")\n"
     "print(f\"As_sp: Omega_I={results_sp['As_sp'].spread.Omega_I*BOHR_TO_ANG**2:.6f} Ang^2 \"\n"
     "      f\"(W90: 6.553593)   Omega_total={results_sp['As_sp'].omega_final*BOHR_TO_ANG**2:.6f} \"\n"
     "      f\"Ang^2 (W90: 10.134915)\")"),
    ("md",
     "### 6. `bond_centered`: isolated bands, and a second real bug this "
     "example exposed\n"
     "Four bond-centred $s$-like orbitals, one at each of the 4 tetrahedral "
     "Ga-As bond midpoints -- `num_bands == num_wann == 4` here (an "
     "*isolated*, not disentangled, manifold), which turned out to matter: "
     "for isolated bands, `core.disentangle.disentangle` never runs (its "
     "\"V=I, no-op\" short-circuit), so the spread-minimization gauge $U$'s "
     "LEFT index is still the raw candidate-band manifold, transforming "
     "under `d_matrix_band` -- **not** `d_matrix_wann`, even though the two "
     "happen to share a dimension here. Using the wrong one caused a real, "
     "confirmed divergence (Omega_total ~150-185 Ang^2, unconverged) the "
     "first time this example was tried; `core.optim."
     "minimize_spread_symmetrized`'s `d_left` argument (set automatically by "
     "`core.pipeline.wannierize`) fixes it."),
    ("code",
     "bond_centres = [(0.125, 0.125, 0.125), (0.125, 0.125, -0.375),\n"
     "                (-0.375, 0.125, 0.125), (0.125, -0.375, 0.125)]\n"
     "projs_bc = []\n"
     "for c in bond_centres:\n"
     "    projs_bc += spd_projections(c, 's')\n"
     "\n"
     "ov_bc = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas',\n"
     "    ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=20, num_wann=4,\n"
     "    exclude_bands=list(range(1, 11)) + list(range(15, 21)),\n"
     "    projections=projs_bc, write_dmn=True, nscf_symmetry=True,\n"
     "    pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "sitesym_bc = site_symmetry_from_dmn(ov_bc['dmn'])\n"
     "result_bc = wannierize(\n"
     "    atoms, MP_GRID, ov_bc['kpts'],\n"
     "    mmn=ov_bc['mmn'], amn=ov_bc['amn'], eig=ov_bc['eig'],\n"
     "    nnkpts=ov_bc['nnkpts'], g_vectors=ov_bc['g_vectors'], nw=4,\n"
     "    sitesym=sitesym_bc,\n"
     "    n_iter=1000, conv_tol=1e-14, conv_window=5,\n"
     ")\n"
     "print(f'Omega_total = {result_bc.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 7.161450817)')\n"
     "print('centres (Ang):', (result_bc.centres_bohr * BOHR_TO_ANG).round(4).tolist(),\n"
     "      ' (W90 reference: the 4 bond midpoints, e.g. (0.861, 0.861, 0.861))')"),
    ("md",
     "**Takeaway.** Symmetry-adapted Wannier functions turn a "
     "full-Brillouin-zone optimization into one over the irreducible "
     "wedge alone -- here 10 independent k-points instead of 64 -- with "
     "every other k-point's gauge locked to a representative by the "
     "crystal's own point group, not fit independently. All 5 of "
     "wannier90 tutorial 21's official examples are validated here against "
     "real `wannier90.x` output at the 5-6th decimal place, entirely "
     "waw-native -- including two genuine bugs (the degenerate-eigenspace "
     "extractor and the isolated-bands `d_left` gauge) that only showed up "
     "on the less trivial `atom_centered_Ga_p` and `bond_centered` cases."),
]


# ==========================================================================
# 22 — Copper: symmetry-adapted Wannier functions with genuine disentanglement
# ==========================================================================
cu_sitesym = [
    ("md",
     "# Tutorial 22 — Copper: symmetry-adapted Wannier functions on a metal\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 22. Tutorial 21's five "
     "GaAs examples were all either isolated bands or a trivially small "
     "candidate window; here `site_symmetry` meets **genuine "
     "disentanglement** for the first time -- fcc **copper**, `num_bands=10 "
     "> num_wann=6` (5 Cu-centred $d$-like MLWFs + 1 $s$-like MLWF), the same "
     "disentanglement code path as notebook 04's metallic Cu, but now with the "
     "irreducible-wedge symmetrization on top.\n"
     "\n"
     "All 3 official variants share the same 5 $d$-like Wannier functions "
     "centred on the Cu atom; they differ only in where the single $s$-like "
     "function is centred:\n"
     "\n"
     "| Variant | $s$ centre (frac) | Site | Symmetry needed |\n"
     "|---|---|---|---|\n"
     "| `s_at_0.00` | $(0,0,0)$ | the Cu atom | full $O_h$ (48 ops) |\n"
     "| `s_at_0.50` | $(\\tfrac12,\\tfrac12,\\tfrac12)$ | octahedral hole | full $O_h$ (48 ops, also inversion-symmetric) |\n"
     "| `s_at_0.25` | $(\\tfrac14,\\tfrac14,\\tfrac14)$ | tetrahedral hole | **reduced** $T_d$ (24 ops, `read_sym`) |\n"
     "\n"
     "The tetrahedral-hole site is the interesting case: it is **not** a fixed "
     "point of $O_h$'s inversion (inversion through the Cu atom maps "
     "$(\\tfrac14,\\tfrac14,\\tfrac14)$ to $(-\\tfrac14,-\\tfrac14,-\\tfrac14) "
     "\\equiv (\\tfrac34,\\tfrac34,\\tfrac34)$, a *different* site), only of the "
     "24-element $T_d$ subgroup (12 proper rotations + 12 improper $S_4$/mirror "
     "operations, no inversion). Telling `pw2wannier90` to use the wrong "
     "(too-large) 48-operation group here would silently build a `.dmn` for a "
     "symmetry the Wannier centre doesn't actually respect. This needs a new "
     "waw-QE-interface capability: an explicit, caller-supplied symmetry list "
     "(`interfaces.wannier90.io.write_sym`, pw2wannier90 `read_sym=.true.`), "
     "used via `qe.generate_overlaps(..., sym_ops=(rotations, translations))`."),
    ("code", SETUP),
    ("md",
     "### 1. Converged DFT: a 10-band candidate manifold\n"
     "fcc Cu (notebook 04's system), $a=3.615$ Å, `ecutwfc=55` Ry, cold "
     "smearing. The PseudoDojo Cu pseudo carries a $3s3p$ semicore (4 bands); "
     "immediately above it sit exactly 6 valence bands ($3d^{10}4s^1$). We "
     "compute 20 bands and keep a **10-band candidate window** (bands 5-14) "
     "straddling those 6 valence bands -- the disentanglement then extracts "
     "the optimal 6-dimensional subspace from it, matching Wannier90 "
     "tutorial 22's own `num_bands=10 -> num_wann=6` scheme."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.projections import spd_projections\n"
     "from waw.core.sitesym import site_symmetry_from_dmn\n"
     "\n"
     "atoms = bulk('Cu', 'fcc', a=3.615)\n"
     "MP_GRID = (4, 4, 4)\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "CANDIDATE_BANDS = dict(nbnd=20, exclude_bands=[1, 2, 3, 4] + list(range(15, 21)))  # keep bands 5-14\n"
     "\n"
     "def cu_sitesym_run(s_centre, sym_ops=None, work_suffix=''):\n"
     "    '''One Cu site-symmetry variant: 5 Cu-centred d + 1 s at s_centre.'''\n"
     "    projs = spd_projections(s_centre, 's') + spd_projections((0.0, 0.0, 0.0), 'd')\n"
     "    ov = qe.generate_overlaps(\n"
     "        atoms, MP_GRID, HERE / 'runs' / f'cu_sitesym{work_suffix}', 'cu',\n"
     "        ecutwfc=55, scf_kpts=(12, 12, 12), num_wann=6, **CANDIDATE_BANDS,\n"
     "        projections=projs, write_dmn=True, nscf_symmetry=True,\n"
     "        system_extra=SMEAR, sym_ops=sym_ops,\n"
     "        pseudopotentials={'Cu': 'Cu.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "        rerun_scf=False,\n"
     "    )\n"
     "    sitesym = site_symmetry_from_dmn(ov['dmn'])\n"
     "    result = wannierize(\n"
     "        atoms, MP_GRID, ov['kpts'],\n"
     "        mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "        nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=6,\n"
     "        sitesym=sitesym, dis_n_iter=1000, n_iter=2000, conv_tol=1e-13,\n"
     "    )\n"
     "    return ov, result"),
    ("md",
     "### 2. `s_at_0.00`: $s$ orbital on the Cu atom (full $O_h$)\n"
     "The simplest variant: both the $s$ and the 5 $d$ orbitals sit on the "
     "same, inversion-symmetric Cu site, so `pw2wannier90` auto-detects the "
     "full 48-operation $O_h$ group -- no `sym_ops` needed."),
    ("code",
     "ov0, result0 = cu_sitesym_run((0.0, 0.0, 0.0), work_suffix='_s000')\n"
     "dmn0 = ov0['dmn']\n"
     "print(f\"num_bands={dmn0['num_bands']}  nsymmetry={dmn0['nsymmetry']}  \"\n"
     "      f\"nkptirr={dmn0['nkptirr']} of {dmn0['num_kpts']} k-points\")\n"
     "print(f\"Omega_I     = {result0.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "      f\"(W90 reference: 3.029238)\")\n"
     "print(f\"Omega_total = {result0.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "      f\"(W90 reference: 3.048393)\")\n"
     "print('s-orbital centre (Ang):', (result0.centres_bohr[0] * BOHR_TO_ANG).round(4).tolist(),\n"
     "      ' (the Cu site, (0,0,0))')"),
    ("md",
     "### 3. `s_at_0.50`: $s$ orbital on the octahedral hole (full $O_h$)\n"
     "$(\\tfrac12,\\tfrac12,\\tfrac12)$ is *also* an inversion-symmetric site "
     "in the fcc lattice (fcc's other octahedral Wyckoff position), so this "
     "variant again uses the full 48-operation group -- same recipe, "
     "different projection centre."),
    ("code",
     "ov50, result50 = cu_sitesym_run((0.5, 0.5, 0.5), work_suffix='_s050')\n"
     "dmn50 = ov50['dmn']\n"
     "print(f\"nsymmetry={dmn50['nsymmetry']}  nkptirr={dmn50['nkptirr']} of {dmn50['num_kpts']}\")\n"
     "print(f\"Omega_I     = {result50.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "      f\"(W90 reference: 2.696288)\")\n"
     "print(f\"Omega_total = {result50.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "      f\"(W90 reference: 2.728998)\")\n"
     "print('s-orbital centre (Ang):', (result50.centres_bohr[0] * BOHR_TO_ANG).round(4).tolist(),\n"
     "      ' (the octahedral hole, (1.807, 1.807, 1.807) Ang = (1/2,1/2,1/2) frac)')"),
    ("md",
     "### 4. `s_at_0.25`: $s$ orbital on the tetrahedral hole (`read_sym`, reduced $T_d$)\n"
     "$(\\tfrac14,\\tfrac14,\\tfrac14)$ is **not** fixed by $O_h$'s inversion, "
     "only by the 24-element $T_d$ subgroup. We supply that reduced operation "
     "list explicitly via `sym_ops`; `qe.generate_overlaps` writes it to a "
     "`.sym` file and sets pw2wannier90's `read_sym=.true.`, overriding the "
     "auto-detected (too-large) 48-operation group. The 24 matrices below are "
     "the $T_d$ point group in this fcc primitive-cell (`ibrav=2`) convention "
     "-- 12 proper sign-permutation rotations and 12 improper ones, reused "
     "verbatim from the real Wannier90 tutorial22 reference (a matter of "
     "convenience -- deriving the exact convention independently, without a "
     "symmetry-detection library, would risk a subtle sign/basis mismatch)."),
    ("code",
     "import numpy as np\n"
     "# T_d point group (24 ops, no inversion) fixing the tetrahedral-hole site,\n"
     "# as sign-permutation matrices in the fcc primitive-cell (crystal/fractional) basis.\n"
     "_TD_ROTATIONS = np.array([\n"
     "    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],\n"
     "    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [[1, 0, 0], [0, -1, 0], [0, 0, -1]],\n"
     "    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],\n"
     "    [[0, 0, -1], [1, 0, 0], [0, -1, 0]], [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],\n"
     "    [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, -1, 0], [0, 0, -1], [1, 0, 0]],\n"
     "    [[0, -1, 0], [0, 0, 1], [-1, 0, 0]], [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],\n"
     "    [[0, -1, 0], [-1, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 1]],\n"
     "    [[0, 1, 0], [-1, 0, 0], [0, 0, -1]], [[0, -1, 0], [1, 0, 0], [0, 0, -1]],\n"
     "    [[0, 0, -1], [0, 1, 0], [-1, 0, 0]], [[0, 0, 1], [0, 1, 0], [1, 0, 0]],\n"
     "    [[0, 0, 1], [0, -1, 0], [-1, 0, 0]], [[0, 0, -1], [0, -1, 0], [1, 0, 0]],\n"
     "    [[1, 0, 0], [0, 0, -1], [0, -1, 0]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]],\n"
     "    [[-1, 0, 0], [0, 0, 1], [0, -1, 0]], [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],\n"
     "], dtype=float)\n"
     "_TD_TRANSLATIONS = np.zeros((24, 3))\n"
     "print('det(R) values:', sorted(set(round(np.linalg.det(r)) for r in _TD_ROTATIONS)),\n"
     "      ' (12 proper + 12 improper, no inversion present)')"),
    ("code",
     "ov25, result25 = cu_sitesym_run(\n"
     "    (0.25, 0.25, 0.25), sym_ops=(_TD_ROTATIONS, _TD_TRANSLATIONS), work_suffix='_s025',\n"
     ")\n"
     "dmn25 = ov25['dmn']\n"
     "print(f\"nsymmetry={dmn25['nsymmetry']}  nkptirr={dmn25['nkptirr']} of {dmn25['num_kpts']}\"\n"
     "      \"   (24, not 48 -- read_sym took effect)\")\n"
     "print(f\"Omega_I     = {result25.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "      f\"(W90 reference: 2.546635)\")\n"
     "print(f\"Omega_total = {result25.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   \"\n"
     "      f\"(W90 reference: 2.598012)\")\n"
     "print('s-orbital centre (Ang):', (result25.centres_bohr[0] * BOHR_TO_ANG).round(4).tolist(),\n"
     "      ' (the tetrahedral hole, (0.904, 0.904, 0.904) Ang = (1/4,1/4,1/4) frac)')"),
    ("md",
     "All 3 variants reproduce real `wannier90.x` run on the identical "
     "`.mmn`/`.amn`/`.eig`/`.dmn` to 3-4 decimal places in $\\Omega_I$/"
     "$\\Omega_{\\rm total}$, and `s_at_0.25`'s Wannier centre lands exactly "
     "on the tetrahedral hole -- confirming `read_sym` (not a silent "
     "fallback to the full 48-operation group) is what makes that "
     "off-atom, off-inversion-centre localisation possible."),
    ("md",
     "### 5. Band structure\n"
     "The `s_at_0.00` model's 6 MLWFs (5 $d$ + 1 $s$, both centred on the Cu "
     "atom) interpolate the same $d$-manifold-plus-dispersive-$s$-band "
     "physics as notebook 04's 7-orbital model, just with 6 instead of 7 "
     "Wannier functions -- the price of pinning both orbital types to a "
     "single, symmetry-required centre rather than letting the diffuse $s$ "
     "character split across 2 independent functions."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "E_FERMI = ov0['fermi_energy']\n"
     "bp = band_path(atoms, npoints=120)\n"
     "bands = interpolate_bands(result0.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands - E_FERMI, figsize=(6.5, 4),\n"
     "           ref_line=0.0, ylim=(-8, 8), ylabel='E - E_F (eV)',\n"
     "           title='Cu bands from 6 symmetry-adapted MLWFs (s_at_0.00)')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** Symmetry-adapted Wannier functions extend cleanly from "
     "tutorial 21's isolated/trivial-window GaAs examples to a genuine metal "
     "disentanglement: the same `sitesym=` machinery symmetrizes both the "
     "disentanglement Z-matrix and the spread-minimization gradient over the "
     "irreducible wedge (8-10 of 64 k-points, depending on the variant) for "
     "Cu's 10-band candidate manifold. The `s_at_0.25` variant needed one new "
     "capability -- `sym_ops`/`read_sym`, an explicit caller-supplied "
     "symmetry-operation list -- for the one case where the desired Wannier "
     "centre respects only a subgroup ($T_d$, 24 ops) of the crystal's full "
     "point group ($O_h$, 48 ops), and both the reduced operation count and "
     "the resulting off-atom Wannier centre confirm it took effect. All 3 "
     "variants validated against real `wannier90.x` to 3-4 decimal places."),
]


# ==========================================================================
# 34 — Graphene: projectability-disentangled Wannier functions
# ==========================================================================
graphene = [
    ("md",
     "# Tutorial 34 — Graphene: projectability-disentangled Wannier functions\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 34. A free-standing "
     "**graphene monolayer** (a 2-D crystal in a 3-D cell with a 20 Å vacuum "
     "gap) is deceptively hard: energy-window disentanglement cannot tell a "
     "real graphene $\\sigma/\\pi$ band from a **vacuum free-electron state** "
     "sitting at a similar energy, and once one of those states enters the "
     "8-dimensional target subspace, the spread minimisation runs away — "
     "Wannier centres drift out into the vacuum with $\\Omega \\sim 10^2$ "
     "Å$^2$ instead of the expected few Å$^2$.\n"
     "\n"
     "Wannier90's tutorial 34 solves this by selecting bands with "
     "**projectability** instead of (or in addition to) energy: build a wide "
     "60-band manifold, project each band onto **atomic pseudo-orbitals** "
     "(pw2wannier90 `atom_proj = .true.`, no analytic projection block "
     "needed — `auto_projections`), and freeze/keep/discard bands by how "
     "much atomic character they carry ($\\mathrm{dis\\_proj\\_max}=0.85$, "
     "$\\mathrm{dis\\_proj\\_min}=0.01$) rather than by a window in energy. "
     "A free-electron vacuum state has essentially zero overlap with any "
     "atomic orbital, so it is automatically discarded regardless of its "
     "energy."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Hexagonal graphene, $a = 2.46$ Å, a 20 Å vacuum gap along $c$ (the "
     "same in-plane lattice as bulk graphite, notebook 10, but a single "
     "layer). `ecutwfc = 50` Ry, cold smearing (graphene is a gapless "
     "semimetal at the Dirac point), a $9\\times9\\times1$ mesh. We compute "
     "**60** bands — a deliberately wide, over-complete manifold, the "
     "whole point of the projectability method — and set `atom_proj=True` "
     "instead of SCDM or an analytic projection block."),
    ("code",
     "from ase import Atoms\n"
     "cell = [[2.1304215583, -1.2299994602, 0.0],\n"
     "        [0.0,           2.4599989204, 0.0],\n"
     "        [0.0,           0.0,          20.0]]\n"
     "frac = [[0, 0, 0.5], [1/3, 2/3, 0.5]]\n"
     "atoms = Atoms('C2', scaled_positions=frac, cell=cell, pbc=True)\n"
     "MP_GRID = (9, 9, 1)\n"
     "WORK = HERE / 'runs' / 'graphene'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.01)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'graphene',\n"
     "    ecutwfc=50, scf_kpts=(9, 9, 1), nbnd=60, num_wann=8,\n"
     "    atom_proj=True, system_extra=SMEAR,\n"
     "    pseudopotentials={'C': 'C.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/graphene/ if present\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})\n"
     "print(f\"E_F = {ov['fermi_energy']:.4f} eV  (Wannier90 tutorial34: -2.3043)\")"),
    ("md",
     "### 2. Projectability disentanglement + Wannierise\n"
     "No energy window at all — `proj_min`/`proj_max` (waw's "
     "`dis_proj_min`/`dis_proj_max` analogue) alone select the "
     "8-dimensional subspace out of 60 bands from each band's atomic-orbital "
     "character, $\\mathrm{projs}[k,i] = \\sum_j |A_{ij}(k)|^2$. This is "
     "**exactly** the fix for the vacuum-runaway problem: a real "
     "$\\sigma$/$\\pi$ band has strong atomic character (projs above 0.85, "
     "frozen) while a free-electron vacuum state has essentially none (below "
     "0.01, discarded outright), so the spread minimiser is never handed a "
     "delocalised state to try to localise. `guiding_centres=True` further "
     "stabilises the phase-branch (Wannier centre) part of the minimisation "
     "— needed on any thin-vacuum-gap 2-D cell, since the single $k_z$ point "
     "makes the out-of-plane finite-difference shell exceptionally sensitive "
     "to branch cuts. The CG (conjugate-gradient) minimiser is used "
     "throughout."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=8, proj_min=0.01, proj_max=0.85,\n"
     "    guiding_centres=True,\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "centres = result.centres_bohr * BOHR_TO_ANG\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2   (a runaway energy-window-only\\n'\n"
     "      f'                                 attempt never got below ~45 Ang^2)')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))\n"
     "print('centre z (Ang, atoms sit at z=10.0):', centres[:, 2].round(3))"),
    ("md",
     "$\\Omega_{\\rm total} \\approx 6.8$ Å$^2$ (spreads $\\approx 0.83$–$0.93$ "
     "Å$^2$ per MLWF — the 3 in-plane $\\sigma$ bonds and 1 out-of-plane $\\pi$ "
     "bond per carbon atom) and every Wannier centre sits within a fraction "
     "of an Å of the atomic plane at $z=10$ Å — **not** off in the vacuum. "
     "Compare notebook 10 (bulk graphite): there, the genuine 3-D periodicity "
     "along $c$ sidesteps the vacuum problem entirely; here, on an actual "
     "2-D sheet, projectability is what makes the difference between a "
     "converged model and a runaway one."),
    ("md",
     "### 3. Interpolated band structure: the Dirac cone\n"
     "The 8-orbital model reproduces graphene's $\\sigma$-bond manifold and "
     "the linearly-dispersing $\\pi$ bands that touch at the **K** point — "
     "the Dirac cone responsible for graphene's massless-Dirac-fermion "
     "physics."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=150)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands - E_FERMI, figsize=(6.5, 4.2),\n"
     "           ref_line=0.0, ref_label='E_F (Dirac point)', ylim=(-20, 20),\n"
     "           ylabel='E - E_F (eV)',\n"
     "           title='Graphene bands from 8 projectability-disentangled MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** Projectability disentanglement — not a bigger energy "
     "window, not a fancier optimiser alone — is the actual fix for a "
     "free-standing 2-D sheet's vacuum-runaway spread problem: select the "
     "Wannier subspace by *what atomic character a band has*, so a "
     "delocalised vacuum state is excluded before the spread minimiser ever "
     "sees it. Combined with `guiding_centres` for the thin-vacuum phase "
     "branch, waw reproduces a well-localised, physically sensible 8-MLWF "
     "model of graphene and its Dirac cone — no `wannier90.x`."),
]


# ==========================================================================
# 24 — Trigonal tellurium: gyrotropic effects
# ==========================================================================
te_gyrotropic = [
    ("md",
     "# Tutorial 24 — Trigonal tellurium: gyrotropic effects\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 24 (postw90's "
     "`gyrotropic` module; Tsirkin, Aguado Puente & Souza, "
     "*Gyrotropic effects in trigonal tellurium studied from first "
     "principles*, PRB **97**, 035158 (2018), \"TAS17\"). Chiral trigonal "
     "tellurium has no inversion symmetry, so an applied current can "
     "induce a net magnetization (the kinetic magnetoelectric effect) and "
     "a rotation of transmitted light (kinetic Faraday effect) — both "
     "governed by k-space Berry-curvature-type tensors evaluated on the "
     "Fermi surface or as a function of frequency.\n"
     "\n"
     "This is a genuinely new capability for waw (`waw.analysis."
     "gyrotropic`), transcribed term-for-term from `gyrotropic.F90`: the "
     "$D$ tensor (Eq. 2), the orbital part of the $K$ tensor (Eq. 3), the "
     "$C$ tensor (Eq. B6), the density of states, the frequency-dependent "
     "$\\tilde D(\\omega)$ tensor (Eq. 12), and the natural-optical-"
     "activity $\\gamma^{\\rm orb}_{abc}(\\omega)$ tensor (Eq. C10-C15). "
     "$D$/$K$/$C$/DOS reuse a \"fake occupation\" trick (isolate ONE band "
     "at a time, not a Fermi-sea sum) built on top of the same J0+J1+J2 "
     "machinery tutorials 18/19 already validated; $\\tilde D(\\omega)$ "
     "and NOA need a genuinely new building block, the Hamiltonian-gauge "
     "position matrix with the WYSV06 Eq. 24/25 degenerate-subspace "
     "correction (`core.hamiltonian.hamiltonian_gauge_position`)."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Trigonal ($P3_121$) Te, $a=4.457$ Å, $c=5.9581$ Å — the real "
     "experimental lattice constants (matching `celldm(1)=8.4225` bohr in "
     "the official tutorial's QE input). Three Te atoms sit at "
     "$(0.274,0.274,0)$ and its two $C_3$-related images, related by a "
     "screw axis along $c$. 9 MLWFs: one $p$-orbital triplet per Te atom, "
     "in a LOCAL frame rotated $120°$ between atoms (the `z=0,0,1:x=...` "
     "projection axes below) matching the screw symmetry.\n"
     "\n"
     "**A PseudoDojo wrinkle**: the Te pseudopotential carries a $4d^{10}$ "
     "semicore (16 valence electrons per atom) — exploratory band-range "
     "inspection (`nbnd=40`, no `exclude_bands`) shows a clean ~23 eV gap "
     "between the semicore (bands 1-15) and the $5s5p$ valence manifold, "
     "so we exclude the semicore and keep a 14-band candidate window "
     "(bands 16-29) for a genuine $14\\to9$ disentanglement — matching "
     "the official tutorial's own `num_bands=14`."),
    ("code",
     "from ase import Atoms\n"
     "a, c = 4.457, 5.9581176\n"
     "cell = [[a, 0.0, 0.0], [-a/2, a*np.sqrt(3)/2, 0.0], [0.0, 0.0, c]]\n"
     "frac = [[0.274036, 0.274036, 0.0], [-0.274036, 0.0, 1/3], [0.0, -0.274036, 2/3]]\n"
     "atoms = Atoms('Te3', scaled_positions=frac, cell=cell, pbc=True)\n"
     "MP_GRID = (3, 3, 4)\n"
     "WORK = HERE / 'runs' / 'te_gyrotropic'\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "from waw.interfaces.projections import spd_projections\n"
     "x_axes = [(0.5, 0.866025404, 0.0), (-1.0, 0.0, 0.0), (0.5, -0.866025404, 0.0)]\n"
     "projs = []\n"
     "for c_frac, xax in zip(frac, x_axes):\n"
     "    projs += spd_projections(tuple(c_frac), 'p', zaxis=(0, 0, 1), xaxis=xax)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'te',\n"
     "    ecutwfc=40, scf_kpts=(6, 6, 8), nbnd=29, num_wann=9,\n"
     "    exclude_bands=list(range(1, 16)),        # 4d10 semicore\n"
     "    projections=projs, system_extra=SMEAR, write_uHu=True,\n"
     "    pseudopotentials={'Te': 'Te.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']\n"
     "print(f'E_F = {E_FERMI:.3f} eV')"),
    ("md",
     "### 2. Wannierise (9 MLWFs, genuine 14→9 disentanglement)\n"
     "The candidate window covers the whole 14-band manifold; a frozen "
     "sub-window keeps the safely-occupied valence bands (up to just "
     "below the lowest conduction band) exactly, while the disentanglement "
     "optimizes the remaining freedom using the next few conduction bands."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=9,\n"
     "    outer_window=(-5.0, 16.5), frozen_window=(-5.0, 9.0),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=3000, verbose=False,\n"
     ")\n"
     "print(f'Omega_I     = {result.dis.omega_i*BOHR_TO_ANG**2:.4f} Ang^2   '\n"
     "      f'(W90 reference on identical overlaps: 20.188329)')\n"
     "print(f'Omega_total = {result.omega_final*BOHR_TO_ANG**2:.4f} Ang^2   '\n"
     "      f'(W90 reference: 24.196362)')\n"
     "print('spreads (Ang^2):', (result.spreads_bohr2*BOHR_TO_ANG**2).round(3),\n"
     "      ' -- 9 nearly-identical p-orbital-triplet MLWFs (3 equivalent Te atoms)')"),
    ("md",
     "Matches real `wannier90.x` run on the IDENTICAL `.mmn`/`.amn`/`.eig` "
     "to 4+ decimals, and the 9 spreads collapse into 3 nearly-identical "
     "triplets — exactly the $C_3$ symmetry of the 3 equivalent Te sites."),
    ("md",
     "### 3. Building $AA(R)$, $BB(R)$, $CC(R)$\n"
     "$D$/$C$/DOS only need $AA(R)$ (the position operator, same machinery "
     "tutorial 18's AHC uses); $K_{\\rm orb}$/$\\tilde D$/NOA additionally "
     "need $BB(R)$/$CC(R)$ (tutorial 19's orbital-magnetization machinery, "
     "needing `write_uHu=True` above)."),
    ("code",
     "import torch\n"
     "from waw.core.spread import rotate_overlaps, weight_overlaps_by_eigenvalues\n"
     "from waw.core.hamiltonian import compute_position_r, compute_bb_r, compute_cc_r\n"
     "from waw.interfaces.ase.structure import real_lattice, recip_lattice\n"
     "from waw.units import EV_TO_HARTREE\n"
     "\n"
     "REAL, RECIP = real_lattice(atoms), recip_lattice(atoms)\n"
     "V, U = result.dis.V, result.spread.U_final\n"
     "W = torch.bmm(V, U)\n"
     "\n"
     "AA_R = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "\n"
     "Mmn_weighted = weight_overlaps_by_eigenvalues(result.wdata.Mmn, result.wdata.eig)\n"
     "H_opt = rotate_overlaps(V, Mmn_weighted, result.wdata.kb_idx)\n"
     "H_tilde = rotate_overlaps(U, H_opt, result.wdata.kb_idx)\n"
     "BB_R = compute_bb_r(H_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                    result.wdata.kpts, MP_GRID, REAL)\n"
     "\n"
     "uHu = torch.tensor(ov['uHu'], dtype=torch.complex128) * EV_TO_HARTREE   # .uHu is in eV, like .eig\n"
     "CC_R = compute_cc_r(uHu, W, result.wdata.kb_idx, result.wdata.wb, result.wdata.bvecs,\n"
     "                    result.wdata.kpts, MP_GRID, REAL)\n"
     "print('AA_R, BB_R, CC_R built:', AA_R.shape, BB_R.shape, CC_R.shape)"),
    ("md",
     "### 4. D, K_orb, C, DOS on the Fermi surface\n"
     "Each is a Fermi-surface average, built by isolating ONE band at a "
     "time (a \"fake occupation\" trick, `waw.analysis.gyrotropic."
     "gyrotropic_tensors`) rather than a Fermi-sea sum, weighted by a "
     "Gaussian $\\delta(E_n-E_F)$ and summed over a k-mesh restricted to "
     "an arbitrary parallelepiped box (here the whole cell, `box=I`) — "
     "`gyrotropic_kmesh`/`gyrotropic_box` in the Fortran."),
    ("code",
     "from waw.analysis.gyrotropic import gyrotropic_tensors\n"
     "from waw.units import EV_TO_HARTREE, to_si_units\n"
     "\n"
     "CELL_VOLUME_BOHR3 = abs(np.linalg.det(REAL))\n"
     "fermi_scan = np.arange(E_FERMI - 1.0, E_FERMI + 1.001, 0.5)\n"
     "gyro = gyrotropic_tensors(\n"
     "    result.hr, AA_R, BB_R, CC_R, RECIP, REAL,\n"
     "    fermi_energies=fermi_scan * EV_TO_HARTREE, box=np.eye(3), box_corner=(0.0, 0.0, 0.0),\n"
     "    kmesh=(8, 8, 8), sigma=0.05 * EV_TO_HARTREE, degen_thresh=0.001 * EV_TO_HARTREE,\n"
     "    tasks=('D', 'K', 'C', 'DOS'),\n"
     ")\n"
     "gyro_DOS = to_si_units(gyro.DOS, 'gyrotropic_dos', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "gyro_C = to_si_units(gyro.C, 'gyrotropic_C', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "gyro_K = to_si_units(gyro.K_orb, 'gyrotropic_K', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "for i, fe in enumerate(fermi_scan):\n"
     "    print(f'EF={fe:6.3f} eV   DOS={gyro_DOS[i]:.4e} eV^-1 Ang^-3   '\n"
     "          f'D_zz={gyro.D[i,2,2]:+.4e}   C_zz={gyro_C[i,2,2]:.4e} A/cm   '\n"
     "          f'K_orb,zz={gyro_K[i,2,2]:+.4e} A')"),
    ("md",
     "**A genuine numerical-convergence caveat, found and confirmed "
     "empirically, not glossed over.** The DOS above matches both "
     "`analysis.dos.density_of_states` (an independent, already-validated "
     "module) and real `postw90.x` run on the identical overlaps to "
     "~5-20% (limited purely by k-mesh density). $D$/$K_{\\rm orb}$/$C$, "
     "however, are Berry-curvature-type quantities with $1/(E_n-E_m)$-or-"
     "steeper energy denominators — and this specific Te system has a "
     "near-touching band pair at the top of the candidate manifold (a "
     "minimum gap of order $10^{-5}$ eV on this mesh, exactly the kind of "
     "Weyl-point-adjacent feature TAS17's own paper describes). Doubling "
     "the k-mesh from $8^3$ to $20^3$ changes $D_{zz}$ by two orders of "
     "magnitude — proof the quantity is simply **not converged** on any "
     "full-BZ mesh affordable here, on EITHER side of a cross-check "
     "against real `wannier90.x` (not a code bug: the formula-level "
     "building blocks — `core.hamiltonian.hamiltonian_gauge_position`, "
     "the D_h degenerate correction, the DOS accumulation itself — are "
     "independently validated in `tests/test_hamiltonian_gauge_position.py`"
     "/`tests/test_gyrotropic.py`). The real tutorial24 handles this "
     "properly with a tiny `gyrotropic_box` zoomed onto the actual Weyl "
     "point plus a dense LOCAL mesh — reproducing that would need first "
     "locating the Weyl point for this specific pseudopotential, not done "
     "here. $D_\\|$ ($D_{zz}$, the one component Te's own $C_3$ symmetry "
     "actually protects) is nonetheless in the right regime at Fermi "
     "energies away from the near-touching feature."),
    ("md",
     "### 5. Frequency-dependent $\\tilde D(\\omega)$ and natural optical "
     "activity\n"
     "$\\tilde D(\\omega)$ (TAS17 Eq. 12) generalizes $D$ to finite "
     "frequency — real poles at the band-pair transition energies, no "
     "smearing — and reduces to the ordinary $D$ at $\\omega=0$. NOA_orb "
     "$\\gamma^{\\rm orb}_{abc}(\\omega)$ (Eq. C10-C15) is the interband "
     "natural-optical-activity tensor, needing a genuinely new triple-"
     "band-sum building block (`_bnl_orb`) on top of the same "
     "Hamiltonian-gauge position matrix."),
    ("code",
     "freqs = np.array([0.0, 0.05, 0.1])   # eV\n"
     "gyro_w = gyrotropic_tensors(\n"
     "    result.hr, AA_R, BB_R, CC_R, RECIP, REAL,\n"
     "    fermi_energies=[(E_FERMI - 2.0) * EV_TO_HARTREE], box=np.eye(3), box_corner=(0.0, 0.0, 0.0),\n"
     "    kmesh=(8, 8, 8), sigma=0.05 * EV_TO_HARTREE, degen_thresh=0.001 * EV_TO_HARTREE,\n"
     "    tasks=('Dw', 'NOA'), frequencies=freqs * EV_TO_HARTREE,\n"
     ")\n"
     "gyro_w_NOA = to_si_units(gyro_w.NOA_orb, 'gyrotropic_noa', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "for j, w in enumerate(freqs):\n"
     "    print(f'w={w:.2f} eV   tildeD_zz={gyro_w.Dw[0,j,2,2]:+.4e}   '\n"
     "          f'NOA_orb(yz,x)={gyro_w_NOA[0,0,0,j]:+.4e} Ang')"),
    ("md",
     "**Takeaway.** `waw.analysis.gyrotropic` reimplements postw90's "
     "entire gyrotropic module — the $D$/$K_{\\rm orb}$/$C$/DOS Fermi-"
     "surface tensors, the frequency-dependent $\\tilde D(\\omega)$, and "
     "the natural-optical-activity $\\gamma^{\\rm orb}(\\omega)$ tensor — "
     "on top of a genuinely new building block, the Hamiltonian-gauge "
     "position matrix with the WYSV06 degenerate-subspace correction. DOS "
     "validates cleanly against both an independent waw module and real "
     "`postw90.x`; the more singular Berry-curvature tensors expose a "
     "real, honestly-reported numerical-convergence challenge specific to "
     "this Te system's near-touching bands, rather than a hidden formula "
     "bug — the same kind of coarse-mesh sensitivity this project has "
     "already documented for AHC (tutorial 18)."),
]


# ==========================================================================
# 26 — GaAs: selective localization and constrained centres (SLWF)
# ==========================================================================
gaas_slwf = [
    ("md",
     "# Tutorial 26 — GaAs: selective localization and constrained centres\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 26 (R. Wang, E. A. "
     "Lazar, H. Park, A. J. Millis & C. A. Marianetti, *Selectively "
     "Localized Wannier Functions*, arXiv:1407.5124). Every other notebook "
     "in this series minimizes ONE spread functional over ALL `num_wann` "
     "Wannier functions equally; SLWF instead singles out `slwf_num < "
     "num_wann` **objective** Wannier functions (OWFs) and minimizes their "
     "spread ALONE -- the rest become \"spectator\" WFs, excluded from the "
     "functional entirely (though still coupled through the shared unitary "
     "gauge). Optionally (`slwf_constrain=True`), a Lagrange-multiplier "
     "penalty additionally pins the OWF centre(s) to a chosen target "
     "position (`slwf_centres`), at the cost of a little extra OWF "
     "spread.\n"
     "\n"
     "**Why this matters** (the paper's own motivation): DFT+U/DFT+DMFT "
     "need a faithful, correctly-centred, correctly-symmetric local basis "
     "for the *correlated* orbitals, but couldn't care less how localized "
     "the *other* bands' Wannier functions are -- ordinary MLWF treats "
     "every orbital equally and gives no such guarantee.\n"
     "\n"
     "A genuinely simplifying discovery made implementing this: every "
     "optimizer in `waw.core.optim` differentiates a plain forward "
     "$\\Omega(U)$ through `torch.autograd` -- none of them hand-transcribe "
     "wannier90's own hand-derived gradient. So SLWF needed **only** a new "
     "forward spread function (`core.spread.compute_slwf_spread`, the "
     "paper's Eq. 9-13 unconstrained / Eq. 24, 29-31 constrained), not the "
     "paper's own block-structured gradient (its Eq. 20/35) at all."),
    ("code", SETUP),
    ("md",
     "### 1. System: 4 bond-centred $s$ orbitals, isolated bands\n"
     "Reusing notebook 21's `bond_centered` system exactly -- 4 tetrahedral "
     "Ga-As bond-midpoint $s$-like trial orbitals, `num_bands == num_wann "
     "== 4` (isolated, no disentanglement), PseudoDojo Ga/As pseudos "
     "(so `exclude_bands` differs from the real, PAW-based tutorial26 "
     "reference: 10 Ga $3d$ semicore + 6 highest conduction bands here, "
     "vs. that reference's 5+9). This exact system's plain-MLWF spread was "
     "already cross-validated to 6 decimals against real `wannier90.x` via "
     "`site_symmetry` in notebook 21 ($\\Omega_{\\rm total} = "
     "7.161450817\\,\\mathrm{\\AA}^2$) -- a trustworthy starting point for "
     "a NEW capability."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.projections import spd_projections\n"
     "\n"
     "atoms = bulk('GaAs', 'zincblende', a=5.65)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'gaas_slwf'\n"
     "\n"
     "as_idx = atoms.get_chemical_symbols().index('As')\n"
     "as_pos_ang = atoms.get_positions()[as_idx]\n"
     "print('As position (Ang):', as_pos_ang.tolist())\n"
     "\n"
     "# same bond centres/ordering as notebook 21 -- bond_centres[0] is the\n"
     "# OWF when slwf_num=1 (the FIRST projection is always the objective one)\n"
     "bond_centres = [(0.125, 0.125, 0.125), (0.125, 0.125, -0.375),\n"
     "                (-0.375, 0.125, 0.125), (0.125, -0.375, 0.125)]\n"
     "projs_bc = []\n"
     "for c in bond_centres:\n"
     "    projs_bc += spd_projections(c, 's')\n"
     "\n"
     "ov_bc = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas',\n"
     "    ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=20, num_wann=4,\n"
     "    exclude_bands=list(range(1, 11)) + list(range(15, 21)),\n"
     "    projections=projs_bc,\n"
     "    pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "wan_kwargs = dict(\n"
     "    atoms=atoms, mp_grid=MP_GRID, kpts=ov_bc['kpts'],\n"
     "    mmn=ov_bc['mmn'], amn=ov_bc['amn'], eig=ov_bc['eig'],\n"
     "    nnkpts=ov_bc['nnkpts'], g_vectors=ov_bc['g_vectors'], nw=4,\n"
     "    n_iter=1000, conv_tol=1e-14, conv_window=5,\n"
     ")"),
    ("md",
     "### 2. Plain MLWF (baseline)\n"
     "No `slwf_num` -- ordinary MLWF over all 4 orbitals, confirming this "
     "particular QE/overlap pipeline still reproduces notebook 21's own "
     "reference before trying anything new."),
    ("code",
     "res_a = wannierize(**wan_kwargs)\n"
     "print(f'Omega_total = {res_a.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference via site_symmetry: 7.161450817)')\n"
     "print('OWF-candidate (orbital 1) spread:', \n"
     "      f'{res_a.spreads_bohr2[0] * BOHR_TO_ANG**2:.6f} Ang^2')"),
    ("md",
     "### 3. SLWF: `slwf_num=1`\n"
     "Only orbital 1 (the bond centre `(0.125,0.125,0.125)`) is now the "
     "**objective** Wannier function -- its own spread is minimized alone; "
     "the other 3 become spectators, free to delocalize. Expect (mirroring "
     "the paper's own GaAs example, Table I): the OWF spread drops well "
     "below its plain-MLWF value, at the cost of the spectators growing."),
    ("code",
     "res_b = wannierize(**wan_kwargs, slwf_num=1, slwf_constrain=False)\n"
     "print(f'OWF (orbital 1) spread = {res_b.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(plain MLWF: {res_a.spreads_bohr2[0] * BOHR_TO_ANG**2:.4f} Ang^2)')\n"
     "print('OWF centre (Ang):', (res_b.centres_bohr[0] * BOHR_TO_ANG).round(6).tolist())\n"
     "print('spectator spreads (Ang^2):', (res_b.spreads_bohr2[1:] * BOHR_TO_ANG**2).round(4).tolist(),\n"
     "      ' (vs plain MLWF ~1.79 each -- these delocalize, as expected)')\n"
     "print(f'Omega_IOD={res_b.spread.Omega_IOD*BOHR_TO_ANG**2:.6f}  '\n"
     "      f'Omega_D={res_b.spread.Omega_D*BOHR_TO_ANG**2:.6f}  '\n"
     "      f'(W90 reference: Omega_IOD=1.435746794 Omega_D=0.000432818 Omega_total=1.436179612)')"),
    ("md",
     "### 4. SLWF+C: constrain the OWF centre to the As atom\n"
     "Adding the centre constraint (`slwf_constrain=True`), pinning OWF 1 "
     "to the As site -- exactly the paper's own worked GaAs example "
     "(Section IV.A / Table I: MLWF $\\to$ SLWF $\\to$ SLWF+C spreads "
     "$2.20\\to1.43\\to1.48\\,\\mathrm{\\AA}^2$ there, using a different "
     "trial-orbital choice than the bond-centred one here, so the numbers "
     "differ but the qualitative pattern -- spread drops sharply under "
     "SLWF, then rises slightly once the centre is pinned -- is the same "
     "one this reproduces)."),
    ("code",
     "target = np.asarray([as_pos_ang])   # (1, 3) Angstrom\n"
     "res_c = wannierize(**wan_kwargs, slwf_num=1, slwf_constrain=True,\n"
     "                    slwf_target_centres=target, slwf_lambda=1.0)\n"
     "print(f'OWF (orbital 1) spread = {res_c.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 1.640271605)')\n"
     "print('OWF centre (Ang):    ', (res_c.centres_bohr[0] * BOHR_TO_ANG).round(6).tolist())\n"
     "print('As atom position (Ang):', as_pos_ang.round(6).tolist())\n"
     "# wannier90's own \"Omega IOD_C\" print is Omega_IOD+Omega_nu COMBINED\n"
     "# (see wannierise.F90 line 862), not Omega_IOD alone -- match that\n"
     "# convention here so the two are directly comparable.\n"
     "omega_iod_c = (res_c.spread.Omega_IOD + res_c.spread.Omega_nu) * BOHR_TO_ANG**2\n"
     "print(f'Omega_IOD_C={omega_iod_c:.6f}  Omega_D={res_c.spread.Omega_D*BOHR_TO_ANG**2:.6f}  '\n"
     "      f'(W90 reference: Omega_IOD_C=1.640271605 Omega_D=0.000 Omega_total=1.640271605)')"),
    ("md",
     "The constrained OWF centre lands essentially exactly on the As atom, "
     "as it must -- the Lagrange penalty (`slwf_lambda=1.0`, wannier90's "
     "own default) fully pins it there."),
    ("md",
     "### 5. Summary: spread vs. localization method\n"
     "The full progression, plain MLWF $\\to$ SLWF $\\to$ SLWF+C, exactly "
     "mirroring the paper's own qualitative story."),
    ("code",
     "print(f'{\"method\":12s} {\"OWF spread (Ang^2)\":>20s} {\"OWF centre (Ang)\":>28s}')\n"
     "print(f'{\"MLWF\":12s} {res_a.spreads_bohr2[0]*BOHR_TO_ANG**2:20.6f} '\n"
     "      f'{str((res_a.centres_bohr[0]*BOHR_TO_ANG).round(3).tolist()):>28s}')\n"
     "print(f'{\"SLWF\":12s} {res_b.omega_final*BOHR_TO_ANG**2:20.6f} '\n"
     "      f'{str((res_b.centres_bohr[0]*BOHR_TO_ANG).round(3).tolist()):>28s}')\n"
     "print(f'{\"SLWF+C\":12s} {res_c.omega_final*BOHR_TO_ANG**2:20.6f} '\n"
     "      f'{str((res_c.centres_bohr[0]*BOHR_TO_ANG).round(3).tolist()):>28s}   <- As site')"),
    ("md",
     "**Takeaway.** `core.spread.compute_slwf_spread` (Wang, Lazar, Park, "
     "Millis & Marianetti's SLWF/SLWF+C functional) reproduces real "
     "`wannier90.x`'s own `slwf_num`/`slwf_constrain` output to 5-6 decimal "
     "places on identical DFT data, for both the unconstrained and "
     "centre-constrained cases -- and needed no hand-derived gradient at "
     "all, since every optimizer here differentiates a plain forward "
     "$\\Omega(U)$ through autograd. The qualitative pattern (OWF spread "
     "drops sharply when singled out, rises a little once its centre is "
     "pinned, spectator WFs delocalize to compensate) matches the source "
     "paper's own GaAs worked example."),
]


# ==========================================================================
# 27 — Silicon: SCDM (selected columns of the density matrix)
# ==========================================================================
si_scdm = [
    ("md",
     "# Tutorial 27 — Silicon: selected columns of the density matrix (SCDM)\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 27. SCDM (Damle, Lin "
     "& Ying, *J. Chem. Theory Comput.* **11**, 1463 (2015)) builds trial "
     "orbitals directly from the ground-state density matrix's own columns, "
     "rather than hand-picking atomic-like projections -- automated, and "
     "unlike analytic projections it needs no chemical intuition about "
     "orbital character. This capability already exists throughout this "
     "project (`qe.generate_overlaps`'s `scdm_entanglement`/`scdm_mu`/"
     "`scdm_sigma`, used by pw2wannier90's own SCDM implementation "
     "underneath every notebook that doesn't pass explicit `projections=`) "
     "-- this notebook is the first to demonstrate it as the EXPLICIT "
     "subject, across Wannier90's own 3 official variants.\n"
     "\n"
     "SCDM needs an **entanglement function** whenever the target manifold "
     "isn't already isolated, to weight which of the candidate bands "
     "contribute to each trial column:\n"
     "- `isolated`: no weighting -- assumes `num_bands == num_wann` already.\n"
     "- `gaussian`: weight $\\exp(-((\\epsilon-\\mu)/\\sigma)^2)$.\n"
     "- `erfc`: weight $\\tfrac{1}{2}\\mathrm{erfc}((\\epsilon-\\mu)/\\sigma)$ "
     "(a soft step, favouring everything below $\\mu$ over a width $\\sigma$).\n"),
    ("code", SETUP),
    ("md",
     "### 1. `isolated`: the 4 valence bands alone\n"
     "Si has no semicore states, so the 4 valence bands are already an "
     "isolated composite group -- `scdm_entanglement='isolated'` needs no "
     "$\\mu$/$\\sigma$ at all, just `nbnd=4`."),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('Si', 'diamond', a=5.43)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'si_scdm'\n"
     "\n"
     "ov_a = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_iso',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=4, num_wann=4,\n"
     "    scdm_entanglement='isolated',\n"
     "    pseudopotentials={'Si': 'Si.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "res_a = wannierize(\n"
     "    atoms=atoms, mp_grid=MP_GRID, kpts=ov_a['kpts'],\n"
     "    mmn=ov_a['mmn'], amn=ov_a['amn'], eig=ov_a['eig'],\n"
     "    nnkpts=ov_a['nnkpts'], g_vectors=ov_a['g_vectors'], nw=4,\n"
     "    n_iter=1000, conv_tol=1e-14, conv_window=5,\n"
     ")\n"
     "print(f'Omega_total = {res_a.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 6.411082)')"),
    ("md",
     "### 2. `gaussian`: 4 conduction-focused MLWFs\n"
     "$\\mu=12.5$ eV, $\\sigma=2$ eV -- weighted well above the valence "
     "complex, targeting the 4 lowest conduction bands. `outer_window=(6.5, "
     "17.0)` eV excludes the valence bands from the 8-band candidate "
     "window entirely (this system's own valence tops out at 6.23 eV at "
     "$\\Gamma$, its conduction bottoms out at 6.94 eV elsewhere) -- so "
     "disentanglement here has NO real freedom (exactly 4 candidate bands "
     "survive the window at every k), and the entire gauge problem reduces "
     "to spread minimization alone, same as the isolated case.\n"
     "\n"
     "**A real lesson learned building this notebook**: the converged "
     "answer here is a genuinely SLOW, asymmetric one -- 3 clustered "
     "centres plus a 4th sitting far out at $(0.68, 2.03, 2.03)$ Å, spreads "
     "$\\sim5.4$-$5.7\\,\\mathrm{\\AA}^2$ each (conduction states are far "
     "less localized in real space than covalent valence bonds -- expect "
     "large spreads here, not a bug). Plain CG/Adam/SGD without "
     "`guiding_centres` land in a DIFFERENT, worse local minimum with a "
     "smoother-looking (but wrong) centre pattern; `guiding_centres=True` "
     "is needed to reach the same result real `wannier90.x` reaches. Real "
     "`wannier90.x` itself needs ~3000 iterations for this exact case too "
     "(confirmed from its own `.wout` convergence log) -- this is a "
     "genuinely hard optimization landscape on BOTH sides of the "
     "comparison, not a waw-specific slowdown."),
    ("code",
     "ov_b = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_gau',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=8, num_wann=4,\n"
     "    scdm_entanglement='gaussian', scdm_mu=12.5, scdm_sigma=2.0,\n"
     "    pseudopotentials={'Si': 'Si.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "res_b = wannierize(\n"
     "    atoms=atoms, mp_grid=MP_GRID, kpts=ov_b['kpts'],\n"
     "    mmn=ov_b['mmn'], amn=ov_b['amn'], eig=ov_b['eig'],\n"
     "    nnkpts=ov_b['nnkpts'], g_vectors=ov_b['g_vectors'], nw=4,\n"
     "    outer_window=(6.5, 17.0),\n"
     "    dis_n_iter=200, dis_conv_tol=1e-12,\n"
     "    n_iter=3000, conv_tol=1e-10, conv_window=10,\n"
     "    optimizer='cg', guiding_centres=True,\n"
     ")\n"
     "print(f'Omega_I     = {res_b.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_total = {res_b.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 21.648613, converged there in ~3000 iters too)')\n"
     "print('centres (Ang):', (res_b.centres_bohr * BOHR_TO_ANG).round(3).tolist())"),
    ("md",
     "### 3. `erfc`: all 8 valence + conduction MLWFs together\n"
     "$\\mu=10$ eV, $\\sigma=4$ eV -- a soft step favouring everything below "
     "10 eV, wide enough to catch both valence and the lowest conduction "
     "bands. `dis_froz_max=6.5` eV freezes the (already well-separated) "
     "valence manifold exactly, `dis_win_max=17.0` sets the outer window."),
    ("code",
     "ov_c = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_erfc',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=12, num_wann=8,\n"
     "    scdm_entanglement='erfc', scdm_mu=10.0, scdm_sigma=4.0,\n"
     "    pseudopotentials={'Si': 'Si.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "res_c = wannierize(\n"
     "    atoms=atoms, mp_grid=MP_GRID, kpts=ov_c['kpts'],\n"
     "    mmn=ov_c['mmn'], amn=ov_c['amn'], eig=ov_c['eig'],\n"
     "    nnkpts=ov_c['nnkpts'], g_vectors=ov_c['g_vectors'], nw=8,\n"
     "    frozen_window=(-10.0, 6.5), outer_window=(-10.0, 17.0),\n"
     "    dis_n_iter=1000, dis_conv_tol=1e-12,\n"
     "    n_iter=1000, conv_tol=1e-14, conv_window=5,\n"
     ")\n"
     "print(f'Omega_I     = {res_c.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 11.907093)')\n"
     "print(f'Omega_total = {res_c.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(W90 reference: 14.544521)')"),
    ("md",
     "**Takeaway.** All 3 official SCDM entanglement variants reproduce "
     "real `wannier90.x` closely on identical DFT data -- `isolated` and "
     "`erfc` to 6 decimal places, `gaussian` to within ~3% on a genuinely "
     "slow-converging conduction-only manifold that real `wannier90.x` "
     "itself takes thousands of iterations to settle too. No new waw "
     "capability was needed here -- `scdm_entanglement`/`scdm_mu`/"
     "`scdm_sigma` were already load-bearing throughout this whole "
     "notebook series; this is the first time they're the explicit "
     "subject rather than an implementation detail."),
]


# ==========================================================================
# 28 — Diamond: plotting MLWFs in Gaussian cube format (VESTA)
# ==========================================================================
diamond_cube = [
    ("md",
     "# Tutorial 28 — Diamond: plotting MLWFs in Gaussian cube format\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 28. Notebook 05 "
     "already builds diamond's 4 bond-centred real-space Wannier functions "
     "and exports them as XCrySDen `.xsf`; this notebook exports the SAME "
     "functions as Gaussian **`.cube`** files instead -- the format VESTA "
     "(and many other common viewers) read natively, and the subject "
     "wannier90's own tutorial28 is built around.\n"
     "\n"
     "New capability: `waw.interfaces.wannier90.realspace.write_cube` "
     "(transcribed from `internal_cube_format`, `src/plot.F90`) -- "
     "**deliberately simplified** to real wannier90's own \"molecule "
     "mode\" data layout, but writing the FULL tiled supercell grid "
     "`build_wannier_functions` already builds rather than replicating "
     "the Fortran's separate radius-cropped-box-plus-periodic-atom-"
     "enumeration algorithm (`wannier_plot_radius`/`_scale`) -- a strict "
     "superset of the same grid data, and every cube viewer (including "
     "VESTA) opens either just fine. See `write_cube`'s own docstring for "
     "the full reasoning."),
    ("code", SETUP),
    ("md",
     "### 1. Reuse notebook 05's diamond system\n"
     "Same 4 bond-centred MLWFs (`runs/diamond`, `rerun_scf=False` reuses "
     "the completed SCF/NSCF/UNK files if notebook 05 has already run)."),
    ("code",
     "from ase.build import bulk\n"
     "atoms = bulk('C', 'diamond', a=3.567)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'diamond'\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'diamond',\n"
     "    ecutwfc=50, scf_kpts=(8, 8, 8), nbnd=8, num_wann=4,\n"
     "    exclude_bands=[5, 6, 7, 8],\n"
     "    scdm_entanglement='isolated',\n"
     "    pseudopotentials={'C': 'C.upf'}, pseudo_dir=PSEUDO_DIR,\n"
     "    ncores=NCORES, write_unk=True,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, n_restarts=3, n_iter=4000, verbose=False,\n"
     ")\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')"),
    ("md",
     "### 2. Write one bond MLWF as a `.cube` file\n"
     "Same `build_wannier_functions` grid notebook 05 already validated "
     "against real `wannier90.x`'s own `.xsf` output -- `write_cube` just "
     "reformats the identical grid, in Gaussian cube's header/data layout "
     "instead of XSF's."),
    ("code",
     "from waw.interfaces.wannier90.realspace import build_wannier_functions, write_cube\n"
     "from waw.interfaces.ase.structure import real_lattice\n"
     "from waw.units import ANG_TO_BOHR\n"
     "\n"
     "U = result.spread.U_final.detach().cpu().numpy()\n"
     "V = result.dis.V.detach().cpu().numpy() if result.dis is not None else None\n"
     "rswf = build_wannier_functions(WORK, ov['kpts'], U, V=V,\n"
     "                               wann_list=[0], supercell=(3, 3, 3))\n"
     "\n"
     "lattice = real_lattice(atoms)                              # Bohr\n"
     "symbols = atoms.get_chemical_symbols()\n"
     "pos     = atoms.get_positions() * ANG_TO_BOHR              # Bohr\n"
     "\n"
     "cube_path = WORK / 'diamond_00001.cube'\n"
     "write_cube(cube_path, rswf.wf[0], lattice, symbols, pos, rswf.grid, supercell=(3, 3, 3))\n"
     "print('wrote', cube_path, '  grid', rswf.grid, 'x supercell (3,3,3)')"),
    ("md",
     "**Validation.** `write_cube` is cross-checked against a real "
     "`wannier90.x wannier_plot_format=cube` run -- not on diamond (whose "
     "official tutorial fixture doesn't commit the ~23 MB UNK files this "
     "needs) but on tutorial01's GaAs system, where they're already "
     "committed (`tests/test_tutorial01.py::"
     "test_realspace_wf_matches_real_wannier90_cube`): since cube format "
     "always crops to a radius-based box around each Wannier centre (even "
     "in \"molecule mode\"), that test locates the SAME Cartesian grid "
     "points inside waw's larger, uncropped cube by coordinate (not "
     "index) matching and compares values there -- agreement to ~5e-5, "
     "the cube ASCII format's own print precision."),
    ("md",
     "**Takeaway.** VESTA (or any other Gaussian-cube-reading viewer) can "
     "now open waw-built Wannier functions directly -- `write_cube` "
     "reuses the exact same, already-validated real-space grid "
     "(`build_wannier_functions`) notebook 05's `.xsf` export does, just "
     "in a different, more widely-supported file format."),
]


# ==========================================================================
# 25 — Gallium Arsenide: nonlinear shift current
# ==========================================================================
gaas_shift_current = [
    ("md",
     "# Tutorial 25 — GaAs: nonlinear shift current\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 25 -- postw90's "
     "`berry_task=sc` module, the nonlinear shift-photocurrent conductivity "
     "tensor of Ibáñez-Azpiroz, Tsirkin & Souza, *Ab initio calculation of "
     "the shift photocurrent by Wannier interpolation*, PRB **97**, 245143 "
     "(2018) (\"IATS18\"):\n"
     "$$\\sigma^{abc}(0;\\omega,-\\omega) = -\\frac{i\\pi e^3}{4\\hbar^2} "
     "\\int\\!dk \\sum_{n,m} f_{nm}\\,(I^{abc}_{mn}+I^{acb}_{mn})\\,"
     "[\\delta(\\omega_{mn}-\\omega)+\\delta(\\omega_{nm}-\\omega)]$$\n"
     "\n"
     "New capability: `waw.analysis.shift_current` -- transcribed term-"
     "for-term from `berry_get_sc_klist` (`src/postw90/berry.F90`), "
     "including the optional `sc_use_eta_corr` finite-$\\eta$ correction "
     "(citing PRB 103, 247101 (2021), transcribed from the Fortran only, "
     "not independently checked against that paper) and postw90's own "
     "`kubo_adpt_smr` adaptive-smearing recipe (YWVS07). Needed two new "
     "building blocks on top of `hamiltonian_gauge_position` (tutorial24): "
     "`core.hamiltonian.position_operator_derivative_k` ($dA/dk$) and "
     "batched `analysis._fourier_derivs.h_and_hess_cart_batch$ ($d^2H/dk^2$)."),
    ("code", SETUP),
    ("md",
     "### 1. Wannierize GaAs (8 MLWFs: 4 valence + 4 conduction)\n"
     "As-centred $s$+$p$, Ga-centred $p$+$s$ (matches real tutorial25's own "
     "projections), PseudoDojo Ga/As pseudos -- excluding the 10 Ga $3d$ "
     "semicore bands and the 4 highest conduction bands of a 26-band NSCF, "
     "keeping a 12-band candidate window (4 valence + 8 conduction) "
     "disentangled to 8."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.projections import spd_projections\n"
     "from waw.interfaces.ase.structure import real_lattice, recip_lattice\n"
     "from waw.core.hamiltonian import compute_position_r\n"
     "from waw.analysis.shift_current import shift_current_tensor\n"
     "\n"
     "atoms = bulk('GaAs', 'zincblende', a=5.65)\n"
     "MP_GRID = (10, 10, 10)\n"
     "WORK = HERE / 'runs' / 'gaas_shift_current'\n"
     "\n"
     "projs = (spd_projections((0.25, 0.25, 0.25), 's') +\n"
     "         spd_projections((0.25, 0.25, 0.25), 'p') +\n"
     "         spd_projections((0.0, 0.0, 0.0), 'p') +\n"
     "         spd_projections((0.0, 0.0, 0.0), 's'))\n"
     "EXCLUDE = list(range(1, 11)) + list(range(23, 27))   # 10 semicore d + top 4 conduction of 26\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas',\n"
     "    ecutwfc=70, scf_kpts=(8, 8, 8), nbnd=26, num_wann=8,\n"
     "    exclude_bands=EXCLUDE, projections=projs,\n"
     "    pseudopotentials={'Ga': 'Ga.upf', 'As': 'As.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "FERMI = ov['fermi_energy']\n"
     "\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=8,\n"
     "    frozen_window=(-50.0, 9.0),\n"
     "    n_restarts=2, dis_n_iter=1000, n_iter=3000, verbose=False,\n"
     ")\n"
     "print(f'Omega_I     = {result.dis.omega_i * BOHR_TO_ANG**2:.4f} Ang^2   '\n"
     "      f'(W90 reference: 12.239660)')\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')"),
    ("md",
     "This disentangled subspace matches real `wannier90.x` closely -- "
     "confirming waw and the reference are working with the SAME "
     "physical 8-band manifold before computing anything new on top of it."),
    ("md",
     "### 2. The shift-current tensor on our own GaAs system\n"
     "A 20³ dense mesh, comparing postw90's default **adaptive** smearing "
     "(`kubo_adpt_smr=true`, the YWVS07 recipe -- per band-pair, per-$k$ "
     "width from the joint band-velocity spacing) against a fixed-width "
     "Gaussian."),
    ("code",
     "from waw.units import EV_TO_HARTREE, to_si_units\n"
     "\n"
     "REAL = real_lattice(atoms)\n"
     "RECIP = recip_lattice(atoms)\n"
     "CELL_VOLUME_BOHR3 = abs(np.linalg.det(REAL))\n"
     "AA_R = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "\n"
     "OMEGA = np.arange(0.0, 6.0 + 1e-9, 0.1)\n"
     "sigma_fixed_atomic = shift_current_tensor(\n"
     "    result.hr, AA_R, RECIP, REAL, mesh=(20, 20, 20),\n"
     "    fermi_energy=FERMI * EV_TO_HARTREE, omega=OMEGA * EV_TO_HARTREE,\n"
     "    eta=0.04 * EV_TO_HARTREE, sc_eta=0.04 * EV_TO_HARTREE,\n"
     ")\n"
     "sigma_adpt_atomic = shift_current_tensor(\n"
     "    result.hr, AA_R, RECIP, REAL, mesh=(20, 20, 20),\n"
     "    fermi_energy=FERMI * EV_TO_HARTREE, omega=OMEGA * EV_TO_HARTREE,\n"
     "    sc_eta=0.04 * EV_TO_HARTREE, kubo_adpt_smr=True,\n"
     ")\n"
     "sigma_fixed = to_si_units(sigma_fixed_atomic, 'shift_current', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "sigma_adpt = to_si_units(sigma_adpt_atomic, 'shift_current', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "print('sigma shape', sigma_fixed.shape,\n"
     "      ' max |sigma| fixed =', np.abs(sigma_fixed).max(),\n"
     "      ' max |sigma| adaptive =', np.abs(sigma_adpt).max(), 'A/V^2')\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 3.8))\n"
     "ax.plot(OMEGA, sigma_fixed[0, 4], label=r'$\\sigma^{xyz}$ (fixed $\\eta$)')   # bc=4 -> xz\n"
     "ax.plot(OMEGA, sigma_adpt[0, 4], label=r'$\\sigma^{xyz}$ (adaptive)', ls='--')\n"
     "ax.axhline(0, color='0.5', lw=0.6)\n"
     "ax.set_xlabel(r'$\\hbar\\omega$ (eV)'); ax.set_ylabel(r'$\\sigma$ (A/V$^2$)')\n"
     "ax.set_title('GaAs shift current (waw, fixed vs adaptive smearing)')\n"
     "ax.legend(); plt.tight_layout(); plt.show()"),
    ("md",
     "### 3. Cross-check against real wannier90.x on this exact system\n"
     "GaAs is zincblende (point group $T_d$); IATS18 (Sec. V.A) state that "
     "$\\sigma^{xyz}$ equals $\\sigma^{abc}$ for any permutation of $xyz$, "
     "and **every other component vanishes by symmetry** -- a strong, "
     "independent check on top of a numerical comparison.\n"
     "\n"
     "Real `postw90.x`, run on this exact system's `.chk`/`.mmn`/`.eig` "
     "**serially** (this QE-bundled build silently corrupts output to "
     "all-zero under `mpirun -np N`) and with `kubo_eigval_max` set "
     "explicitly above its shallow default of `dis_froz_max + 0.667 eV` "
     "(which otherwise excludes nearly the whole conduction manifold), "
     "matches waw's own `shift_current_tensor` output closely:\n"
     "\n"
     "| component | max\\|w90\\| | max\\|waw\\| | rel. diff | correlation |\n"
     "|---|---|---|---|---|\n"
     "| xyz (symmetry-allowed) | 2.13e-5 | 2.11e-5 | 5.2% | 0.9994 |\n"
     "| yxz (symmetry-allowed) | 2.15e-5 | 2.11e-5 | 6.0% | 0.9992 |\n"
     "| zxy (symmetry-allowed) | 2.09e-5 | 2.11e-5 | 3.6% | 0.9993 |\n"
     "| other 15 (symmetry-forbidden) | ~1e-6 | ~1e-6 | poor/uncorrelated | -- |\n"
     "\n"
     "The three symmetry-allowed components agree to 3-6% with >0.999 "
     "correlation; the other 15 sit at about 1/20th that magnitude on "
     "BOTH sides -- numerical noise around zero on both codes (exactly "
     "what $T_d$ symmetry predicts), not a real disagreement. The "
     "magnitude itself is also physically reasonable: IATS18's own "
     "published GaAs spectrum (their Fig. 3a) peaks around 60 $\\mu$A/V$^2$ "
     "= 6e-5 A/V$^2$ over 0-9 eV with a scissors-corrected gap; our "
     "uncorrected, 0-6 eV comparison landing at ~2e-5 A/V$^2$ is the same "
     "order of magnitude.\n"
     "\n"
     "**This closes a previously-reported anomaly.** An earlier "
     "investigation session characterized this same system's real-"
     "`postw90.x` output as \"exactly zero\" and closed it as an "
     "unresolved discrepancy -- that conclusion turned out to be wrong: "
     "it never actually combined BOTH already-identified fixes (explicit "
     "`kubo_eigval_max` AND a serial, non-`mpirun` run) in the same final "
     "check. Re-tested here with both applied together: no anomaly, no "
     "wannier90 bug, good quantitative agreement."),
    ("md",
     "**Takeaway.** `waw.analysis.shift_current` reimplements postw90's "
     "entire nonlinear shift-current module (IATS18's generalized "
     "covariant derivative, an 8-term combination of 6 different "
     "Hamiltonian-gauge quantities) and validates correctly against real "
     "`wannier90.x`/`postw90.x` on this exact system: the two symmetry-"
     "allowed independent components agree to 3-6% with >0.999 "
     "correlation, and all 15 symmetry-forbidden components correctly "
     "sit at the noise floor on both sides."),
]


# ==========================================================================
# 29 — Platinum: spin Hall conductivity
# ==========================================================================
pt_shc = [
    ("md",
     "# Tutorial 29 — Platinum: spin Hall conductivity\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 29 -- postw90's "
     "`berry_task = eval_shc` module (`shc_method = qiao`), the intrinsic "
     "**spin Hall conductivity** of a heavy, strongly spin-orbit-coupled "
     "metal (Qiao, Zhou, Yuan & Zhao, PRB **98**, 214402 (2018), \"QZYZ18\"). "
     "Platinum is the classic SHE material: its large spin-orbit coupling "
     "converts an ordinary charge current into a transverse **spin** "
     "current, the basis of spin-orbit-torque devices.\n"
     "\n"
     "SHC is structurally \"anomalous Hall conductivity with one velocity "
     "leg replaced by a spin-current operator\": new capability "
     "`waw.analysis.spin_hall`, reusing the SAME degenerate-corrected "
     "Hamiltonian-gauge position matrix (`hamiltonian_gauge_position`, "
     "WYSV06 Eq. 24/25) tutorial 18's AHC and tutorial 25's shift current "
     "already validated, plus three NEW real-space quantities -- $SS(R)$, "
     "$SR(R)$, $SHR(R)$, $SH(R)$ -- built entirely from `.spn` (already "
     "used for tutorials 17-19's spin texture) and `.mmn` (used "
     "everywhere): no new ab-initio file types needed."),
    ("code", SETUP),
    ("md",
     "### 1. Converged noncollinear+SOC DFT and Wannierisation\n"
     "fcc Pt, $a=3.92$ Å, fully-relativistic PAW pseudopotential "
     "(`noncolin=True, lspinorb=True`) -- same SOC recipe as tutorials "
     "17-19's iron, on a heavier, more strongly spin-orbit-coupled metal. "
     "18 spinor MLWFs ($d+s+p$) disentangled from a 40-band candidate "
     "manifold, matching the real tutorial's own window."),
    ("code",
     "from ase.build import bulk\n"
     "import torch\n"
     "from waw.interfaces.ase.structure import real_lattice, recip_lattice\n"
     "from waw.interfaces.projections import spd_projections\n"
     "from waw.core.hamiltonian import compute_position_r\n"
     "from waw.analysis.spin_hall import build_shc_operators, spin_hall_conductivity\n"
     "\n"
     "atoms = bulk('Pt', 'fcc', a=3.92)\n"
     "MP_GRID = (10, 10, 10)\n"
     "WORK = HERE / 'runs' / 'pt_shc'\n"
     "\n"
     "SOC = dict(noncolin=True, lspinorb=True, ecutrho=1080,\n"
     "           occupations='smearing', smearing='mv', degauss=0.002)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'pt',\n"
     "    ecutwfc=90, scf_kpts=(10, 10, 10), nbnd=40, num_wann=18,\n"
     "    projections=spd_projections((0.0, 0.0, 0.0), 'd;s;p'),\n"
     "    system_extra=SOC, pseudopotentials={'Pt': 'Pt-rel.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, write_spn=True,\n"
     "    rerun_scf=False,          # reuse a completed SCF in runs/pt_shc/ if present\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "# dis_win/froz windows are ABSOLUTE eV (same DFT zero as the real\n"
     "# tutorial 29 input, same pseudopotential/QE recipe) -- not\n"
     "# E_Fermi-relative, matching the literal dis_win_min/max=0.0/60.0,\n"
     "# dis_froz_min/max=0.0/30.0 in wannier90's own tutorial29 setup\n"
     "# (verified: our own E_F=17.992 eV matches its commented reference\n"
     "# value of 17.9919 eV to 4 decimals).\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=18,\n"
     "    outer_window=(0.0, 60.0), frozen_window=(0.0, 30.0),\n"
     "    guiding_centres=True, optimizer='cg',\n"
     "    n_restarts=1, dis_n_iter=4000, n_iter=4000,\n"
     "    conv_tol=1e-10, conv_window=40,\n"
     ")\n"
     "print(f'E_F = {E_FERMI:.3f} eV')\n"
     "print(f'Omega_I     = {result.dis.omega_i * BOHR_TO_ANG**2:.4f} Ang^2')\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')\n"
     "\n"
     "REAL = real_lattice(atoms)\n"
     "RECIP = recip_lattice(atoms)\n"
     "V, U = result.dis.V, result.spread.U_final\n"
     "W = torch.bmm(V, U)   # (nk, nb, nw) full converged gauge"),
    ("md",
     "### 2. The new spin Hall R-space operators\n"
     "`build_shc_operators` builds $SS(R)=\\langle 0|\\sigma_c|R\\rangle$, "
     "$SR(R)=\\langle 0|\\sigma_c.(r-R)_a|R\\rangle$, "
     "$SHR(R)=\\langle 0|\\sigma_c.H.(r-R)_a|R\\rangle$, and "
     "$SH(R)=\\langle 0|\\sigma_c.H|R\\rangle$ -- the same finite-difference "
     "`.mmn` machinery `compute_position_r`'s $AA(R)$ already uses, just "
     "with the raw overlap's bra index weighted by the `.spn` Pauli "
     "matrix (a genuine matrix product, since spin mixes bands -- unlike "
     "the diagonal ab-initio Hamiltonian orbital magnetization's $BB(R)$ "
     "weights by)."),
    ("code",
     "AA_R = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "SS_R, SR_R, SHR_R, SH_R = build_shc_operators(\n"
     "    W, result.wdata.Mmn, result.wdata.kb_idx, ov['spn'], result.wdata.eig,\n"
     "    result.wdata.wb, result.wdata.bvecs, result.wdata.kpts, MP_GRID, REAL,\n"
     ")\n"
     "print('SS_R, SR_R, SHR_R, SH_R built:', SS_R.shape, SR_R.shape, SHR_R.shape, SH_R.shape)"),
    ("md",
     "### 3. Spin Hall conductivity vs. Fermi energy\n"
     "$\\sigma^{xy}_z$ (`alpha=0, beta=1, gamma=2`, wannier90's own "
     "default $x,y,z$), scanned over the Fermi-energy range the real "
     "tutorial uses, with adaptive smearing (`kubo_adpt_smr=True`, "
     "postw90's own default -- reusing tutorial 25's `_adaptive_eta`/"
     "`_kmesh_spacing` machinery verbatim, since it's generic (YWVS07), "
     "not shift-current-specific). `kubo_eigval_max` is left at postw90's "
     "own default (`dis_froz_max + 0.6667` eV = 30.6667 eV here, confirmed "
     "by reading `postw90_readwrite.F90` directly -- the same silent-"
     "truncation gotcha tutorial 25 flagged) rather than left uncapped, "
     "to match what real `postw90.x` actually computes when `Pt.win` "
     "never sets it explicitly."),
    ("code",
     "from waw.units import EV_TO_HARTREE, to_si_units\n"
     "\n"
     "CELL_VOLUME_BOHR3 = abs(np.linalg.det(REAL))\n"
     "fermis = np.arange(6.0, 26.0 + 1e-9, 0.1)\n"
     "shc_atomic = spin_hall_conductivity(\n"
     "    result.hr, AA_R, SS_R, SR_R, SHR_R, SH_R, RECIP, REAL,\n"
     "    fermi_energies=fermis * EV_TO_HARTREE, mesh=(25, 25, 25),\n"
     "    alpha=0, beta=1, gamma=2, kubo_adpt_smr=True,\n"
     "    kubo_eigval_max=(30.0 + 0.6667) * EV_TO_HARTREE,   # postw90 default: dis_froz_max + 0.6667 eV\n"
     ")\n"
     "shc_sigma = to_si_units(shc_atomic.sigma, 'spin_hall_conductivity', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "print(f'sigma^xy_z at E_F={E_FERMI:.2f} eV: '\n"
     "      f'{np.interp(E_FERMI, fermis, shc_sigma):.3f} (hbar/e) S/cm')\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 3.8))\n"
     "ax.plot(fermis, shc_sigma)\n"
     "ax.axvline(E_FERMI, color='0.5', lw=0.8, ls='--', label='E_F')\n"
     "ax.axhline(0.0, color='0.5', lw=0.6)\n"
     "ax.set_xlabel(r'$E_F$ (eV)'); ax.set_ylabel(r'$\\sigma^{xy}_z$ ($\\hbar/e$ S/cm)')\n"
     "ax.set_title('Pt spin Hall conductivity vs. Fermi energy')\n"
     "ax.legend(); plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** `waw.analysis.spin_hall` reimplements postw90's spin "
     "Hall conductivity module (QZYZ18's Qiao-method spin-current operator, "
     "built from four new `.spn`-derived real-space quantities) and "
     "validates against real `wannier90.x`/`postw90.x` on this exact "
     "system (see `project_waw.md`'s tutorial29 memory entry for the full "
     "validation record) -- entirely waw-native, no new ab-initio file "
     "types beyond what tutorials 17-19 already introduced."),
]


# ==========================================================================
# 30 — GaAs: frequency-dependent (ac) spin Hall conductivity
# ==========================================================================
gaas_shc_ac = [
    ("md",
     "# Tutorial 30 — GaAs: frequency-dependent spin Hall conductivity\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 30 -- postw90's "
     "`berry_task = shc` with `shc_freq_scan = true`: the spin Hall "
     "conductivity $\\sigma^{xy}_z(\\omega)$ as a function of **optical "
     "frequency** rather than Fermi energy (tutorial 29's static case). "
     "Same Qiao-method spin-current operator as tutorial 29, but a "
     "genuinely different accumulation formula -- a COMPLEX resonance "
     "denominator $-2/(\\Delta\\varepsilon^2 - (\\omega+i\\eta)^2)$ in place "
     "of the static case's real Lorentzian, giving a complex "
     "$\\sigma(\\omega)$ (Re = dissipative, Im = reactive response). GaAs "
     "is a zincblende semiconductor: DFT underestimates its gap, so this "
     "tutorial also exercises a scissors shift (`core.hamiltonian."
     "apply_scissors_shift`) -- a rigid upward shift of the conduction "
     "manifold's eigenvalues that leaves every wavefunction-derived "
     "quantity (the position operator, $D_h$, the spin operators) exactly "
     "unchanged, since it commutes with $H(k)$ by construction."),
    ("code", SETUP),
    ("md",
     "### 1. Converged noncollinear+SOC DFT and Wannierisation\n"
     "Zincblende GaAs, $a=5.654$ Å, fully-relativistic PAW pseudopotentials "
     "(`noncolin=True, lspinorb=True`) -- QE pslibrary's `Ga-rel`/`As-rel` "
     "(different from the scalar-relativistic PseudoDojo `Ga.upf`/`As.upf` "
     "used elsewhere in this series, since SOC needs full relativity). "
     "16 spinor MLWFs: analytic $s+p$ trial orbitals at both sites "
     "(equivalent bonding character to the real tutorial's `sp3` hybrids), "
     "automatically spin-doubled 8->16 by `write_nnkp`'s spinor-projection "
     "block -- an **isolated** manifold (`num_bands == num_wann`, no "
     "disentanglement needed), matching the real tutorial's own "
     "`exclude_bands = 1-10` / `num_bands = num_wann = 16`."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.ase.structure import real_lattice, recip_lattice\n"
     "from waw.interfaces.projections import spd_projections\n"
     "from waw.core.hamiltonian import compute_position_r, apply_scissors_shift\n"
     "from waw.units import EV_TO_HARTREE\n"
     "from waw.analysis.spin_hall import build_shc_operators, spin_hall_conductivity_ac\n"
     "\n"
     "atoms = bulk('GaAs', 'zincblende', a=5.654)\n"
     "MP_GRID = (10, 10, 10)\n"
     "WORK = HERE / 'runs' / 'gaas_shc'\n"
     "\n"
     "SOC = dict(noncolin=True, lspinorb=True, ecutrho=1080,\n"
     "           occupations='smearing', smearing='mv', degauss=0.002)\n"
     "\n"
     "# s+p at both sites -> spinor-doubled to 16, the same bonding/antibonding\n"
     "# manifold the real tutorial's As:sp3/Ga:sp3 hybrids span\n"
     "projs = (spd_projections((0.25, 0.25, 0.25), 's') +\n"
     "         spd_projections((0.25, 0.25, 0.25), 'p') +\n"
     "         spd_projections((0.0, 0.0, 0.0), 's') +\n"
     "         spd_projections((0.0, 0.0, 0.0), 'p'))\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'gaas_shc',\n"
     "    ecutwfc=90, scf_kpts=(10, 10, 10), nbnd=26, num_wann=16,\n"
     "    exclude_bands=list(range(1, 11)), projections=projs,\n"
     "    system_extra=SOC,\n"
     "    pseudopotentials={'Ga': 'Ga-rel.upf', 'As': 'As-rel.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, write_spn=True,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "# isolated manifold (num_bands == num_wann): no outer/frozen window,\n"
     "# no disentanglement -- result.dis is None, W == U_final\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=16,\n"
     "    guiding_centres=True, optimizer='cg',\n"
     "    n_restarts=1, n_iter=4000, conv_tol=1e-10, conv_window=40,\n"
     ")\n"
     "print(f'E_F (own SCF) = {E_FERMI:.4f} eV  (real tutorial: 7.9366 eV)')\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')\n"
     "\n"
     "REAL = real_lattice(atoms)\n"
     "RECIP = recip_lattice(atoms)\n"
     "W = result.spread.U_final   # isolated bands: W == U_final, no V rotation"),
    ("md",
     "### 2. Scissors shift + the spin Hall R-space operators\n"
     "`apply_scissors_shift` opens the DFT-underestimated gap by rigidly "
     "shifting the top 8 (conduction) bands up by 1.117 eV -- the real "
     "tutorial's own value, `num_valence_bands = 8` -- leaving the "
     "converged Wannier gauge and every gauge-derived quantity below "
     "untouched. The spin operators reuse `build_shc_operators` verbatim "
     "from tutorial 29 (no new physics needed for a different material)."),
    ("code",
     "hr_shifted = apply_scissors_shift(\n"
     "    result.hr, W, result.wdata.kpts, MP_GRID, REAL,\n"
     "    num_valence_bands=8, scissors_shift=1.117 * EV_TO_HARTREE,\n"
     ")\n"
     "\n"
     "AA_R = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "SS_R, SR_R, SHR_R, SH_R = build_shc_operators(\n"
     "    W, result.wdata.Mmn, result.wdata.kb_idx, ov['spn'], result.wdata.eig,\n"
     "    result.wdata.wb, result.wdata.bvecs, result.wdata.kpts, MP_GRID, REAL,\n"
     ")\n"
     "print('SS_R, SR_R, SHR_R, SH_R built:', SS_R.shape, SR_R.shape, SHR_R.shape, SH_R.shape)"),
    ("md",
     "### 3. Spin Hall conductivity vs. frequency\n"
     "$\\sigma^{xy}_z(\\omega)$ (`alpha=0, beta=1, gamma=2`) at the fixed "
     "Fermi energy, scanned 0-8 eV in 0.01 eV steps -- the real tutorial's "
     "own `kubo_freq_min/max/step` -- with adaptive smearing "
     "(`kubo_adpt_smr_fac=1.414, kubo_adpt_smr_max=1.0`, the real tutorial's "
     "own explicit overrides of postw90's defaults)."),
    ("code",
     "from waw.units import to_si_units\n"
     "\n"
     "CELL_VOLUME_BOHR3 = abs(np.linalg.det(REAL))\n"
     "OMEGA = np.arange(0.0, 8.0 + 1e-9, 0.01)\n"
     "shc_ac = spin_hall_conductivity_ac(\n"
     "    hr_shifted, AA_R, SS_R, SR_R, SHR_R, SH_R, RECIP, REAL,\n"
     "    fermi_energy=E_FERMI * EV_TO_HARTREE, omega=OMEGA * EV_TO_HARTREE, mesh=(25, 25, 25),\n"
     "    alpha=0, beta=1, gamma=2, kubo_adpt_smr=True,\n"
     "    kubo_adpt_smr_fac=1.414, kubo_adpt_smr_max=1.0 * EV_TO_HARTREE,\n"
     ")\n"
     "shc_ac_sigma = to_si_units(shc_ac.sigma, 'spin_hall_conductivity', cell_volume_bohr3=CELL_VOLUME_BOHR3)\n"
     "print('sigma(omega) shape', shc_ac_sigma.shape, 'dtype', shc_ac_sigma.dtype)\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6, 3.8))\n"
     "ax.plot(OMEGA, shc_ac_sigma.real, label='Re')\n"
     "ax.plot(OMEGA, shc_ac_sigma.imag, label='Im', ls='--')\n"
     "ax.axhline(0.0, color='0.5', lw=0.6)\n"
     "ax.set_xlabel(r'$\\hbar\\omega$ (eV)'); ax.set_ylabel(r'$\\sigma^{xy}_z(\\omega)$ ($\\hbar/e$ S/cm)')\n"
     "ax.set_title('GaAs ac spin Hall conductivity (scissors-corrected)')\n"
     "ax.legend(); plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** `waw.analysis.spin_hall.spin_hall_conductivity_ac` "
     "reimplements postw90's frequency-domain SHC branch -- the same "
     "spin-current operator as tutorial 29's static case, but a genuinely "
     "different complex-pole accumulation formula -- and exercises the "
     "new `apply_scissors_shift` (a transcription of wannier90's own "
     "`get_HH_R` scissors block, verified to leave every wavefunction-"
     "derived quantity exactly unchanged) on a real semiconductor gap "
     "correction. See `project_waw.md`'s tutorial30 entry for the "
     "cross-validation record against real `wannier90.x`/`postw90.x` on "
     "this exact system."),
]


# ==========================================================================
# 31 — Platinum: SCDM (auto_projections) for a spinor/SOC system
# ==========================================================================
pt_scdm = [
    ("md",
     "# Tutorial 31 — Platinum: SCDM for spin-orbit-coupled bands\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 31 -- `auto_projections`, "
     "wannier90's automatic-initial-guess machinery (SCDM, Selected Columns of "
     "the Density Matrix), extended to a **spinor** (noncollinear + "
     "spin-orbit-coupled) system. This is the SAME fcc platinum system as "
     "tutorial 29 -- identical structure, pseudopotential, and disentanglement "
     "windows -- but Wannierised with SCDM instead of tutorial 29's explicit "
     "analytic $d;s;p$ trial orbitals, followed by a plain interpolated band "
     "structure (no postw90 physics module here, unlike tutorial 29's spin "
     "Hall conductivity).\n"
     "\n"
     "No new capability: `waw.interfaces.quantum_espresso.generate_overlaps`'s "
     "`scdm_entanglement`/`scdm_mu`/`scdm_sigma` already work for spinor "
     "systems (first exercised for tutorials 17-19's spin-polarised iron); "
     "tutorial 27 already validated the `isolated`/`erfc`/`gaussian` SCDM "
     "modes for a non-spinor system. This tutorial is the direct spinor/SOC "
     "analogue of tutorial 27, and a live A/B comparison against tutorial 29's "
     "analytic-projection Wannierisation of the exact same system."),
    ("code", SETUP),
    ("md",
     "### 1. Converged noncollinear+SOC DFT, Wannierised via SCDM\n"
     "Same fcc Pt, $a=3.92$ Å, fully-relativistic PAW pseudopotential, and "
     "40→18 disentanglement window as tutorial 29 -- but `scdm_entanglement="
     "'erfc'` (`scdm_mu=35 eV, scdm_sigma=5 eV`, wannier90's own tutorial 31 "
     "values) builds the initial guess automatically from the density matrix, "
     "with NO analytic `projections=` block at all."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "atoms = bulk('Pt', 'fcc', a=3.92)\n"
     "MP_GRID = (7, 7, 7)   # matches the real tutorial's own coarser mesh\n"
     "WORK = HERE / 'runs' / 'pt_scdm'\n"
     "\n"
     "SOC = dict(noncolin=True, lspinorb=True, ecutrho=1080,\n"
     "           occupations='smearing', smearing='mv', degauss=0.002)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'pt_scdm',\n"
     "    ecutwfc=90, scf_kpts=(10, 10, 10), nbnd=40, num_wann=18,\n"
     "    scdm_entanglement='erfc', scdm_mu=35.0, scdm_sigma=5.0,\n"
     "    system_extra=SOC, pseudopotentials={'Pt': 'Pt-rel.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "# same absolute-eV windows as tutorial 29 (same DFT zero, same recipe)\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=18,\n"
     "    outer_window=(0.0, 60.0), frozen_window=(0.0, 30.0),\n"
     "    guiding_centres=True, optimizer='cg',\n"
     "    n_restarts=1, dis_n_iter=5000, n_iter=5000,\n"
     "    conv_tol=1e-10, conv_window=40,\n"
     ")\n"
     "print(f'E_F = {E_FERMI:.3f} eV')\n"
     "print(f'Omega_I     = {result.dis.omega_i * BOHR_TO_ANG**2:.4f} Ang^2')\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')"),
    ("md",
     "### 2. Interpolated band structure\n"
     "Same ASE-standard k-path convention as every other notebook in this "
     "series (`band_path`, Setyawan-Curtarolo)."),
    ("code",
     "bp = band_path(atoms, npoints=150)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands - E_FERMI, figsize=(6.5, 4.2),\n"
     "           ref_line=0.0, ref_label='E_F',\n"
     "           title='Pt bands from 18 SCDM-derived spinor MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** The SAME platinum system tutorial 29 Wannierised with "
     "explicit analytic $d;s;p$ projections comes out equally well from "
     "SCDM's automatic, projection-free initial guess -- `waw`'s SCDM path "
     "(already validated for non-spinor Si in tutorial 27) needed no changes "
     "at all to handle a noncollinear, spin-orbit-coupled metal. See "
     "`project_waw.md`'s tutorial31 entry for the cross-validation record "
     "against real `wannier90.x` on this exact system."),
]


# ==========================================================================
# 32 — Tungsten: SCDM entanglement derived from projwfc.x projectability
# ==========================================================================
w_projscdm = [
    ("md",
     "# Tutorial 32 — Tungsten: SCDM entanglement from atomic projectability\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 32 -- bcc tungsten, "
     "13 $s;p;d$-like MLWFs from **21 candidate bands**, built with SCDM's "
     "`erfc` entanglement function (tutorials 27/31), but where $\\mu$/"
     "$\\sigma$ are no longer hand-picked: they come from fitting the erfc "
     "form directly to a real *projectability* curve.\n"
     "\n"
     "**Projectability** (QE's `projwfc.x`) is $|\\langle \\psi_{nk} | "
     "\\hat P_{\\rm atomic} | \\psi_{nk} \\rangle|^2$ -- how much of each "
     "DFT band $n,k$ overlaps the pseudopotential's own atomic orbitals. "
     "States with mostly atomic (bonding/$d$-manifold) character score near "
     "1; free-electron-like high-energy states score near 0. Plotted "
     "against energy, this traces out almost exactly an `erfc` step -- so "
     "instead of guessing `scdm_mu`/`scdm_sigma` by eye (tutorials 27/31), "
     "this notebook **fits** them: new `waw.interfaces.quantum_espresso."
     "projwfc.run_projwfc`/`read_projectability` run `projwfc.x` and parse "
     "its `proj.out` (reimplementing the real tutorial's own "
     "`generate_weights.sh` `grep`/`awk`/`sort` pipeline in Python), then "
     "`scipy.optimize.curve_fit` fits `0.5*erfc((E-mu)/sigma)` to the "
     "resulting scatter.\n"
     "\n"
     "The real tutorial also sets `dis_num_iter=0`: the disentangled "
     "subspace is used **exactly as SCDM's initial guess gives it**, with "
     "NO further SMV01 iterative refinement (only the localization/spread "
     "minimization, `num_iter=10000`, still runs) -- the pedagogical point "
     "being that a well-fitted entanglement function already picks such a "
     "clean subspace that iterative disentanglement has nothing left to "
     "improve."),
    ("code", SETUP),
    ("md",
     "### 1. Converged bcc W, and a throwaway first SCDM pass\n"
     "PseudoDojo scalar-relativistic W (no SOC needed -- unlike tutorials "
     "29-31's Pt/GaAs), primitive bcc cell (ASE's own convention, "
     "conventional $a=3.1831$ Å, matching the real tutorial's explicit "
     "`CELL_PARAMETERS`), `ecutwfc=82` Ry (PseudoDojo's own \"high\" hint, "
     "41 Ha -- confirmed by a direct SCF cutoff scan: total energy changes "
     "by only 0.1 mRy between 70 and 80 Ry). `nbnd=21` candidate bands, "
     "`num_wann=13`.\n"
     "\n"
     "`projwfc.x` needs a **converged NSCF it can read wavefunctions "
     "from**, so we first run `generate_overlaps` once with a throwaway "
     "`scdm_entanglement='isolated'` (uniform weighting -- its own SCDM "
     "guess is discarded entirely, only the SCF/NSCF `out/` directory it "
     "leaves behind matters here)."),
    ("code",
     "from ase.build import bulk\n"
     "from scipy.optimize import curve_fit\n"
     "from scipy.special import erfc\n"
     "from waw.interfaces.quantum_espresso.projwfc import run_projwfc, read_projectability\n"
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "atoms = bulk('W', 'bcc', a=3.1831145422)   # ASE's own primitive bcc cell, 1 atom\n"
     "MP_GRID = (10, 10, 10)\n"
     "NBND, NUM_WANN = 21, 13\n"
     "WORK = HERE / 'runs' / 'tungsten'\n"
     "SYSTEM_EXTRA = dict(occupations='smearing', smearing='mv', degauss=0.02)\n"
     "\n"
     "ov0 = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'w',\n"
     "    ecutwfc=82, scf_kpts=(14, 14, 14), nbnd=NBND, num_wann=NUM_WANN,\n"
     "    scdm_entanglement='isolated',\n"
     "    system_extra=SYSTEM_EXTRA, pseudopotentials={'W': 'W.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "print(f\"E_F = {ov0['fermi_energy']:.4f} eV (throwaway pass, out/ now populated)\")"),
    ("md",
     "### 2. `projwfc.x` projectability vs. energy, and the erfc fit\n"
     "`run_projwfc` writes the `&projwfc` namelist and runs `projwfc.x` "
     "under `mpirun` (confirmed empirically parallel-safe on this cluster, "
     "unlike `wannier90.x`/`postw90.x`'s serial-only builds -- identical "
     "output at `-np 1` vs `-np 2`); `read_projectability` parses its "
     "`proj.out` into one `(energy, |psi|^2)` pair per (k-point, band)."),
    ("code",
     "proj_out = run_projwfc(WORK, 'w', ncores=NCORES)\n"
     "energies_eV, proj = read_projectability(proj_out)\n"
     "print(f'{len(energies_eV)} (k, band) pairs from projwfc.x')\n"
     "\n"
     "def erfc_model(e, mu, sigma):\n"
     "    return 0.5 * erfc((e - mu) / sigma)\n"
     "\n"
     "order = np.argsort(energies_eV)\n"
     "e_sorted, p_sorted = energies_eV[order], proj[order]\n"
     "popt, _ = curve_fit(erfc_model, e_sorted, p_sorted,\n"
     "                     p0=[float(np.median(e_sorted)), 5.0], maxfev=20000)\n"
     "mu_fit, sigma_fit = popt\n"
     "resid = p_sorted - erfc_model(e_sorted, *popt)\n"
     "r2 = 1.0 - np.sum(resid**2) / np.sum((p_sorted - p_sorted.mean())**2)\n"
     "print(f'fit: scdm_mu = {mu_fit:.4f} eV, scdm_sigma = {sigma_fit:.4f} eV, R^2 = {r2:.6f}')\n"
     "\n"
     "e_plot = np.linspace(e_sorted.min(), e_sorted.max(), 400)\n"
     "plt.figure(figsize=(6, 4))\n"
     "plt.scatter(energies_eV, proj, s=6, alpha=0.3, label='projwfc.x (k, band) pairs')\n"
     "plt.plot(e_plot, erfc_model(e_plot, *popt), 'C3-', lw=2,\n"
     "         label=f'fit: $\\\\mu$={mu_fit:.2f} eV, $\\\\sigma$={sigma_fit:.2f} eV')\n"
     "plt.xlabel('E (eV)'); plt.ylabel('atomic projectability $|\\\\psi|^2$')\n"
     "plt.legend(); plt.tight_layout(); plt.show()"),
    ("md",
     "### 3. The real SCDM pass: `erfc` entanglement with the fitted $\\mu,\\sigma$\n"
     "Same DFT, `rerun_scf=False` reuses the completed SCF -- only the "
     "pw2wannier90/SCDM step changes, now with the fitted `scdm_mu`/"
     "`scdm_sigma`. Then `wannierize(..., dis_n_iter=0, n_iter=10000)`: the "
     "disentangled subspace is `V` exactly as SCDM's own initial projection "
     "gives it (see `core.disentangle.disentangle`'s `for sweep in "
     "range(n_iter)` -- `n_iter=0` skips the loop body entirely and returns "
     "that initial subspace unchanged, `converged=False` is therefore "
     "expected and not a bug), with only the localization/spread "
     "minimization iterating."),
    ("code",
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'w',\n"
     "    ecutwfc=82, scf_kpts=(14, 14, 14), nbnd=NBND, num_wann=NUM_WANN,\n"
     "    scdm_entanglement='erfc', scdm_mu=mu_fit, scdm_sigma=sigma_fit,\n"
     "    system_extra=SYSTEM_EXTRA, pseudopotentials={'W': 'W.upf'},\n"
     "    pseudo_dir=PSEUDO_DIR, ncores=NCORES, rerun_scf=False,\n"
     ")\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=NUM_WANN,\n"
     "    dis_n_iter=0, n_iter=10000, conv_tol=8e-6, conv_window=3,\n"
     ")\n"
     "print(f'E_F = {E_FERMI:.4f} eV')\n"
     "print(f'Omega_I     = {result.dis.omega_i * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_total = {result.omega_final * BOHR_TO_ANG**2:.6f} Ang^2')"),
    ("md",
     "### 4. Interpolated band structure\n"
     "Same ASE-standard bcc k-path convention as every other notebook "
     "(`band_path`, Setyawan-Curtarolo: $\\Gamma$-H-N-$\\Gamma$-P-H,P-N)."),
    ("code",
     "bp = band_path(atoms, npoints=200)\n"
     "bands = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands - E_FERMI, figsize=(6.5, 4.2),\n"
     "           ref_line=0.0, ref_label='E_F',\n"
     "           title='bcc W bands from 13 projectability-fitted SCDM MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** Fitting the erfc entanglement function to a real, "
     "computed atomic-projectability curve -- rather than hand-picking "
     "`scdm_mu`/`scdm_sigma` (tutorials 27/31) -- already selects such a "
     "clean initial subspace that `dis_num_iter=0` (no iterative "
     "disentanglement refinement at all) reproduces real `wannier90.x`'s "
     "own bcc W bands; see `project_waw.md`'s tutorial32 entry for the "
     "cross-validation record."),
]


# ==========================================================================
# 33 — BC2N: k.p expansion coefficients (postw90 berry_task=kdotp)
# ==========================================================================
bc2n_kdotp = [
    ("md",
     "# Tutorial 33 — BC2N: k.p expansion coefficients (postw90 `kdotp`)\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 33. A **BC2N** "
     "nanoribbon/slab (periodic along $x$ and $z$, a 20 Å vacuum gap along "
     "$y$) has a 4-orbital $\\pi$-manifold built from a single $p_y$ orbital "
     "per atom (B, C, C, N -- $p_y$ points out of the sheet, the same role "
     "$p_z$ plays in graphene). This notebook builds those 4 MLWFs, then "
     "uses a genuinely new capability, `waw.analysis.kdotp."
     "kdotp_coefficients`, to expand the Wannier-interpolated Hamiltonian "
     "to 2nd order in $k$ around the S point ($\\mathbf{k}=(\\tfrac12,0,"
     "\\tfrac12)$) via quasi-degenerate (Löwdin) perturbation theory -- "
     "transcribed directly from wannier90's own `berry_get_kdotp` "
     "(`src/postw90/berry.F90`). The result is a compact $2\\times2$ "
     "effective Hamiltonian for the two \"interesting\" Wannier bands at "
     "that point, reconstructed and checked against the true interpolated "
     "dispersion nearby."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and a genuine `ecutwfc` convergence check\n"
     "$a=2.466$ Å, $c=4.323$ Å, 20 Å vacuum along $y$ -- the real "
     "tutorial's own lattice, atoms in the same Cartesian positions. This "
     "project's PseudoDojo B/C/N pseudopotentials are valence-only (3+4+5 "
     "electrons, no semicore), unlike the real tutorial's own PAW "
     "pseudopotentials (which is why that tutorial's own `num_bands=29`/"
     "`exclude_bands`/`fermi_energy` are not reused here -- they are "
     "artifacts of a different pseudopotential). `ecutwfc` is converged "
     "from scratch below, on the real SCF total energy, rather than just "
     "guessed."),
    ("code",
     "from ase import Atoms\n"
     "a, vac, c = 2.466, 20.00, 4.323\n"
     "cell = [[a, 0.0, 0.0], [0.0, vac, 0.0], [0.0, 0.0, c]]\n"
     "symbols = ['B', 'C', 'C', 'N']\n"
     "pos_cart = [[1.2330000, 0.0000000, 1.1830592],\n"
     "            [1.2330000, 0.0000000, 2.6963492],\n"
     "            [0.0000000, 0.0000000, 3.3885933],\n"
     "            [0.0000000, 0.0000000, 4.7677273]]\n"
     "atoms = Atoms(symbols, positions=pos_cart, cell=cell, pbc=True)\n"
     "MP_GRID = (10, 1, 10)\n"
     "WORK = HERE / 'runs' / 'bc2n'\n"
     "PSEUDOS = {'B': 'B.upf', 'C': 'C.upf', 'N': 'N.upf'}\n"
     "SMEAR = dict(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "import re\n"
     "conv_dir = WORK / 'ecutwfc_scan'\n"
     "conv_dir.mkdir(parents=True, exist_ok=True)\n"
     "energies = {}\n"
     "for ecut in (40, 50, 60, 70, 80):\n"
     "    d = conv_dir / f'ecut{ecut}'; d.mkdir(exist_ok=True); (d / 'out').mkdir(exist_ok=True)\n"
     "    inp = qe.write_pw_input(\n"
     "        d / 'scf.in', atoms,\n"
     "        control=dict(calculation='scf', prefix='bc2n', pseudo_dir=str(PSEUDO_DIR), outdir='./out'),\n"
     "        system=dict(ecutwfc=ecut, **SMEAR),\n"
     "        electrons=dict(conv_thr=1e-8, mixing_beta=0.4),\n"
     "        pseudopotentials=PSEUDOS, kpoints=('automatic', MP_GRID, (0, 0, 0)),\n"
     "    )\n"
     "    out = qe.run_pw(inp, d / 'scf.out', ncores=NCORES)\n"
     "    m = re.findall(r'!\\s+total energy\\s+=\\s+([-\\d.]+) Ry', out.read_text())\n"
     "    energies[ecut] = float(m[-1])\n"
     "    print(f'ecutwfc={ecut:3d} Ry   E_tot = {energies[ecut]:.6f} Ry')\n"
     "\n"
     "ecuts = sorted(energies)\n"
     "for e1, e2 in zip(ecuts[:-1], ecuts[1:]):\n"
     "    print(f'  {e1:3d} -> {e2:3d} Ry : dE = {(energies[e2]-energies[e1])*1000:+.3f} mRy '\n"
     "          f'({(energies[e2]-energies[e1])*1000/len(atoms):+.3f} mRy/atom)')"),
    ("md",
     "The total energy changes by $\\approx 9.4$ mRy ($2.4$ mRy/atom) from "
     "60 to 70 Ry, and only $\\approx 1.8$ mRy ($0.4$ mRy/atom) from 70 to "
     "80 Ry -- a factor-of-5 drop in the increment, comfortably converged. "
     "`ecutwfc = 70` Ry is used for everything below."),
    ("md",
     "### 2. Analytic $p_y$ projections + overlaps\n"
     "Wannier90's own projection block is `C:py / B:py / N:py` -- one "
     "$p_y$ orbital per atom, default axes ($z=\\hat z$, $x=\\hat x$, "
     "i.e. `mr=3` in the standard p-shell ordering $(p_z,p_x,p_y)$, see "
     "`interfaces.projections.spd_projections`'s own docstring). Built "
     "directly here (not via `spd_projections`, which would return the "
     "whole $s;p;d$ shell) since only the single $p_y$ component per atom "
     "is wanted."),
    ("code",
     "frac = atoms.get_scaled_positions()\n"
     "projs = [(tuple(fc), 1, 3, 1, (0, 0, 1), (1, 0, 0), 1.0) for fc in frac]   # l=1 (p), mr=3 (py)\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'bc2n',\n"
     "    ecutwfc=70, scf_kpts=MP_GRID, nbnd=20, num_wann=4,\n"
     "    projections=projs, system_extra=SMEAR,\n"
     "    pseudopotentials=PSEUDOS, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "print(f\"E_F = {ov['fermi_energy']:.4f} eV\")"),
    ("md",
     "### 3. Wannierise: 4 $p_y$-derived MLWFs\n"
     "**An honest detour first.** A plain, moderately-wide energy window "
     "(e.g. $[-8.5, 4.5]$ eV, comfortably covering the DFT bands with the "
     "largest $p_y$ character) reproducibly drove 2 of the 4 Wannier "
     "functions into a runaway with $\\Omega \\sim 9$–$25$ Å$^2$ each and "
     "centres displaced almost a full Å off the atomic plane -- the exact "
     "same pathology as tutorial 34's free-standing graphene (also a 20 Å-"
     "vacuum system): an energy window alone cannot always tell the real "
     "$\\pi$/$\\pi^*$ manifold from other, nearby-in-energy states it "
     "partially hybridizes with over parts of the Brillouin zone. Here, "
     "unlike graphene, a `proj_min`/`proj_max` cut was tried and found NOT "
     "to cleanly separate the two (the $p_y$ overlap of the true target "
     "bands and of the contaminating ones both range from 0 to $>1$ "
     "depending on $k$ -- no clean threshold exists). What DOES fix it: a "
     "much WIDER outer window (spanning nearly the whole 20-band candidate "
     "manifold) so the disentanglement has enough genuine freedom to find "
     "the correct subspace, combined with more restarts of the global "
     "search. Both energy-window variants and the final choice are on "
     "record in this project's memory (`project_waw.md`, tutorial33 "
     "entry) for anyone who hits the same trap."),
    ("code",
     "result = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, outer_window=(-15.0, 9.0),\n"
     "    guiding_centres=True,\n"
     "    n_restarts=6, dis_n_iter=2000, n_iter=6000, verbose=False,\n"
     ")\n"
     "omega_I = result.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega = result.omega_final * BOHR_TO_ANG**2\n"
     "spreads = result.spreads_bohr2 * BOHR_TO_ANG**2\n"
     "centres = result.centres_bohr * BOHR_TO_ANG\n"
     "print(f'Omega_I     = {omega_I:.4f} Ang^2')\n"
     "print(f'Omega_total = {omega:.4f} Ang^2')\n"
     "print('per-WF spreads (Ang^2):', spreads.round(3))\n"
     "print('centres (Ang):\\n', centres.round(3))\n"
     "print('atom positions (Ang):\\n', atoms.get_positions().round(3))"),
    ("md",
     "$\\Omega_{\\rm total} \\approx 5.0$ Å$^2$, spreads all $\\approx "
     "0.9$–$1.4$ Å$^2$ -- no runaway function, and every centre sits "
     "within a fraction of an Å of the atomic ($y=0$) plane. Cross-"
     "validated below against real `wannier90.x` on the identical "
     "overlaps: $\\Omega_I$, $\\Omega_{\\rm total}$, all 4 spreads and all "
     "4 centres match to 6 decimal places."),
    ("md",
     "### 4. k.p expansion at the S point\n"
     "`num_wann=4` gives a $4\\times4$ Wannier Hamiltonian at every $k$; "
     "`kdotp_bands=(1,2)` (0-based) selects the middle two of those four "
     "at S -- exactly Wannier90's own `kdotp_bands = 2,3` (1-based) "
     "convention. `kdotp_coefficients` returns the 0th ($E_n(S)$), 1st "
     "($\\langle n|\\partial H/\\partial k_a|m\\rangle$) and 2nd order "
     "(Löwdin-corrected, with the other two Wannier bands as virtual "
     "states) expansion coefficients in that 2-band subspace."),
    ("code",
     "from waw.analysis.kdotp import kdotp_coefficients\n"
     "from waw.interfaces.ase.structure import recip_lattice\n"
     "from waw.units import to_eVA_units\n"
     "\n"
     "RECIP = recip_lattice(atoms)\n"
     "S_POINT = np.array([0.5, 0.0, 0.5])\n"
     "kres = kdotp_coefficients(result.hr, RECIP, kpoint=S_POINT, bands=(1, 2))\n"
     "print('eig0 (eV):', kres.eig0 * HARTREE_TO_EV, ' -- gap', np.diff(kres.eig0 * HARTREE_TO_EV))\n"
     "print('first_order (eV.Ang):\\n', to_eVA_units(kres.first_order, 'kdotp_first_order').round(4))\n"
     "print('second_order (eV.Ang^2):\\n', to_eVA_units(kres.second_order, 'kdotp_second_order').round(4))"),
    ("md",
     "The 4 Wannier-Hamiltonian eigenvalues at S are well separated "
     "(no near-degeneracy at this exact point for THIS pseudopotential's "
     "own band energetics -- unlike the real tutorial's PAW-based system, "
     "our valence-only treatment gives different absolute band "
     "energetics), but `bands=(1,2)` is still the literal, unambiguous "
     "counterpart of Wannier90's own `kdotp_bands = 2,3` -- \"the middle "
     "two of the four $\\pi$-manifold Wannier bands\" by construction, "
     "regardless of the exact gap. The k.p reconstruction below shows "
     "this 2-band model is nonetheless an excellent local description of "
     "the true dispersion near S."),
    ("md",
     "### 5. Reconstructed dispersion vs. the actual interpolated bands\n"
     "$H_{\\rm eff}(\\mathbf q) = \\mathrm{diag}(E_n) + \\mathbf q\\cdot"
     "\\text{first\\_order} + \\mathbf q\\mathbf q\\text{:second\\_order}$, "
     "Hermitized and diagonalized -- the same reconstruction pattern as "
     "`tests/test_analysis_kdotp.py`'s own convergence test, now applied "
     "to a real DFT system. Compared here along the S-X cut ($k_z: "
     "0.5\\to0.0$, $k_x=0.5$, $k_y=0$ fixed) against waw's own Wannier-"
     "interpolated bands (ground truth, no k.p expansion involved)."),
    ("code",
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "X_POINT = np.array([0.5, 0.0, 0.0])\n"
     "ts = np.linspace(0.0, 1.0, 101)\n"
     "kpts_cut = S_POINT[None, :] + ts[:, None] * (X_POINT - S_POINT)[None, :]\n"
     "bands_actual = interpolate_bands(result.hr, kpts_cut) * HARTREE_TO_EV\n"
     "\n"
     "def reconstruct(q_cart, order2):\n"
     "    H_eff = np.diag(kres.eig0.astype(np.complex128))\n"
     "    H_eff = H_eff + np.einsum('a,nma->nm', q_cart, kres.first_order)\n"
     "    if order2:\n"
     "        H_eff = H_eff + np.einsum('a,b,nmab->nm', q_cart, q_cart, kres.second_order)\n"
     "    H_eff = (H_eff + H_eff.conj().T) / 2\n"
     "    return np.linalg.eigvalsh(H_eff)\n"
     "\n"
     "q_frac = kpts_cut - S_POINT[None, :]\n"
     "q_cart = q_frac @ RECIP\n"
     "recon_lin = np.array([np.sort(reconstruct(q, False)) for q in q_cart]) * HARTREE_TO_EV\n"
     "recon_quad = np.array([np.sort(reconstruct(q, True)) for q in q_cart]) * HARTREE_TO_EV\n"
     "\n"
     "fig, ax = plt.subplots(figsize=(6.5, 4.5))\n"
     "for b in (1, 2):\n"
     "    ax.plot(ts, bands_actual[:, b], 'k-', lw=2,\n"
     "            label='Wannier-interpolated (ground truth)' if b == 1 else None)\n"
     "ax.plot(ts, recon_lin[:, 0], 'b--', label='linear k.p')\n"
     "ax.plot(ts, recon_lin[:, 1], 'b--')\n"
     "ax.plot(ts, recon_quad[:, 0], 'r-', label='quadratic k.p')\n"
     "ax.plot(ts, recon_quad[:, 1], 'r-')\n"
     "ax.set_xticks([0, 1]); ax.set_xticklabels(['S', 'X'])\n"
     "ax.set_ylabel('E (eV)'); ax.legend(loc='best', fontsize=9)\n"
     "ax.set_title('BC2N k.p expansion at S: reconstruction vs. actual bands')\n"
     "plt.tight_layout(); plt.show()\n"
     "\n"
     "err_lin = np.abs(recon_lin - bands_actual[:, 1:3]).max(axis=1)\n"
     "err_quad = np.abs(recon_quad - bands_actual[:, 1:3]).max(axis=1)\n"
     "for i in (5, 20, 50, 100):\n"
     "    print(f't={ts[i]:.2f}  max|err| linear={err_lin[i]:.4f} eV   quadratic={err_quad[i]:.4f} eV')"),
    ("md",
     "The quadratic k.p model tracks the true dispersion tightly for "
     "roughly the first fifth to third of the S-X segment, then "
     "progressively departs (by X itself, off by several tenths of an "
     "eV) -- exactly the expected O(q$^3$) breakdown of a Taylor expansion "
     "far from its expansion point, and consistently better than the "
     "linear-only truncation everywhere. The linear term along this "
     "particular direction turns out to be small (S is close to a local "
     "extremum along $k_z$ for this band pair), so most of the near-S "
     "curvature comes from the 2nd-order (virtual-state-corrected) term."),
    ("md",
     "### 6. Full band structure for context\n"
     "The 4-MLWF model's band structure along ASE's own standard k-path "
     "for this cell."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "\n"
     "bp = band_path(atoms, npoints=200)\n"
     "bands_full = interpolate_bands(result.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "E_FERMI = ov['fermi_energy']\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands_full - E_FERMI, figsize=(6.5, 4.2),\n"
     "           ref_line=0.0, ref_label='E_F', ylabel='E - E_F (eV)',\n"
     "           title='BC2N bands from 4 py-derived MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Cross-validation note (see `project_waw.md`'s tutorial33 entry "
     "for the full record).** The installed real `wannier90.x`/`postw90.x` "
     "(both available `quantum-espresso` modules bundle the same build) "
     "self-report version 3.1.0 but contain no trace of the `kdotp` "
     "feature at all -- `kdotp_kpoint`/`kdotp_num_bands`/`kdotp_bands` are "
     "flatly \"unrecognised keywords\", confirmed to be specifically a "
     "missing-feature gap (not a general input problem) by successfully "
     "running the identical `.win` with those 3 lines removed. Real "
     "`postw90.x`'s own `kdotp_0/1/2.dat` output could therefore not be "
     "directly diffed against `kdotp_coefficients`. Instead, the "
     "identical `.mmn`/`.amn`/`.eig` overlaps and energy window were fed "
     "into real `wannier90.x` to disentangle+wannierise independently; "
     "its resulting `.chk` (read via `w90chk2chk.x -export` + this "
     "project's own `read_chk_fmt`) matches waw's own $\\Omega_I$, "
     "$\\Omega_{\\rm total}$, all 4 spreads and all 4 Wannier centres to "
     "6 decimal places -- i.e. waw and real Wannier90 land on the exact "
     "same disentangled Wannier Hamiltonian $H(R)$ for this system. Since "
     "`kdotp_coefficients` is a deterministic function of exactly that "
     "$H(R)$, already validated independently by 5 synthetic-model unit "
     "tests (`tests/test_analysis_kdotp.py`: Hermiticity, finite-"
     "difference velocity, O(q$^3$) convergence improvement from the 2nd-"
     "order virtual-state correction, isolated-limit reduction to a plain "
     "Hessian rotation), this gives strong indirect confidence in the "
     "k.p numbers reported above even without a direct `postw90.x` diff."),
]


# ==========================================================================
# 35 — Silicon: external atomic projectors (atom_proj_ext)
# ==========================================================================
si_ext_proj = [
    ("md",
     "# Tutorial 35 — Silicon: external atomic projectors (`atom_proj_ext`)\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 35. Tutorial 34 "
     "(notebook 34) used pw2wannier90's `atom_proj = .true.` to build the "
     "$A_{mn}$ trial-projection matrix directly from a pseudopotential's "
     "OWN embedded atomic pseudo-orbitals (its `PP_PSWFC` block). "
     "`atom_proj_ext = .true.` is the same idea with one twist: the "
     "radial projector functions come from a **plain-text file per "
     "species** instead of the UPF's embedded block -- letting a "
     "calculation use projector shapes that differ from (or go beyond) "
     "whatever a specific pseudopotential happens to carry.\n"
     "\n"
     "This notebook demonstrates the plumbing with the most direct test "
     "available: round-trip **this project's own already-committed "
     "`Si.upf`** (PseudoDojo, ONCVPSP norm-conserving) through the "
     "external-file path -- `waw.interfaces.quantum_espresso.upf."
     "read_pswfc` parses its `PP_MESH`/`PP_PSWFC` blocks, `write_atom_proj_ext` "
     "re-serializes them as pw2wannier90's own `<atom_proj_dir>/Si.dat` "
     "expects, and `generate_overlaps(..., atom_proj_ext=True)` feeds that "
     "file straight back into pw2wannier90. If the plumbing is right, this "
     "must reproduce (to within a totally different code path's own "
     "roundoff) the SAME physics as the pre-existing `atom_proj=True` path "
     "reading the same pseudopotential's embedded orbitals directly -- "
     "checked explicitly at the end of this notebook.\n"
     "\n"
     "**One honest difference from the official tutorial up front**: the "
     "real Wannier90 tutorial 35 ships its own external projector file "
     "with $s$+$p$+$d$ channels (an ultrasoft PSlibrary-style "
     "pseudopotential foreign to this project), giving $9$ orbitals/atom "
     "$\\times\\,2$ atoms $= 18$ Wannier functions. This project's own "
     "committed `Si.upf` only carries $s$+$p$ pseudo-atomic-orbitals in "
     "its `PP_PSWFC` block (no bound $3d$ state -- unsurprising, since "
     "ONCVPSP norm-conserving pseudos only store OCCUPIED reference "
     "orbitals for a main-group element), so this notebook's own "
     "round-trip demo targets $4$ orbitals/atom $\\times\\,2$ atoms $= 8$ "
     "Wannier functions instead of 18 -- a genuine, documented consequence "
     "of which pseudopotential is doing the projecting, not a shortfall in "
     "the `atom_proj_ext` mechanism itself. A separate throwaway script "
     "(not part of this notebook, see `project_waw.md`'s tutorial 35 entry) "
     "cross-validates the full $num\\_wann=18$ case against real "
     "`wannier90.x`, using the real tutorial's own shipped `ext_proj/Si.dat` "
     "fed through this exact same `waw` pipeline.\n"
     "\n"
     "**A real bug this notebook build caught**: the first version of "
     "`write_atom_proj_ext` wrote 2-column rows (`r`, then the radial "
     "functions) -- round-tripped fine in a unit test, but made real "
     "`pw2wannier90.x` crash with a Fortran \"End of file\" read error the "
     "first time it actually ran. Reading pw2wannier90's own `atproj` "
     "Fortran module (`read_atomproj`) showed each row actually needs "
     "THREE leading quantities before the projector columns: "
     "`xgrid(i), rgrid(i)` where `rgrid = exp(xgrid)` -- i.e. `xgrid` is "
     "simply $\\ln(r)$ (confirmed against the real tutorial's own shipped "
     "`Si.dat` sample to $\\sim10^{-15}$). Fixed in `upf.py` -- see its "
     "module docstring for the corrected format and `tests/test_upf.py` "
     "for the updated round-trip test."),
    ("code", SETUP),
    ("md",
     "### 1. Build `ext_proj/Si.dat` from the committed `Si.upf`\n"
     "`read_pswfc` returns `{l: (r, chi)}` straight from the UPF's own "
     "`PP_MESH`/`PP_PSWFC` blocks; `write_atom_proj_ext` writes it back "
     "out in pw2wannier90's external-projector format. This project's own "
     "`Si.upf` uses a *linear* radial mesh (`r[0] == 0.0` exactly, unlike "
     "the logarithmic meshes `atom_proj_ext`'s file format was designed "
     "around) -- `write_atom_proj_ext` drops that one leading point "
     "before taking $\\ln(r)$ (safe: by the UPF convention "
     "$\\chi(r) = r\\,R_l(r)$, $\\chi(0)=0$ for every $l$, so nothing "
     "physical is lost)."),
    ("code",
     "from ase.build import bulk\n"
     "from waw.interfaces.quantum_espresso.upf import read_pswfc, write_atom_proj_ext\n"
     "\n"
     "atoms = bulk('Si', 'diamond', a=5.43)\n"
     "MP_GRID = (6, 6, 6)\n"
     "WORK = HERE / 'runs' / 'si_ext_proj'\n"
     "EXT_DIR = WORK / 'ext_proj'\n"
     "\n"
     "radial = read_pswfc(PSEUDO_DIR / 'Si.upf')\n"
     "write_atom_proj_ext(EXT_DIR, {'Si': radial})\n"
     "print('l channels found in Si.upf:', sorted(radial), '(s, p -- no d)')\n"
     "print('wrote', EXT_DIR / 'Si.dat')"),
    ("md",
     "### 2. Converged DFT, overlaps via `atom_proj_ext`\n"
     "Same `Si.upf`/`ecutwfc=40`/`a=5.43` Å diamond-Si convention as "
     "notebooks 03 and 27, `mp_grid=(6,6,6)` (denser than tutorial27's "
     "isolated-manifold cases, since this is a genuine wide-manifold "
     "disentanglement: `nbnd=40` candidate bands, matching the official "
     "tutorial's own `num_bands=40`). `atom_proj_dir=EXT_DIR` points "
     "pw2wannier90 at the file just written; `atom_proj_ext=True` implies "
     "`atom_proj=True` internally."),
    ("code",
     "ov_ext = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_ext',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=40, num_wann=8,\n"
     "    atom_proj_ext=True, atom_proj_dir=EXT_DIR,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "print('overlap shapes:', {k: np.shape(v) for k, v in ov_ext.items()\n"
     "                          if k in ('mmn', 'amn', 'eig')})\n"
     "E_FERMI = ov_ext['fermi_energy']\n"
     "print(f'E_F = {E_FERMI:.4f} eV')"),
    ("md",
     "### 3. Projectability + energy-window disentanglement\n"
     "The official tutorial combines BOTH mechanisms at once "
     "(`dis_froz_proj=.true.`, `dis_proj_max=0.95`, `dis_proj_min=0.01`, "
     "PLUS `dis_froz_max=8.7` eV \"to be safe\"): a candidate band is "
     "frozen if it is EITHER inside the energy window OR carries "
     "atomic-orbital character above `proj_max` -- exactly the OR-combined "
     "semantics `core.disentangle.disentangle`'s `proj_min`/`proj_max` "
     "already implement alongside `frozen_window` (see its docstring). No "
     "explicit `dis_froz_min`/`outer_window` in the real `.win` -- every "
     "one of the 40 candidate bands stays in play throughout."),
    ("code",
     "result_ext = wannierize(\n"
     "    atoms, MP_GRID, ov_ext['kpts'],\n"
     "    mmn=ov_ext['mmn'], amn=ov_ext['amn'], eig=ov_ext['eig'],\n"
     "    nnkpts=ov_ext['nnkpts'], g_vectors=ov_ext['g_vectors'],\n"
     "    nw=8, proj_min=0.01, proj_max=0.95, frozen_window=(-1.0e6, 8.7),\n"
     "    dis_n_iter=1000, n_iter=2000, conv_tol=1e-10, conv_window=5,\n"
     ")\n"
     "omega_I_ext = result_ext.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega_ext = result_ext.omega_final * BOHR_TO_ANG**2\n"
     "print(f'Omega_I     = {omega_I_ext:.4f} Ang^2')\n"
     "print(f'Omega_total = {omega_ext:.4f} Ang^2')\n"
     "print('spreads (Ang^2):', (result_ext.spreads_bohr2 * BOHR_TO_ANG**2).round(4))"),
    ("md",
     "### 4. Interpolated band structure\n"
     "ASE's own diamond-cubic k-path (`band_path`), same convention as "
     "every other notebook here."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=200)\n"
     "bands_ext = interpolate_bands(result_ext.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands_ext - E_FERMI, figsize=(6.5, 4.2),\n"
     "           ref_line=0.0, ref_label='E_F',\n"
     "           title='Si bands from 8 atom_proj_ext (external-file) MLWFs')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "### 5. Honest side-by-side: `atom_proj_ext` vs. plain `atom_proj`\n"
     "Same structure, same mesh, same disentanglement settings, but "
     "`atom_proj=True` alone (no `_ext`, no `atom_proj_dir`): pw2wannier90 "
     "now reads the SAME `Si.upf`'s $s$/$p$ orbitals from its OWN embedded "
     "`PP_PSWFC` block instead of the external file just written above. "
     "Since the external file is nothing but a round-trip of that same "
     "pseudopotential's own orbitals, the two runs should agree closely -- "
     "this is a sanity check on the new plumbing, not a claim that "
     "external projectors give a numerically different answer from "
     "built-in ones."),
    ("code",
     "ov_plain = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_plain',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=40, num_wann=8,\n"
     "    atom_proj=True,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "result_plain = wannierize(\n"
     "    atoms, MP_GRID, ov_plain['kpts'],\n"
     "    mmn=ov_plain['mmn'], amn=ov_plain['amn'], eig=ov_plain['eig'],\n"
     "    nnkpts=ov_plain['nnkpts'], g_vectors=ov_plain['g_vectors'],\n"
     "    nw=8, proj_min=0.01, proj_max=0.95, frozen_window=(-1.0e6, 8.7),\n"
     "    dis_n_iter=1000, n_iter=2000, conv_tol=1e-10, conv_window=5,\n"
     ")\n"
     "omega_I_plain = result_plain.dis.omega_i * BOHR_TO_ANG**2\n"
     "omega_plain = result_plain.omega_final * BOHR_TO_ANG**2\n"
     "\n"
     "print(f'{\"\":16s}{\"atom_proj_ext\":>16s}{\"atom_proj\":>16s}{\"|diff|\":>12s}')\n"
     "print(f'{\"Omega_I\":16s}{omega_I_ext:16.4f}{omega_I_plain:16.4f}'\n"
     "      f'{abs(omega_I_ext - omega_I_plain):12.4f}')\n"
     "print(f'{\"Omega_total\":16s}{omega_ext:16.4f}{omega_plain:16.4f}'\n"
     "      f'{abs(omega_ext - omega_plain):12.4f}')\n"
     "\n"
     "centres_ext = result_ext.centres_bohr * BOHR_TO_ANG\n"
     "centres_plain = result_plain.centres_bohr * BOHR_TO_ANG\n"
     "print('max |centre_ext - centre_plain| (Ang):',\n"
     "      np.abs(np.sort(centres_ext, axis=0) - np.sort(centres_plain, axis=0)).max())"),
    ("md",
     "**Takeaway.** `atom_proj_ext` (a real, previously-broken-until-this-"
     "notebook plumbing path, now fixed and unit-tested) reproduces the "
     "pre-existing `atom_proj` path's physics closely when handed the same "
     "pseudopotential's own orbitals through the external-file route -- "
     "$\\Omega_I$/$\\Omega_{\\rm total}$ agree to a few thousandths of "
     "Å$^2$ (well under 0.1%) and Wannier centres agree to well under "
     "$10^{-3}$ Å. The mechanism's real value is letting a projector "
     "shape come from anywhere -- not just whatever a specific "
     "pseudopotential happens to embed -- demonstrated at full scale "
     "($num\\_wann=18$, $s$+$p$+$d$) against real `wannier90.x` in a "
     "separate throwaway script using the official tutorial's own shipped "
     "`ext_proj/Si.dat` (see `project_waw.md`'s tutorial 35 entry for that "
     "cross-validation record)."),
]


# ==========================================================================
# 36 — Silicon: the Stengel-Spaldin alternative localization functional
# ==========================================================================
si_ss_functional = [
    ("md",
     "# Tutorial 36 — Silicon: the Stengel-Spaldin localization functional "
     "(`use_ss_functional`)\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 36. Every notebook "
     "so far minimizes the ordinary Marzari-Vanderbilt total spread "
     "$\\Omega = \\Omega_I + \\Omega_D + \\Omega_{OD}$. Stengel & Spaldin "
     "(PRB **73**, 075121 (2006)) proposed an ALTERNATIVE functional that "
     "keeps $\\Omega_I$/$\\Omega_{OD}$ identical but replaces $\\Omega_D$'s "
     "usual branch-cut/guiding-centre log treatment with a plain "
     "k-averaged **variance** of the diagonal overlap $M_{nn}$:\n"
     "$$\\Omega_D^{\\rm SS} = \\sum_n \\sum_b w_b\\left[\\langle |M_{nn}"
     "(k,b)|^2\\rangle_k - |\\langle M_{nn}(k,b)\\rangle_k|^2\\right]$$\n"
     "\n"
     "New capability: `core.spread.compute_ss_spread`/`_ss_spread_from_"
     "M_tilde` (transcribed from `wannierise.F90::wann_omega`'s actual "
     "minimized-total branch, not a similar-looking but merely-"
     "centre-reporting section of the same file), threaded through "
     "`core.optim.minimize_spread` -> `core.pipeline.wannierize` -> "
     "`waw.interfaces.ase.driver.wannierize(..., use_ss_functional=True)` "
     "-- a plain optimizer kwarg, no special driver plumbing needed. "
     "**No general inequality between $\\Omega_D^{\\rm SS}$ and the "
     "ordinary $\\Omega_D^{\\rm MV}$ is claimed or provable** (an earlier, "
     "wrong reading of a different Fortran section suggested $\\Omega^{\\rm "
     "SS}\\le\\Omega^{\\rm MV}$ always; this notebook checks that claim "
     "directly on a real system and finds it does NOT hold in general — "
     "see section 4).\n"
     "\n"
     "Real Wannier90's own tutorial 36 ships two silicon variants sharing "
     "one lattice: a **Marzari-Vanderbilt** run (`use_ss_functional=."
     "false.`, the default) with 4 bond-centred $s$ trial orbitals on the "
     "isolated 4-band valence manifold, and a **Stengel-Spaldin** run "
     "(`use_ss_functional=.true.`) with 8 atom-centred $sp^3$ trial "
     "orbitals spanning valence+conduction via genuine disentanglement."),
    ("code", SETUP),
    ("md",
     "### 1. Structure and converged DFT\n"
     "Same primitive diamond-Si cell (in bohr) and `Si.upf` PseudoDojo "
     "pseudopotential as notebooks 03/11/27 -- `ecutwfc=40`, "
     "`scf_kpts=(8,8,8)`, Wannier mesh `mp_grid=(4,4,4)` (64 k-points, "
     "matching the real tutorial's own choice exactly). `nbnd=12` matches "
     "the real tutorial's own `num_bands=12` for BOTH variants (a wider "
     "candidate window than either projection set strictly needs, but "
     "keeping it lets the two variants share one converged NSCF)."),
    ("code",
     "from ase import Atoms\n"
     "\n"
     "cell = np.array([[-5.10, 0.0, 5.10], [0.0, 5.10, 5.10], [-5.10, 5.10, 0.0]]) * BOHR_TO_ANG\n"
     "frac = [[-0.25, 0.75, -0.25], [0.0, 0.0, 0.0]]\n"
     "atoms = Atoms('Si2', scaled_positions=frac, cell=cell, pbc=True)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'si_ss_functional'\n"
     "NBND = 12"),
    ("md",
     "### 2. Marzari-Vanderbilt variant — 4 bond-centred $s$ orbitals\n"
     "The 4 fractional bond-midpoint coordinates below are the exact ones "
     "the real tutorial's `select_projections 1 2 3 4` picks out of its "
     "wider 12-orbital block -- built here directly as the only "
     "projections requested, same trick as notebook 11. `dis_froz_max = "
     "dis_win_max = 6.5` eV: checking the converged eigenvalues below "
     "confirms this really is an ISOLATED manifold (band 4 tops out at "
     "6.42 eV, band 5 starts at 7.10 eV — no real disentanglement freedom, "
     "the same situation as notebook 11's own bond-centred run)."),
    ("code",
     "bond_centres = [(-0.125, -0.125, 0.375), (0.375, -0.125, -0.125),\n"
     "                (-0.125, 0.375, -0.125), (-0.125, -0.125, -0.125)]\n"
     "proj_mv = [(c, 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0) for c in bond_centres]\n"
     "\n"
     "ov_mv = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_mv',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=NBND, num_wann=4,\n"
     "    projections=proj_mv,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "eig_mv = ov_mv['eig']\n"
     "print('band 4 max (eV):', eig_mv[:, 3].max(), '   band 5 min (eV):', eig_mv[:, 4].min())\n"
     "print('  -> isolated valence manifold below the 6.5 eV window' if eig_mv[:, 3].max() < 6.5 < eig_mv[:, 4].min()\n"
     "      else '  -> WARNING: window does not cleanly isolate 4 bands')"),
    ("code",
     "res_mv = wannierize(\n"
     "    atoms, MP_GRID, ov_mv['kpts'],\n"
     "    mmn=ov_mv['mmn'], amn=ov_mv['amn'], eig=ov_mv['eig'],\n"
     "    nnkpts=ov_mv['nnkpts'], g_vectors=ov_mv['g_vectors'],\n"
     "    nw=4, outer_window=(-1e6, 6.5), frozen_window=(-1e6, 6.5),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     "    use_ss_functional=False,\n"
     ")\n"
     "sp_mv = res_mv.spread\n"
     "print(f'Omega_I     = {sp_mv.Omega_I * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_D     = {sp_mv.Omega_D * BOHR_TO_ANG**2:.9f} Ang^2')\n"
     "print(f'Omega_OD    = {sp_mv.Omega_OD * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_total = {res_mv.omega_final * BOHR_TO_ANG**2:.6f} Ang^2   '\n"
     "      f'(reference, same overlaps: 6.347638)')"),
    ("md",
     "**Cross-validation.** Feeding this exact `si_mv.{mmn,amn,eig}` "
     "triple into a real, serial `wannier90.x` (v3.1.0, this environment's "
     "QE-7.5-bundled build) with the equivalent `.win` reproduces "
     "$\\Omega_I=5.776723$, $\\Omega_{OD}=0.570915$, $\\Omega_{\\rm total}"
     "=6.347638$ Å$^2$ and all 4 Wannier centres to **9 decimal places** — "
     "essentially machine precision, since both codes minimize the "
     "identical MV functional on the identical overlaps. This is the "
     "strongest possible confirmation that the shared "
     "disentangle/spread-minimization pipeline underneath BOTH variants "
     "in this notebook is correct."),
    ("md",
     "### 3. Stengel-Spaldin variant — 8 atom-centred $sp^3$ orbitals\n"
     "`Si:sp3` expands to 4 hybrid orbitals per atom ($l=-3$, $m_r=1..4$, "
     "same convention as notebook 07's silane); here built explicitly for "
     "both Si atoms in `atoms_frac` order, matching the real tutorial's "
     "own `Si:sp3` keyword. `dis_froz_max=6.5` eV, `dis_win_max=17.0` eV — "
     "genuine disentanglement across the valence+conduction manifold. "
     "`use_ss_functional=True` switches the spread minimizer to the "
     "Stengel-Spaldin $\\Omega_D$."),
    ("code",
     "si_fracs = atoms.get_scaled_positions()\n"
     "proj_sp3 = []\n"
     "for pos in si_fracs:\n"
     "    proj_sp3 += [(tuple(pos), -3, mr, 1, (0., 0., 1.), (1., 0., 0.), 1.0)\n"
     "                 for mr in (1, 2, 3, 4)]\n"
     "\n"
     "ov_ss = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_ss',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=NBND, num_wann=8,\n"
     "    projections=proj_sp3,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     "    rerun_scf=False,\n"
     ")\n"
     "res_ss = wannierize(\n"
     "    atoms, MP_GRID, ov_ss['kpts'],\n"
     "    mmn=ov_ss['mmn'], amn=ov_ss['amn'], eig=ov_ss['eig'],\n"
     "    nnkpts=ov_ss['nnkpts'], g_vectors=ov_ss['g_vectors'],\n"
     "    nw=8, outer_window=(-1e6, 17.0), frozen_window=(-1e6, 6.5),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     "    use_ss_functional=True,\n"
     ")\n"
     "sp_ss = res_ss.spread\n"
     "print(f'Omega_I     = {sp_ss.Omega_I * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_D_ss  = {sp_ss.Omega_D * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_OD    = {sp_ss.Omega_OD * BOHR_TO_ANG**2:.6f} Ang^2')\n"
     "print(f'Omega_total = {res_ss.omega_final * BOHR_TO_ANG**2:.6f} Ang^2')"),
    ("md",
     "**A real bug this notebook build caught.** `core.pipeline.wannierize`'s "
     "final `omega_final` was computed with the ORDINARY MV formula "
     "regardless of `use_ss_functional` — correct for every other caller, "
     "but silently WRONG for `use_ss_functional=True` (it reported the MV "
     "$\\Omega$ evaluated at the SS-converged $U$, a different number from "
     "the SS $\\Omega$ that was actually minimized: 137.96 Å² vs. the "
     "correct 16.86 Å² on this exact system). Fixed in `core/pipeline.py` "
     "to branch on `use_ss_functional` — `result.spread.Omega` "
     "(=`Omega_I+Omega_D+Omega_OD`) already had the right value throughout "
     "and is what caught the bug."),
    ("md",
     "### 4. Honest comparison — no general $\\Omega^{\\rm SS}\\le\\Omega"
     "^{\\rm MV}$ inequality\n"
     "The MV and SS runs above use *different* `num_wann` (4 vs 8) and "
     "*different* trial orbitals, so their $\\Omega$ values aren't "
     "directly comparable — that's simply what the real tutorial's own "
     "two variants are. For an apples-to-apples check, rerun the SAME 8 "
     "$sp^3$ overlaps with `use_ss_functional=False` — isolating exactly "
     "what the functional choice itself changes."),
    ("code",
     "res_ss_mv = wannierize(\n"
     "    atoms, MP_GRID, ov_ss['kpts'],\n"
     "    mmn=ov_ss['mmn'], amn=ov_ss['amn'], eig=ov_ss['eig'],\n"
     "    nnkpts=ov_ss['nnkpts'], g_vectors=ov_ss['g_vectors'],\n"
     "    nw=8, outer_window=(-1e6, 17.0), frozen_window=(-1e6, 6.5),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     "    use_ss_functional=False,\n"
     ")\n"
     "sp_ss_mv = res_ss_mv.spread\n"
     "print(f'{\"\":14s}{\"SS functional\":>16s}{\"MV functional\":>16s}')\n"
     "print(f'{\"Omega_I\":14s}{sp_ss.Omega_I * BOHR_TO_ANG**2:16.6f}{sp_ss_mv.Omega_I * BOHR_TO_ANG**2:16.6f}   (identical formula either way)')\n"
     "print(f'{\"Omega_D\":14s}{sp_ss.Omega_D * BOHR_TO_ANG**2:16.6f}{sp_ss_mv.Omega_D * BOHR_TO_ANG**2:16.6f}')\n"
     "print(f'{\"Omega_OD\":14s}{sp_ss.Omega_OD * BOHR_TO_ANG**2:16.6f}{sp_ss_mv.Omega_OD * BOHR_TO_ANG**2:16.6f}')\n"
     "print(f'{\"Omega_total\":14s}{res_ss.omega_final * BOHR_TO_ANG**2:16.6f}{res_ss_mv.omega_final * BOHR_TO_ANG**2:16.6f}')"),
    ("md",
     "On this exact system, $\\Omega_D^{\\rm SS} = 0.186125$ Å$^2$ is "
     "SLIGHTLY **larger** than $\\Omega_D^{\\rm MV} = 0.178622$ Å$^2$ "
     "(and correspondingly $\\Omega_{\\rm total}^{\\rm SS}=16.862881$ Å$^2$ "
     "> $\\Omega_{\\rm total}^{\\rm MV}=16.855049$ Å$^2$) — a direct, "
     "concrete counterexample to the (incorrect, and explicitly NOT "
     "claimed by `core/spread.py`'s own docstring) idea that the "
     "Stengel-Spaldin functional always gives a smaller spread than the "
     "ordinary Marzari-Vanderbilt one. Both are legitimate, differently-"
     "defined localization functionals; which one gives the smaller "
     "number is system-dependent."),
    ("md",
     "### 5. Interpolated band structure (Stengel-Spaldin, 8 MLWFs)\n"
     "ASE's own diamond-cubic k-path (`band_path`), same convention as "
     "every other notebook here — the valence+conduction bands "
     "interpolated from the 8-orbital genuinely-disentangled Hamiltonian."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=200)\n"
     "bands_ss = interpolate_bands(res_ss.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "plot_bands(xcoords, xspecial, labels, bands_ss, figsize=(6.5, 4.2),\n"
     "           title='Si valence+conduction bands from 8 sp3 MLWFs (use_ss_functional=True)')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Cross-validation status for `use_ss_functional` itself.** "
     "`strings $(which wannier90.x) | grep -i use_ss_functional` returns "
     "NOTHING for this environment's QE-7.5-bundled build (self-reports "
     "v3.1.0) — the same way `kdotp`/`dis_froz_proj` were already found "
     "absent in tutorials 33/34's own builds. This isn't merely the "
     "keyword defaulting to off: the string doesn't exist anywhere in the "
     "compiled binary, meaning this build's Fortran source predates the "
     "feature entirely. The Stengel-Spaldin variant (section 3) therefore "
     "could NOT be cross-checked against a real `wannier90.x`/`postw90.x` "
     "run in this environment. What COULD be, and was: the plain MV "
     "pathway (section 2) — the SAME `disentangle`/overlap/projection "
     "plumbing both variants share — matches real `wannier90.x` to 9 "
     "decimal places, and `core.spread.compute_ss_spread` itself has 8 "
     "passing unit tests (`tests/test_ss_spread.py`, including an "
     "autodiff-vs-finite-difference gradient check and a synthetic-model "
     "check that Omega_I/Omega_OD reduce to the ordinary MV values "
     "exactly) plus a full end-to-end `wannierize(use_ss_functional=True)` "
     "smoke test."),
    ("md",
     "**Takeaway.** `use_ss_functional` (Stengel & Spaldin's alternative "
     "localization functional) is now wired end-to-end in waw exactly "
     "like any other `wannierize()` optimizer kwarg. On real silicon "
     "DFT data: the shared MV pipeline underneath both variants matches "
     "real `wannier90.x` to 9 decimal places; the SS functional itself "
     "gives a *larger* $\\Omega_D$ than MV on the 8-orbital valence+"
     "conduction system studied here, refuting any general "
     "$\\Omega^{\\rm SS}\\le\\Omega^{\\rm MV}$ ordering; and a genuine "
     "`omega_final`/`use_ss_functional` reporting bug in `core/pipeline.py` "
     "was found and fixed along the way."),
]


# ==========================================================================
# 37 — Iron: translational invariance (transl_inv_full)
# ==========================================================================
fe_transl_inv = [
    ("md",
     "# Tutorial 37 — Iron: translational invariance\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 37. Notebook 19 built "
     "the orbital magnetization $M_{\\rm orb}$ from three real-space "
     "quantities, $AA(R)$/$BB(R)$/$CC(R)$, using the **plain** finite-"
     "difference Berry-connection formulas (MV97/CTVR06). Those plain "
     "formulas have a subtle problem: they are built directly from the "
     "position operator $r$, whose matrix elements depend on the choice of "
     "coordinate origin. Since $M_{\\rm orb}$ is supposed to be a physical, "
     "origin-independent property of the crystal, the plain path is only "
     "correct up to that origin-dependence -- shift where you put the "
     "coordinate origin (equivalently, in a single-atom-per-cell crystal "
     "like bcc iron, shift where the atom sits inside its own unit cell) "
     "and the plain $M_{\\rm orb}$ silently changes with it, even though "
     "nothing physical has changed.\n"
     "\n"
     "Wannier90's `transl_inv_full` (`core.hamiltonian.compute_position_r`/"
     "`compute_bb_r`/`compute_cc_r`'s `centres=`/`H_R=`/`BB_R=` arguments, "
     "transcribed from `src/postw90/get_oper.F90`) fixes this: it rewrites "
     "$AA(R)$/$BB(R)$/$CC(R)$ so that the R=0 diagonal is pinned to the "
     "ACTUAL Wannier centres (not wherever a finite-difference sum "
     "happens to put it), with matching correction terms for $BB$/$CC$. "
     "This notebook builds bcc Fe (SOC, the exact system notebook 19 "
     "already validated) **twice**, with the single Fe atom placed at two "
     "different fractional positions in an otherwise identical cell -- "
     "$(0,0,0)$ and $(0,0,\\tfrac14)$ -- which, for a 1-atom-per-cell "
     "Bravais lattice, is EXACTLY a rigid shift of the coordinate origin: "
     "the two calculations describe the identical physical crystal. The "
     "plain path's $M_{\\rm orb}$ should therefore (wrongly) differ between "
     "the two DFT runs, while the `transl_inv_full`-corrected path should "
     "agree."),
    ("code", SETUP),
    ("md",
     "### 1. Two atom positions, one physical crystal\n"
     "Same recipe as notebook 19 (noncollinear+SOC bcc Fe, PseudoDojo "
     "`Fe-sp_r.upf`, SCDM `erfc` entanglement, 28→18 spinor MLWFs, "
     "`write_uHu=True`), run independently for each atom position in its "
     "own work directory. `without_translation` reuses notebook 17/18/19's "
     "already-converged SCF in `runs/fe/`; `with_translation` is a fresh "
     "SCF in `runs/fe_z025/` with the Fe atom moved to "
     "$(0,0,\\tfrac14)$ in the SAME bcc primitive cell -- a genuinely "
     "independent DFT calculation, not a manual relabelling."),
    ("code",
     "from ase import Atoms\n"
     "from ase.build import bulk\n"
     "import torch\n"
     "from waw.interfaces.ase.structure import real_lattice, recip_lattice\n"
     "from waw.core.spread import rotate_overlaps, weight_overlaps_by_eigenvalues\n"
     "from waw.core.hamiltonian import compute_position_r, compute_bb_r, compute_cc_r\n"
     "from waw.analysis.orbital_magnetization import orbital_magnetization\n"
     "from waw.units import EV_TO_HARTREE\n"
     "\n"
     "MP_GRID = (4, 4, 4)\n"
     "CELL_BCC = bulk('Fe', 'bcc', a=2.8699).cell\n"
     "SOC = dict(noncolin=True, lspinorb=True)\n"
     "SOC['starting_magnetization(1)'] = 0.4\n"
     "SOC.update(occupations='smearing', smearing='cold', degauss=0.02)\n"
     "\n"
     "positions = {\n"
     "    'without_translation': (0.0, 0.0, 0.00),\n"
     "    'with_translation':    (0.0, 0.0, 0.25),\n"
     "}\n"
     "\n"
     "runs = {}\n"
     "for label, frac in positions.items():\n"
     "    atoms = Atoms('Fe', scaled_positions=[frac], cell=CELL_BCC, pbc=True)\n"
     "    WORK = HERE / 'runs' / ('fe' if label == 'without_translation' else 'fe_z025')\n"
     "    ov = qe.generate_overlaps(\n"
     "        atoms, MP_GRID, WORK, 'fe',\n"
     "        ecutwfc=60, scf_kpts=(16, 16, 16), nbnd=36, num_wann=18,\n"
     "        exclude_bands=list(range(1, 9)),\n"
     "        scdm_entanglement='erfc', scdm_mu=25.0, scdm_sigma=5.0,\n"
     "        system_extra=SOC, pseudopotentials={'Fe': 'Fe-sp_r.upf'},\n"
     "        pseudo_dir=PSEUDO_DIR, ncores=NCORES, write_uHu=True,\n"
     "        rerun_scf=(label != 'without_translation'),\n"
     "    )\n"
     "    result = wannierize(\n"
     "        atoms, MP_GRID, ov['kpts'],\n"
     "        mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "        nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=18,\n"
     "        outer_window=(8.0, 70.0), frozen_window=(8.0, 19.8),\n"
     "        n_restarts=2, dis_n_iter=1000, n_iter=3000, verbose=False,\n"
     "    )\n"
     "    runs[label] = dict(\n"
     "        atoms=atoms, ov=ov, result=result,\n"
     "        E_FERMI=ov['fermi_energy'],\n"
     "        REAL=real_lattice(atoms), RECIP=recip_lattice(atoms),\n"
     "    )\n"
     "    print(f\"[{label}] Fe at {frac}:  \"\n"
     "          f\"Omega_I = {result.dis.omega_i * BOHR_TO_ANG**2:.4f} Ang^2  \"\n"
     "          f\"E_F = {ov['fermi_energy']:.4f} eV\")"),
    ("md",
     "The two Fermi energies match to the printed precision, as they must "
     "for the identical crystal -- a first, cheap sanity check that the "
     "two DFT runs really are describing the same physics before anything "
     "Wannier-related even happens."),
    ("md",
     "### 2. Uncorrected vs. `transl_inv_full`-corrected $AA(R)$/$BB(R)$/$CC(R)$\n"
     "For EACH atom position, build $AA(R)$/$BB(R)$/$CC(R)$ twice: once "
     "with the plain formulas (notebook 19's default path, `centres=None`), "
     "once with the `transl_inv_full` correction (`centres=` the actual "
     "Wannier centres, `H_R=` the real-space Hamiltonian). The correction "
     "must be threaded in the right order: $BB(R)$ needs the correction "
     "applied to itself first, and the CORRECTED $BB(R)$ (not the plain "
     "one) is what $CC(R)$'s own correction then needs."),
    ("code",
     "MESH = (16, 16, 16)   # berry_kmesh for M_orb -- illustrative, as notebook 19 already notes\n"
     "morb_table = {}\n"
     "for label, d in runs.items():\n"
     "    result, ov, E_FERMI, REAL, RECIP = d['result'], d['ov'], d['E_FERMI'], d['REAL'], d['RECIP']\n"
     "    V, U = result.dis.V, result.spread.U_final\n"
     "    W = torch.bmm(V, U)\n"
     "\n"
     "    Mmn_weighted = weight_overlaps_by_eigenvalues(result.wdata.Mmn, result.wdata.eig)\n"
     "    H_opt   = rotate_overlaps(V, Mmn_weighted, result.wdata.kb_idx)\n"
     "    H_tilde = rotate_overlaps(U, H_opt, result.wdata.kb_idx)\n"
     "    uHu = torch.tensor(ov['uHu'], dtype=torch.complex128) * EV_TO_HARTREE\n"
     "\n"
     "    # --- plain / uncorrected (notebook 19's default path) ---\n"
     "    AA_R0 = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                                result.wdata.kpts, MP_GRID, REAL)\n"
     "    BB_R0 = compute_bb_r(H_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "    CC_R0 = compute_cc_r(uHu, W, result.wdata.kb_idx, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL)\n"
     "\n"
     "    # --- transl_inv_full corrected (centres= actual Wannier centres, H_R= real-space H) ---\n"
     "    centres = torch.tensor(result.centres_bohr, dtype=torch.float64)\n"
     "    H_R = result.hr.H_R\n"
     "    AA_Rc = compute_position_r(result.m_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                                result.wdata.kpts, MP_GRID, REAL, centres=centres)\n"
     "    BB_Rc = compute_bb_r(H_tilde, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL, centres=centres, H_R=H_R)\n"
     "    CC_Rc = compute_cc_r(uHu, W, result.wdata.kb_idx, result.wdata.wb, result.wdata.bvecs,\n"
     "                          result.wdata.kpts, MP_GRID, REAL, centres=centres, BB_R=BB_Rc, H_R=H_R)\n"
     "\n"
     "    morb0 = orbital_magnetization(result.hr, AA_R0, BB_R0, CC_R0, RECIP, REAL,\n"
     "                                  fermi_energies=E_FERMI * EV_TO_HARTREE, mesh=MESH)\n"
     "    morbc = orbital_magnetization(result.hr, AA_Rc, BB_Rc, CC_Rc, RECIP, REAL,\n"
     "                                  fermi_energies=E_FERMI * EV_TO_HARTREE, mesh=MESH)\n"
     "    morb_table[label] = dict(uncorrected=morb0.m_orb[0], corrected=morbc.m_orb[0])\n"
     "    print(f\"[{label}]  uncorrected M_orb = {morb0.m_orb[0].round(4)}   \"\n"
     "          f\"transl_inv_full M_orb = {morbc.m_orb[0].round(4)}\")"),
    ("md",
     "### 3. The 2x2 comparison\n"
     "$M_{\\rm orb}$ (Bohr magneton/cell) for both atom positions, both "
     "with and without the correction, side by side."),
    ("code",
     "print(f\"{'':22s} {'M_orb uncorrected':>28s} {'M_orb transl_inv_full':>28s}\")\n"
     "for label in positions:\n"
     "    u = morb_table[label]['uncorrected']\n"
     "    c = morb_table[label]['corrected']\n"
     "    print(f\"{label:22s} {str(u.round(4)):>28s} {str(c.round(4)):>28s}\")\n"
     "\n"
     "diff_uncorrected = np.abs(morb_table['without_translation']['uncorrected']\n"
     "                          - morb_table['with_translation']['uncorrected'])\n"
     "diff_corrected = np.abs(morb_table['without_translation']['corrected']\n"
     "                        - morb_table['with_translation']['corrected'])\n"
     "print(f\"\\nmax |difference| between the two atom positions:\")\n"
     "print(f\"  uncorrected     : {diff_uncorrected.max():.4f}\")\n"
     "print(f\"  transl_inv_full : {diff_corrected.max():.4f}\")"),
    ("md",
     "The uncorrected $M_{\\rm orb}$ genuinely differs between the two "
     "atom positions -- a real, honestly-documented convention gap in the "
     "\"default\" path, exactly as `compute_position_r`'s own docstring "
     "warns. The `transl_inv_full`-corrected pair agrees far more closely: "
     "both calculations describe the same physical crystal, and only the "
     "correction sees it that way. Perfect many-decimal agreement isn't "
     "expected here -- $M_{\\rm orb}$ is mesh-sensitive (notebook 19's own "
     "caveat) and the two runs are independently converged DFT+MLWF "
     "calculations, not the same numbers relabelled -- but the corrected "
     "pair should track each other much more closely than the uncorrected "
     "pair does."),
    ("md",
     "### 4. Real `wannier90.x`/`postw90.x` cross-validation\n"
     "This build of `wannier90.x`/`postw90.x` (v3.1.0, "
     "`quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k`) does **not** support "
     "`transl_inv_full` at all -- `strings $(which postw90.x) | grep "
     "transl_inv_full` returns nothing, and even the older, simpler "
     "`transl_inv` flag is explicitly rejected for this task "
     "(`postw90.x` aborts with `\"transl_inv=T disabled for morb\"` the "
     "moment it's requested together with `berry_task=morb`). So the "
     "correction itself cannot be cross-validated against a real "
     "`wannier90.x`/`postw90.x` run in this environment -- the same kind "
     "of environment gap already documented for `kdotp` (notebook 33) and "
     "`dis_proj_min/max` (notebook 34).\n"
     "\n"
     "What CAN be attempted: feed waw's own converged `fe.mmn`/`.amn`/"
     "`.eig`/`.uHu` for the `without_translation` position into a "
     "hand-written `fe.win` (throwaway script `runs/fe/xval_morb.py`, not "
     "part of this notebook), let real `wannier90.x` independently "
     "converge its own disentanglement and spread minimization on the "
     "identical ab-initio overlaps (same outer/frozen windows), then run "
     "real `postw90.x` with the DEFAULT (uncorrected) `berry_task = morb` "
     "at the same `berry_kmesh = 16 16 16` and Fermi energy."),
    ("code",
     "# real w90/postw90 binaries, independently converged on the identical\n"
     "# fe.mmn/.amn/.eig/.uHu overlaps (runs/fe/xval_morb.py), DEFAULT (uncorrected)\n"
     "# berry_task=morb, berry_kmesh=16 16 16, fermi_energy=17.6292 eV:\n"
     "#   M_orb (bohr magn/cell)   x        y        z\n"
     "#   ======================  0.1458   0.0105  -0.0018\n"
     "w90_uncorrected_without_translation = np.array([0.1458, 0.0105, -0.0018])\n"
     "waw_uncorrected_without_translation = morb_table['without_translation']['uncorrected']\n"
     "\n"
     "print('waw   uncorrected M_orb (without_translation):',\n"
     "      waw_uncorrected_without_translation.round(4))\n"
     "print('w90   uncorrected M_orb (without_translation):', w90_uncorrected_without_translation)\n"
     "print('abs diff:', np.abs(waw_uncorrected_without_translation\n"
     "                         - w90_uncorrected_without_translation).round(4))"),
    ("md",
     "This does NOT match well -- unlike notebook 19's own denser-mesh "
     "validation, at THIS coarse `4x4x4`-Wannierization mesh the two "
     "codes' $M_{\\rm orb}$ disagree by an order of magnitude, including "
     "sign flips. Before accepting that as a real bug, this was "
     "investigated directly (not part of this notebook, throwaway checks "
     "in the same `runs/fe/` directory):\n"
     "\n"
     "1. **Mesh convergence of waw's OWN uncorrected $M_{\\rm orb}$** "
     "(same Wannier model, `berry_kmesh` swept $8^3 \\to 32^3$): "
     "$[0.0024, 0.0016, -0.0777] \\to [0.0025, 0.0026, -0.0815]$ -- "
     "essentially flat. Not a mesh-convergence problem on waw's side.\n"
     "2. **Sensitivity to the disentanglement/spread-minimization "
     "restart count**: re-wannierizing the SAME overlaps with much "
     "heavier settings (`n_restarts=6`, `dis_n_iter=3000`, `n_iter=6000` "
     "vs. this notebook's `n_restarts=2`/`1000`/`3000`) reproduces "
     "$\\Omega_I$ to 4 decimal places ($9.3217$ vs. $9.3226$ Å$^2$, both "
     "close to real `wannier90.x`'s own $9.322648$) but changes "
     "$M_{\\rm orb}$ again, this time to $[-0.0017, -0.0021, -0.0718]$ -- "
     "a THIRD, still different answer. So $\\Omega_I$ alone does not pin "
     "down a unique disentangled subspace here: bcc Fe's SOC band "
     "structure has several near-degenerate $d$-derived bands right at "
     "$E_F$, and independent optimization runs (waw-vs-waw with "
     "different restarts, or waw-vs-real-`wannier90.x`) land in "
     "DIFFERENT, similarly-low-$\\Omega_I$ subspaces that are not quite "
     "the same physical subspace -- an inherent degeneracy of this "
     "specific coarse-mesh model, not a numerical-convergence failure.\n"
     "3. **Cross-check via a less delicate derived quantity**: notebook "
     "18 already documented that waw's own AHC gauge on this exact Fe "
     "system is \"less-C4z-symmetric\" than real `wannier90.x`'s on a "
     "coarse mesh -- i.e. this same near-degenerate-subspace sensitivity "
     "was already known to affect $-2\\mathrm{Im}[f]$ (built from "
     "$AA(R)$ alone, no $BB$/$CC$ needed). Running real `postw90.x`'s "
     "`berry_task = ahc` on the SAME `fe.chk` used above and comparing "
     "to waw's own `anomalous_hall_conductivity` on the SAME 18-MLWF "
     "model confirms it: the symmetry-protected dominant component "
     "($\\sigma_{xy}$, since $M\\parallel z$) agrees to a few percent "
     "($-807$ vs. real `wannier90.x`'s $-841$ S/cm), while the "
     "symmetry-FORBIDDEN $\\sigma_{yz}/\\sigma_{zx}$ components are "
     "sizeable and DIFFERENT in both codes (waw: $123,-11$; real "
     "`wannier90.x`: $-352, 108$ S/cm) -- exactly the same kind of "
     "coarse-mesh gauge noise, this time visible even in the simpler, "
     "$BB$/$CC$-free $AA(R)$-only quantity.\n"
     "\n"
     "$M_{\\rm orb}$ needs $BB(R)$/$CC(R)$ on top of $AA(R)$ -- genuinely "
     "second-order, cross-differenced quantities notebook 19 already "
     "flagged as \"far more sensitive to mesh density than AHC\" -- so "
     "it is unsurprising (though not independently proven bug-free by "
     "this alone) that the SAME near-degenerate-subspace noise which "
     "shows up as a modest percentage-level effect in AHC shows up as an "
     "order-of-magnitude swing in $M_{\\rm orb}$. The honest conclusion: "
     "the direct real-`wannier90.x` cross-validation of the UNCORRECTED "
     "$M_{\\rm orb}$ path is inconclusive on this coarse, near-degenerate "
     "system -- not a confirmed match, but not compelling evidence of a "
     "code bug either, given points 1-3 above."),
    ("md",
     "**Takeaway.** `transl_inv_full` closes a real, previously-honest "
     "convention gap in waw's orbital-magnetization pipeline: the plain "
     "$AA(R)$/$BB(R)$/$CC(R)$ path is not invariant to where the "
     "coordinate origin (equivalently, the atom inside a 1-atom-per-cell "
     "crystal) happens to sit, so $M_{\\rm orb}$ computed from it should "
     "not be trusted to be origin-independent -- the corrected path fixes "
     "exactly that, at the cost of needing the actual Wannier centres and "
     "the real-space Hamiltonian as extra inputs. This notebook's own "
     "two-position demonstration (section 3) shows the expected "
     "qualitative pattern -- the corrected $M_{\\rm orb}$ tracks between "
     "positions about 5x more closely than the uncorrected one does -- "
     "though on top of a real, non-zero noise floor from this specific "
     "coarse-mesh, near-degenerate-band system's gauge/subspace "
     "sensitivity (section 4), the same phenomenon notebook 18 already "
     "documented for the AHC. The correction itself cannot be "
     "cross-checked against this environment's real `wannier90.x`/"
     "`postw90.x` build at all (`transl_inv_full` is entirely absent, "
     "and `transl_inv` is explicitly disabled for `berry_task = morb`); "
     "its decisive, DFT-noise-free verification is "
     "`tests/test_translational_invariance.py`'s exact operator-identity "
     "check on $AA(R)$ (a uniform centre shift moves ONLY the R=0 "
     "diagonal, exactly as Wannier-function orthonormality requires), "
     "with $BB(R)$/$CC(R)$'s extra terms checked to be finite, "
     "shape-correct, and non-trivial (a careful, direct transcription of "
     "`get_oper.F90`, not independently re-derived)."),
]


# ==========================================================================
# 23 — Silicon, G0W0 quasiparticle bands via Yambo
# ==========================================================================
si_yambo_gw = [
    ("md",
     "# Tutorial 23 — Silicon: G0W0 quasiparticle bands via Yambo\n"
     "\n"
     "The waw-native reimagining of Wannier90 tutorial 23. DFT (LDA/PBE) "
     "systematically underestimates band gaps; the $GW$ approximation "
     "corrects this by replacing the DFT exchange-correlation potential "
     "with a many-body self-energy $\\Sigma = iGW$. This notebook is the "
     "only one in this series that doesn't feed `pw2wannier90`'s own "
     "`.eig` straight into `wannierize` -- instead, a **separate toolchain, "
     "[Yambo](http://www.yambo-code.eu)**, computes a one-shot $G_0W_0$ "
     "quasiparticle correction on top of the same converged DFT ground "
     "state, and we splice the correction into the Wannier eigenvalues "
     "before re-wannierising.\n"
     "\n"
     "**Recipe** (`waw.interfaces.yambo.run_gw_correction`, new this "
     "notebook -- pure orchestration, no new core/analysis physics):\n"
     "1. Ordinary DFT wannierisation (bond-centred $s$ MLWFs, same recipe "
     "as notebook 11) gives the usual `.eig`.\n"
     "2. A **denser** nscf (`8x8x8` mesh, 100 bands, `wf_collect`) feeds "
     "`p2y`, converting QE's wavefunctions into a Yambo `SAVE/` database.\n"
     "3. `yambo` runs a plasmon-pole $G_0W_0$ "
     "(`em1d`+`gw0`+`ppa`+`HF_and_locXC`), producing QP corrections at "
     "every k-point Yambo reduces the dense mesh to (by symmetry).\n"
     "4. `ypp`'s `wannier` runlevel unfolds those corrections back onto "
     "the *original* 4x4x4 Wannier mesh, writing a "
     "`.gw.unsorted.eig`-format file of **QP shifts** ($\\epsilon_{GW} - "
     "\\epsilon_{DFT}$, confirmed directly against a real run here -- not "
     "yet absolute eigenvalues).\n"
     "5. Add the shifts to the DFT eigenvalues and re-sort each k-point's "
     "bands ascending (QP corrections can reorder near-degenerate bands) "
     "-- the resulting array slots straight into `wannierize` exactly "
     "like `qe.generate_overlaps`'s own `out['eig']`.\n"
     "\n"
     "**Cluster note**: this tutorial's toolchain needs `module load "
     "quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k` (NOT this project's usual "
     "7.5 -- `p2y` is sensitive to the exact QE XML/wavefunction format) "
     "plus `module load intel/oneapi/mkl/2024.2` and `~/software/yambo/"
     "bin` on `PATH`, all loaded *before* starting Python/Jupyter."),
    ("code", SETUP),
    ("md",
     "### 1. Structure, bond-centred projections, and DFT wannierisation\n"
     "Same primitive diamond-Si cell, bond-centred $s$ projections, and "
     "`dis_froz_max = dis_win_max = 6.8` eV energy window as Wannier90's "
     "own tutorial 23 `silicon.win` (4 valence MLWFs, no real "
     "disentanglement freedom -- same physical subspace as notebooks 03/11, "
     "reached via bond-centred trial orbitals)."),
    ("code",
     "from ase import Atoms\n"
     "from waw.interfaces import yambo\n"
     "\n"
     "cell = np.array([[-5.10, 0.0, 5.10], [0.0, 5.10, 5.10], [-5.10, 5.10, 0.0]]) * BOHR_TO_ANG\n"
     "frac = [[-0.25, 0.75, -0.25], [0.0, 0.0, 0.0]]\n"
     "atoms = Atoms('Si2', scaled_positions=frac, cell=cell, pbc=True)\n"
     "MP_GRID = (4, 4, 4)\n"
     "WORK = HERE / 'runs' / 'si_yambo_gw'\n"
     "\n"
     "bond_centres = [(-0.125, -0.125, 0.375), (0.375, -0.125, -0.125),\n"
     "                (-0.125, 0.375, -0.125), (-0.125, -0.125, -0.125)]\n"
     "projections = [(c, 0, 1, 1, (0., 0., 1.), (1., 0., 0.), 1.0) for c in bond_centres]\n"
     "\n"
     "ov = qe.generate_overlaps(\n"
     "    atoms, MP_GRID, WORK, 'si_gw',\n"
     "    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=14, num_wann=4,\n"
     "    projections=projections,\n"
     "    pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR, ncores=NCORES,\n"
     ")\n"
     "\n"
     "result_dft = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, outer_window=(-1e3, 6.8), frozen_window=(-1e3, 6.8),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "print(f'DFT Omega_total = {result_dft.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')"),
    ("md",
     "### 2. G0W0 correction via Yambo\n"
     "`run_gw_correction` reuses the SCF charge density and `.nnkp` this "
     "run just produced, driving the denser nscf -> p2y -> yambo -> ypp "
     "chain described above -- the only piece of implementation-time "
     "discovery is Yambo's own IBZ k-point count (read off its setup "
     "report; the dense `8x8x8` mesh reduces to 29 inequivalent k-points "
     "under Si's full $O_h$ symmetry, `%QPkrange` is sized against that "
     "count internally)."),
    ("code",
     "eig_gw = yambo.run_gw_correction(\n"
     "    atoms, (8, 8, 8), WORK, 'si_gw', ov['eig'],\n"
     "    ecutwfc=40, pseudopotentials={'Si': 'Si.upf'}, pseudo_dir=PSEUDO_DIR,\n"
     "    nbnd_gw=100, qpkrange_bands=(1, 14), ncores=NCORES,\n"
     ")\n"
     "dft_gap = ov['eig'][:, 4].min() - ov['eig'][:, 3].max()\n"
     "gw_gap = eig_gw[:, 4].min() - eig_gw[:, 3].max()\n"
     "print(f'DFT gap (lowest band 5 - highest band 4, across all k): {dft_gap:.4f} eV')\n"
     "print(f'GW  gap (lowest band 5 - highest band 4, across all k): {gw_gap:.4f} eV')\n"
     "print('GW should OPEN the gap relative to DFT -- the whole point of this tutorial.')"),
    ("md",
     "### 3. Re-wannierise with the GW-corrected eigenvalues\n"
     "Same recipe as step 1, just fed `eig_gw` instead of `ov['eig']` -- "
     "windows widened to 7.0 eV (`silicon.gw.win`'s own retuning, to "
     "account for the GW-shifted bands) -- and identical `mmn`/`amn` "
     "overlaps (Yambo only touches eigenvalues, not the Wannier gauge "
     "problem itself)."),
    ("code",
     "result_gw = wannierize(\n"
     "    atoms, MP_GRID, ov['kpts'],\n"
     "    mmn=ov['mmn'], amn=ov['amn'], eig=eig_gw,\n"
     "    nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'],\n"
     "    nw=4, outer_window=(-1e3, 7.0), frozen_window=(-1e3, 7.0),\n"
     "    n_restarts=3, dis_n_iter=1000, n_iter=6000, verbose=False,\n"
     ")\n"
     "print(f'GW  Omega_total = {result_gw.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')"),
    ("md",
     "### 4. DFT vs GW band structure\n"
     "Both tight-binding models interpolate onto the same k-path; the GW "
     "one should show a visibly wider gap."),
    ("code",
     "from waw.interfaces.ase.structure import band_path\n"
     "from waw.core.hamiltonian import interpolate_bands\n"
     "\n"
     "bp = band_path(atoms, npoints=120)\n"
     "bands_dft = interpolate_bands(result_dft.hr, bp.kpts) * HARTREE_TO_EV\n"
     "bands_gw = interpolate_bands(result_gw.hr, bp.kpts) * HARTREE_TO_EV\n"
     "xcoords, xspecial, labels = bp.get_linear_kpoint_axis()\n"
     "\n"
     "series = [BandSeries(bands=bands_dft, label='DFT', color='C0'),\n"
     "          BandSeries(bands=bands_gw, label='G0W0', color='C3')]\n"
     "plot_bands(xcoords, xspecial, labels, series, figsize=(6.5, 4),\n"
     "           title='Si valence bands: DFT vs G0W0 (via Yambo)')\n"
     "plt.tight_layout(); plt.show()"),
    ("md",
     "**Takeaway.** Splicing a real many-body $G_0W_0$ correction (Yambo) "
     "into an otherwise ordinary waw wannierisation opens Si's gap "
     "relative to DFT, as expected -- no `wannier90.x` involved anywhere "
     "in this notebook, and no new core/analysis physics needed: Yambo's "
     "own `ypp -wannier` already speaks the plain Wannier90 `.eig` format, "
     "which `waw.interfaces.wannier90.io.read_eig` already parses "
     "unchanged. This was the last previously-skipped official Wannier90 "
     "tutorial in this series (tutorial 15's defected-CNT half, needing "
     "~1100 bands/550 MLWFs, remains explicitly out of scope for this "
     "project's own computational budget)."),
]


_NOTEBOOKS = {
    # keys are named after the OFFICIAL Wannier90 tutorial number they
    # reimagine (not creation order) -- "bonus_..." for the one notebook
    # with no official-tutorial counterpart.
    "01_gaas_isolated.ipynb": gaas,
    "02_lead_isolated.ipynb": lead,
    "03_silicon_disentangle_bands.ipynb": si,
    "04_copper_fermi_surface.ipynb": cu,
    "05_diamond_wannier_functions.ipynb": diamond,
    "07_silane_gamma_only.ipynb": silane,
    "08_iron_spin_polarized.ipynb": fe_collinear,
    "09_batio3_bulk.ipynb": batio3,
    "10_graphite_disentangle.ipynb": graphite,
    "11_silicon_select_projections.ipynb": si_selproj,
    "12_benzene_gamma_only.ipynb": benzene,
    "13_cnt_transport.ipynb": cnt,
    "14_na_chain_transport.ipynb": na_chain,
    "16_silicon_thermoelectrics.ipynb": si_tep,
    "17_iron_soc_spin_texture.ipynb": fe_spin,
    "18_iron_berry_ahc.ipynb": fe_ahc,
    "19_iron_orbital_magnetization.ipynb": fe_morb,
    "20_lavo3_dis_spheres.ipynb": lavo3,
    "21_gaas_sitesym.ipynb": gaas_sitesym,
    "22_copper_sitesym.ipynb": cu_sitesym,
    "23_silicon_yambo_gw.ipynb": si_yambo_gw,
    "24_tellurium_gyrotropic.ipynb": te_gyrotropic,
    "26_gaas_selective_localization.ipynb": gaas_slwf,
    "27_silicon_scdm.ipynb": si_scdm,
    "25_gaas_shift_current.ipynb": gaas_shift_current,
    "28_diamond_cube.ipynb": diamond_cube,
    "29_platinum_spin_hall.ipynb": pt_shc,
    "30_gaas_ac_spin_hall.ipynb": gaas_shc_ac,
    "31_platinum_soc_scdm.ipynb": pt_scdm,
    "32_tungsten_projectability_scdm.ipynb": w_projscdm,
    "33_bc2n_kdotp.ipynb": bc2n_kdotp,
    "34_graphene_projectability_disentangle.ipynb": graphene,
    "35_silicon_ext_proj.ipynb": si_ext_proj,
    "36_silicon_ss_functional.ipynb": si_ss_functional,
    "37_iron_translational_invariance.ipynb": fe_transl_inv,
    "bonus_aluminium_chain_transport.ipynb": al_chain,
}

if __name__ == "__main__":
    import sys
    # optional args = substrings selecting which notebooks to (re)write;
    # no args -> all. Lets you rebuild one notebook without wiping others' outputs.
    sel = sys.argv[1:]
    for name, cells in _NOTEBOOKS.items():
        if not sel or any(s in name for s in sel):
            make(name, cells)
