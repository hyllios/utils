# waw — Wannier Analysis Workstation

A [PyTorch](https://pytorch.org)-based Wannier-function engine plus an analysis
layer. It reads the same overlap files as `wannier90.x` (`.eig`, `.mmn`,
`.amn`, `.nnkp`), disentangles and minimises the Marzari–Vanderbilt spread
itself, writes the same `_hr.dat`/`_centres.xyz`, and then does the
post-processing — bands, DOS, Fermi surfaces, Berry curvature and anomalous
Hall, spin Hall, electron–phonon, phonons, surface Green's functions, exchange
couplings, Floquet, DMFT, … — in the same Python session, on plain arrays.

## Read this first

This project is **just for fun**. It is shamelessly vibe-programmed: most of
the code here was written by a large language model under supervision, then
checked, wherever a reference existed at all, against real
`wannier90.x`/`postw90.x`/Quantum ESPRESSO output.

It does **not** intend to replace wannier90. wannier90 is the reference
implementation — fast, community-validated, and what you should use for
production work. The point of `waw` is to be *simple and easy to
change/extend*: one Python package, no Fortran build, every intermediate
quantity a NumPy/torch array you can print and plot, so that trying a new
localisation functional, a new interpolation scheme, or a new response
function is an afternoon's work instead of a fork.

Expect rough edges. There is no API-stability promise, and the physics is only
as validated as the notebook that exercises it says it is (each one states
what was compared against what, and where it disagrees).

## Install

```bash
python -m venv venv && . venv/bin/activate
pip install -e .            # numpy, torch, ase, scipy, spglib, threadpoolctl
pip install -e '.[vis,dev]' # matplotlib/plotly/scikit-image, pytest
pip install -e '.[siesta]'  # sisl, for the SIESTA route
```

Python ≥ 3.10. Everything runs on CPU; `waw.set_num_threads(n)` sets the
thread count.

## Quick start

From wannier90-style inputs already on disk (the `seedname.win`/`.eig`/`.mmn`/
`.amn` set `wannier90.x` would have read):

```python
import waw
from waw.units import BOHR_TO_ANG, HARTREE_TO_EV

res = waw.wannierize("path/to/silicon", nw=8)      # disentangle + minimise
print(res.omega_final * BOHR_TO_ANG**2, "Ang^2")   # core/ is in atomic units

bands = waw.interpolate_bands(res.hr, kpts) * HARTREE_TO_EV
```

Or start from a structure and let the Quantum ESPRESSO driver produce the
overlaps (this is what every notebook in `workflows/` does — no `wannier90.x`
binary anywhere in the chain, only QE's `pw2wannier90.x`):

```python
from ase.build import bulk
from waw.interfaces import quantum_espresso as qe
from waw.interfaces.ase.driver import wannierize

atoms, mp_grid = bulk("Si", "diamond", a=5.43), (6, 6, 6)
ov = qe.generate_overlaps(
    atoms, mp_grid, workdir="runs/si", seedname="si",
    ecutwfc=40, scf_kpts=(8, 8, 8), nbnd=12, num_wann=8,
    pseudopotentials={"Si": "Si.upf"}, pseudo_dir="workflows/pseudos",
)
res = wannierize(atoms, mp_grid, ov["kpts"], mmn=ov["mmn"], amn=ov["amn"],
                 eig=ov["eig"], nnkpts=ov["nnkpts"], g_vectors=ov["g_vectors"],
                 nw=8)
```

## What is in here

| | |
|---|---|
| `waw/core/` | disentanglement, spread functionals (Marzari–Vanderbilt, Stengel–Spaldin, Pipek–Mezey), six gauge optimisers (SGD, Adam, CG, L-BFGS, DIIS, Riemannian trust region), site symmetry, Wigner–Seitz interpolation, frozen/outer windows, SCDM |
| `waw/analysis/` | bands, DOS, Fermi surfaces, effective masses, k·p, Berry curvature and anomalous Hall/Nernst, orbital magnetisation, gyrotropic and shift current, spin Hall and spin accumulation, Z₂/Wilson loops, Boltzmann and ballistic transport, surface spectral functions (decimation), phonons, electron–phonon (α²F, λ, Fan–Migdal Σ), Eliashberg and SCDFT, Heisenberg exchange and magnons, Floquet, CPA, DMFT, band unfolding |
| `waw/interfaces/` | Quantum ESPRESSO (`pw.x`, `pw2wannier90.x`, `ph.x`, DFPT), wannier90 file formats, ASE, VASP (VASP2WANNIER90), SIESTA (NAO `H(R)`/`S(R)`, no Wannierisation), PySCF, Yambo (GW), EPW |
| `waw/vis/` | band-structure, Fermi-surface and Wannier-function plotting (matplotlib/plotly) |
| `waw/utils/` | Eliashberg and SCDFT solvers, run-directory manifests |
| `workflows/w90tutorial/` | 35 notebooks reproducing the official wannier90 tutorials, numbered after them, plus a bonus one; waw-native, so no `wannier90.x` binary |
| `workflows/notebooks/` | 29 notebooks going past the tutorials — real materials, real references, and honest statements of where the agreement stops |
| `tests/` | 1233 tests, `pytest -n 16` |

Units: `waw/core/` and `waw/analysis/` are **pure atomic units**;
`waw/units.py` (`to_si_units`, `to_eVA_units`) is the only conversion
boundary.

`CLAUDE.md` is the instruction file for the model that wrote most of this;
it is kept in the repository rather than tidied away.

The notebooks are the real documentation — `workflows/notebooks/INDEX.md` and
the two `README.md` files there say what each one computes and what it was
validated against.

## Licence

GPL-3.0-or-later; see [`LICENSE`](LICENSE). Note that this applies to *this*
subdirectory — the top-level `LICENSE` of the `hyllios/utils` repository (MIT)
does not cover `waw/`.
