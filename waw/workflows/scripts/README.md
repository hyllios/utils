# `siesta_penetration_depth.py`

CIF in, London penetration depth out, with every numerical knob laddered.

```bash
module load siesta/5.4.1-gcc-13.2.0-dfgltdg
source ~/software/venv/wannier/bin/activate

python siesta_penetration_depth.py MgB2.cif --tc 39.0
python siesta_penetration_depth.py *.cif --tc 9.25 --outdir screen   # a series
```

The pipeline is notebook 22's Route A, made batchable: SIESTA (PBE) → NAO
`H`/`S` → Löwdin orthogonalisation → `H(R)` → quantum geometry → superfluid
weight → `λ_L = 1/√(μ₀ D_s)` (Hiorth *et al.*, arXiv:2603.10955).

## The gap is a required input

One of `--tc` (uses weak-coupling BCS, `Δ = 1.764 k_B T_c`) or `--delta-mev`
(measured or computed). It matters less than it looks: in the small-gap limit
`D_conv` is Δ-independent, so λ sees Δ only through `D_geom`. Stage 5 quantifies
that — on Nb, λ moves 5% over a factor 4 in Δ.

## A series needs a manifest, not one `--tc`

Compounds do not share a Tc, so `--manifest` carries what varies per entry while
the numerical settings stay on the command line:

```csv
cif,tc,lambda_ep,spin
Nb.cif,9.25,1.20,non-polarized
V3Si.cif,17.0,1.0,non-polarized
Pd.cif,,,polarized
```

or the same as a JSON list of objects. Recognised fields: `cif` (required),
`tc`, `delta_mev`, `lambda_ep`, `spin`. Blank or absent falls back to the command
line; an explicit `delta_mev` overrides `tc` for that row. Command-line CIFs are
appended to the manifest's, so the two can be mixed.

## What comes out is the *band-theory* λ

The London formula carries the band mass and this calculation supplies band
velocities, so the measured value is larger by the electron–phonon mass
renormalisation:

    λ_meas ≈ λ_band · √(1 + λ_ep)

Pass `--lambda-ep` to have that printed. **Without it, do not compare the output
to a measurement.** Nb: λ_band = 21.3 nm, ×√2.2 = 31.6 nm, measured 39–52 nm —
the remainder plausibly non-local (Pippard) corrections, which a local London
treatment does not contain.

## The five ladders, and what each catches

| stage | knob | criterion | default tol |
|---|---|---|---|
| 1 | SCF k-spacing | `E_F` change between last two rungs | 20 meV |
| 2 | Löwdin `H(R)` mesh | **off-mesh** interpolation error near `E_F` | 50 meV |
| 3 | μ | electron-count μ vs the HSX zero | 50 meV |
| 4 | BZ mesh × σ | plateau in **both** | 1.5% |
| 5 | Δ | λ over Δ/2 … 2Δ | reported |

Each prints `PASS`/`WARN` and the whole table lands in `summary.json`, so a
screen can be triaged rather than trusted wholesale.

Three of these are easy to get wrong in ways that look fine:

- **Stage 2 must be judged off the mesh.** The Löwdin `H(R)` is exact *on* its
  own mesh at any size, so an on-mesh check is vacuous. It is also energy
  resolved: high NAO virtual states are basis-tail artefacts and never
  interpolate in any NAO code, which is why the window is `±--fidelity-window`
  around `E_F` and not the whole spectrum.
- **σ is numerical, not physical.** It is the width representing
  `Δ²/(ε²+Δ²)^{3/2} → 2δ(ε)`, decoupled from the real gap. A plateau in mesh at
  one σ proves nothing; the plateau has to hold across σ too.
- **μ = 0 is a convention, not a measurement.** SIESTA saves `H − E_F·S`, so the
  model's Fermi level is identically zero. Stage 3 re-solves μ from the valence
  electron count anyway: the offset it finds *is* the stage-2 interpolation error
  projected onto the one number `D_s` most cares about. Use `--use-model-mu` to
  adopt it instead of the zero.

## Validated against notebook 22, and the defaults are calibrated there

On bcc Nb at notebook 22's settings the script reproduces it exactly: λ_L =
21.30/21.30/21.49 nm at 60³ for σ = 0.02/0.05/0.10, geometric share 0.0103% vs
0.010%, `E_KS` = −1694.16409 eV matching to the last digit.

Defaults `--kspacing 0.25 0.18 --lowdin-mesh 8 12 16 --ds-mesh 30 40 60 --sigma
0.02 0.05 0.10`. **They are not universal.** A larger cell needs different
`--ds-mesh`/`--lowdin-mesh` for the same k-density — the k-spacing logic covers
the SCF grid, but those two are literal counts; a flat-band or heavy-fermion
system may need far denser sampling, since the conventional prefactor is a shell
of width ~Δ.

### The basis is a systematic, and it is *not* laddered

`--basis` and `--energy-shift-ry` change the answer and no ladder here will tell
you so. Measured on Nb: `PAO.EnergyShift` 0.01 → 0.02 Ry moves λ_L by **1.8%**
(21.30 → 20.92 nm), which is the same size as the convergence tolerances. The
default is 0.01 Ry, written explicitly — SIESTA's own default, and what notebook
22 inherited. ASE writes 0.02. If a screen's numbers are to be compared with each
other, fix both knobs across the series and say which values were used; the JSON
records them.

## Pseudopotentials

PSML, one `<El>.psml` per element in `--pseudo-dir` (default
`workflows/pseudos/psml`, which currently holds only Fe, I, Nb, Ni, Te). The
script fails before running anything and lists what is missing. Get the PBE
standard set from pseudo-dojo.org (`nc-sr-04_pbe_standard`, PSML format).

## Assumptions that a screen will silently violate

- **Single uniform gap**, no interband pairing, time-reversal symmetry. A
  two-gap superconductor (MgB₂) needs one call per gap combined per Fermi
  sheet; the script will print one confident number instead. Check whether the
  compound is known multi-gap before believing the output.
- **`--spin non-polarized` by default.** Correct for most superconductors, wrong
  for a magnetic entry in the database.
- **Tight-binding velocity.** `∂H/∂k` misses intra-atomic
  `⟨φ_a|r|φ_b⟩`; for atom-centred NAOs these are not obviously small. On Nb the
  NAO and MLWF routes agree to 1.1%, which is evidence, not a proof for other
  chemistries.
- **`--ncores` defaults to 4.** This SIESTA build can silently return a wrong
  Hamiltonian at ≥8 MPI ranks; `run_siesta` raises on the consistency check that
  catches it, but 4 avoids the question.
