# waw-native Wannier tutorials

Pedagogical, end-to-end notebooks that reproduce the spirit of the Wannier90
tutorials **the waw way** — with **no `wannier90.x` and no `.win`/`.chk`**. Each
notebook goes from a crystal structure all the way to a physical result:

    ase.Atoms
      → converged Quantum ESPRESSO (pw.x), PseudoDojo pseudos
      → waw-native .nnkp  (waw.core.generate_nnkp; replaces `wannier90.x -pp`)
      → overlaps          (pw2wannier90.x SCDM; the code-agnostic .mmn/.amn/.eig)
      → waw wannierisation (waw.interfaces.ase.driver.wannierize)
      → waw analysis       (bands / spin texture / Berry curvature / AHC)

The only wannier90-ecosystem binary used is QE's `pw2wannier90.x`, which writes
the overlap files; the wannier90 *minimiser* and its file interface are gone.

## The notebooks

Filenames are numbered after the **official Wannier90 tutorial** they
reimagine (not creation order); `bonus_...` has no official-tutorial number.

| Notebook | W90 tutorial | System | What it shows |
|----------|:---:|--------|---------------|
| `01_gaas_isolated` | 1 | GaAs | 4 isolated $sp^3$ MLWFs (excluding the PseudoDojo Ga/As 3d semicore), band interpolation |
| `02_lead_isolated` | 2 | Pb | 4 isolated $sp^3$-like MLWFs on an FCC metal (excluding the PseudoDojo Pb 5d semicore) |
| `03_silicon_disentangle_bands` | 3 | Si | disentanglement (12→8 $sp^3$), $\Omega_I$ vs Wannier90, band structure |
| `04_copper_fermi_surface` | 4/6 | Cu | metal disentanglement (5 d + 2 s MLWFs), band structure + 2-D slice and full 3-D Fermi surface with the Brillouin zone |
| `05_diamond_wannier_functions` | 5 | diamond | real-space Wannier functions from UNK files (bond-centred sp3 orbitals) |
| `07_silane_gamma_only` | 7 | SiH$_4$ | $\Gamma$-only molecular Wannierisation (analytic $sp^3$ projections, half b-vector shell), 4 bond MLWFs |
| `08_iron_spin_polarized` | 8 | bcc Fe (collinear) | spin-polarized (no SOC): 2 independent 9-MLWF ($s;p;d$) Wannierisations, exchange-split bands |
| `09_batio3_bulk` | 9 | BaTiO$_3$ | multi-species perovskite, 9 isolated O 2$p$ MLWFs via `exclude_bands` only (no disentanglement window) |
| `10_graphite_disentangle` | 10 | graphite | non-orthogonal hexagonal lattice, 20→10 disentanglement ($\sigma$+$\pi$), band structure $\Gamma$-M-K-$\Gamma$-A |
| `11_silicon_select_projections` | 11 | Si | 4 bond-centred $s$ MLWFs (`select_projections` reproduced by specifying only the desired projection subset) |
| `12_benzene_gamma_only` | 12 | C$_6$H$_6$ | bigger $\Gamma$-only molecule, 15 isolated MLWFs from a hand-built (not W90's literal `random`) trial-orbital guess |
| `13_cnt_transport` | 13 | (5,5) CNT | 80→50 disentanglement ($\pi$+$\sigma$), bulk ballistic (Landauer) transport: metallic 2-channel transmission |
| `14_na_chain_transport` | 14 | Na chain | both halves: periodic (3-MLWF, `transport_bulk`, 1-channel metal) and defected 13-atom chain (13-MLWF, `transport_lcr`, transmission dip at the defect) |
| `16_silicon_thermoelectrics` | 16 | Si | semiclassical Boltzmann transport (BoltzWann): Seebeck & conductivity vs doping/T |
| `17_iron_soc_spin_texture` | 17 | bcc Fe (SOC) | noncollinear+SOC, 18 spinor MLWFs, spin-coloured bands (`.spn`) |
| `18_iron_berry_ahc` | 18 | bcc Fe (SOC) | Berry curvature + anomalous Hall conductivity (full WYSV06), multi-core dense mesh |
| `19_iron_orbital_magnetization` | 19 | bcc Fe (SOC) | orbital magnetization (postw90 `berry_task=morb`), `.uHu` overlaps, CTVR06/LVTS12 trace formulas |
| `20_lavo3_dis_spheres` | 20 | LaVO3, SrMnO3 | k-space-localized disentanglement (`dis_spheres`): LaVO3 $t_{2g}$ triplet (with a direct with/without-`dis_spheres` comparison) + all three official SrMnO3 examples ($t_{2g}$, $e_g$, full $d$) |
| `21_gaas_sitesym` | 21 | GaAs | symmetry-adapted Wannier functions (`site_symmetry`, irreducible-k-wedge optimization): all 5 official examples (`atom_centered_Ga_s/_p/_sp`, `atom_centered_As_sp`, `bond_centered`) |
| `22_copper_sitesym` | 22 | Cu | symmetry-adapted Wannier functions with genuine disentanglement (10→6 MLWFs, 5 $d$ + 1 $s$) on a metal: all 3 official $s$-centre variants (`s_at_0.00`, `s_at_0.50`: full $O_h$; `s_at_0.25`: reduced $T_d$ via a new `sym_ops`/`read_sym` explicit-symmetry-list mechanism) |
| `23_silicon_yambo_gw` | 23 | Si | $G_0W_0$ quasiparticle band correction via a real [Yambo](http://www.yambo-code.eu) toolchain (new `interfaces.yambo`: p2y/yambo/ppa-GW/ypp orchestration, no new core/analysis physics); denser `8x8x8` nscf → p2y → yambo (`em1d`+`gw0`+`ppa`+`HF_and_locXC`) → `ypp -wannier` unfolds QP corrections back onto the 4x4x4 Wannier mesh, spliced into the DFT `.eig` and re-wannierised — opens Si's DFT gap (0.67 eV) toward the GW value (1.31 eV) |
| `24_tellurium_gyrotropic` | 24 | trigonal Te | postw90 `gyrotropic` module (kinetic magnetoelectric + natural optical activity): $D$/$K_{\rm orb}$/$C$/DOS Fermi-surface tensors, frequency-dependent $\tilde D(\omega)$, and NOA $\gamma^{\rm orb}_{abc}(\omega)$; new Hamiltonian-gauge position matrix (`core.hamiltonian.hamiltonian_gauge_position`, WYSV06 Eq. 24/25) |
| `25_gaas_shift_current` | 25 | GaAs | postw90 `berry_task=sc` nonlinear shift-photocurrent tensor (IATS18), incl. `kubo_adpt_smr` adaptive smearing (YWVS07); validated against wannier90's own test-suite reference (1-12% match, 18 components) AND against real wannier90.x/postw90.x on this exact system (3-6% on the 2 symmetry-allowed components, correctly noise-floor on the 15 symmetry-forbidden ones) |
| `26_gaas_selective_localization` | 26 | GaAs | selectively localized Wannier functions (SLWF/SLWF+C, `slwf_num`/`slwf_constrain`): one bond-centred $s$ orbital singled out and localized alone (spectators delocalize), then its centre pinned to the As site via a Lagrange-multiplier penalty |
| `27_silicon_scdm` | 27 | Si | SCDM entanglement functions (`isolated`/`gaussian`/`erfc`): 4 valence-only, 4 conduction-focused, and 8 valence+conduction MLWFs from the same density-matrix-column trial orbitals every other notebook already uses implicitly |
| `28_diamond_cube` | 28 | diamond | Gaussian `.cube` export of real-space MLWFs (new `interfaces.wannier90.realspace.write_cube`, VESTA-compatible), reusing notebook 05's system and grid |
| `29_platinum_spin_hall` | 29 | Pt | postw90 `berry_task=eval_shc` spin Hall conductivity (Qiao method, QZYZ18); new `.spn`-derived $SS(R)/SR(R)/SHR(R)/SH(R)$ real-space operators; "AHC with one velocity leg replaced by a spin-current operator" |
| `30_gaas_ac_spin_hall` | 30 | GaAs | frequency-dependent (ac) spin Hall conductivity, `shc_freq_scan`; complex resonance-pole accumulation (`spin_hall_conductivity_ac`) instead of tutorial 29's real Lorentzian; new `core.hamiltonian.apply_scissors_shift` (rigid conduction-band gap correction, wavefunction-invariant) |
| `31_platinum_soc_scdm` | 31 | Pt | `auto_projections` (SCDM) extended to a spinor/SOC system; same Pt system as tutorial 29 but Wannierised via SCDM instead of analytic projections — no new capability, `scdm_entanglement` already supported spinors (validated non-spinor in tutorial 27) |
| `32_tungsten_projectability_scdm` | 32 | bcc W | SCDM `erfc` entanglement with `scdm_mu`/`scdm_sigma` *fitted* to a real `projwfc.x` atomic-projectability curve (new `interfaces.quantum_espresso.projwfc`), then `dis_num_iter=0` — the fitted subspace needs no further iterative disentanglement |
| `33_bc2n_kdotp` | 33 | BC2N | k.p expansion coefficients around a single k-point via quasi-degenerate (Löwdin) perturbation theory (new `analysis.kdotp.kdotp_coefficients`, postw90 `berry_task=kdotp`); wannierization cross-checked against real `wannier90.x` to 6 decimals, the kdotp coefficients themselves validated by synthetic-model unit tests since the installed `wannier90.x`/`postw90.x` build (v3.1.0) predates the `kdotp` feature |
| `34_graphene_projectability_disentangle` | 34 | graphene | projectability-based disentanglement (`dis_proj_min/max`) fixes the free-standing 2-D vacuum-runaway spread problem |
| `35_silicon_ext_proj` | 35 | Si | external atomic projectors (`atom_proj_ext`, new `interfaces.quantum_espresso.upf`): a pseudopotential's own radial orbitals round-tripped through pw2wannier90's external-projector file path, cross-checked against the pre-existing built-in `atom_proj` path |
| `36_silicon_ss_functional` | 36 | Si | Stengel-Spaldin alternative localization functional (`use_ss_functional`, new `core.spread.compute_ss_spread`): 4 bond-centred-$s$ Marzari-Vanderbilt run vs. 8 atom-centred-$sp^3$ Stengel-Spaldin run; MV path matches real `wannier90.x` to 9 decimals, `use_ss_functional` itself predates this environment's v3.1.0 build so only unit-tested, and a genuine `core/pipeline.py` `omega_final` reporting bug (ignored `use_ss_functional` entirely) was found and fixed |
| `37_iron_translational_invariance` | 37 | bcc Fe (SOC) | translational-invariance correction (`transl_inv_full`, new `centres=`/`H_R=`/`BB_R=` arguments on `core.hamiltonian.compute_position_r`/`compute_bb_r`/`compute_cc_r`) for the orbital magnetization: the SAME Fe atom placed at two different fractional positions within an otherwise identical bcc cell (a rigid origin shift) shows the plain $AA(R)$/$BB(R)$/$CC(R)$ path's $M_{\rm orb}$ genuinely depends on that choice, while the corrected path tracks much more closely between the two; `transl_inv_full`/`transl_inv` are both entirely unsupported for `berry_task=morb` by this environment's real `wannier90.x`/`postw90.x` build (v3.1.0), so the correction itself is validated by `tests/test_translational_invariance.py`'s exact operator-identity check instead, with an honest discussion of this system's real, pre-existing near-degenerate-subspace gauge sensitivity (already documented for the AHC in notebook 18) limiting the real-DFT cross-validation of the uncorrected path |
| `bonus_aluminium_chain_transport` | — | Al chain | ballistic (Landauer) quantum transport: quantized conductance staircase (same capability as official tutorials 13/14) |

**Coverage notes.** Tutorial 6 (Copper via `pw2wannier90`) has no separate
notebook: every notebook here already generates its own overlaps via
`pw2wannier90` (tutorial 6's whole point, upstream), so it's folded into
`04_copper_fermi_surface`'s "4/6" label rather than duplicated. Tutorial 9's
*surface*-termination follow-up (BaTiO$_3$(001)) lives in the separate
`workflows/notebooks/` series, not here. Tutorial 15 is open future work (see
the parent repo's tracked tasks), needing ~1100 bands/550 MLWFs for its
"defected" half -- ~10x this project's largest system so far.

## Running

```bash
# QE toolchain (this cluster) -- 7.3.1, not 7.5 (which has a bug)
module load quantum-espresso/7.3.1-gcc-13.2.0-6jwmo4k
# notebook 23 (Yambo G0W0) additionally needs:
#   module load intel/oneapi/mkl/2024.2
#   export PATH=~/software/yambo/bin:$PATH
# one-time: register the waw Jupyter kernel + deps
uv pip install nbformat nbconvert ipykernel jupyterlab matplotlib scikit-image plotly
# scikit-image + plotly are used only by notebook 04's interactive 3-D Fermi surface
# (marching cubes + Brillouin-zone isosurface rendering, waw.vis.fermi_surface)
python -m ipykernel install --user --name waw

# regenerate the .ipynb from source and run them
python _generate.py
python -m nbconvert --to notebook --execute --inplace *.ipynb
```

The DFT runs land in a git-ignored `runs/<system>/` scratch dir; re-running a
notebook reuses a completed SCF. The pseudopotentials are committed one level
up, in `../pseudos/` (shared across `workflows/`, not w90tutorial-specific;
see `../PSEUDOS.md`).

## Notes on parallelism (important)

* **QE runs pure-MPI** (`mpirun -np N pw.x`), with `OMP_NUM_THREADS=1` forced per
  rank inside `waw.interfaces.quantum_espresso`. **waw uses CPU threads**
  (`waw.set_num_threads()`). Keeping these separate matters: if the QE child
  inherited a high `OMP_NUM_THREADS`, each of the `N` MPI ranks would spawn `N`
  OpenMP threads (`N²` total) and the SCF slows by ~1-2 orders of magnitude.
* The AHC in notebook 18 evaluates the Berry curvature on a dense mesh; waw's
  batched, multi-threaded `torch.linalg.eigh` makes a 25³ mesh (×9 Fermi
  energies) a ~7 s calculation instead of minutes.

## Files

* `waw.interfaces.quantum_espresso` (imported as `qe` in every notebook) —
  direct-input QE driver (`generate_overlaps`: scf → native `.nnkp` → nscf →
  pw2wannier90/SCDM), living in the main `waw/` package (not
  tutorial-specific glue) since it's a real, general interface alongside
  `waw.interfaces.wannier90`/`.ase`. Davidson NSCF with a band buffer;
  `exclude_bands` for semicore; spinor `.nnkp` auto-selected for noncollinear
  runs; `gamma_only` + `projections` for isolated-molecule $\Gamma$-only runs
  (notebook 07); `spin_component` for collinear spin-polarized runs
  (notebook 08).
* `_generate.py` — builds the notebooks with nbformat.
* `../pseudos/` + `../PSEUDOS.md` — the committed PseudoDojo pseudopotentials
  (shared with any other `workflows/` notebook that needs DFT, not just these).

## These are tutorials, not tests

Bit-level cross-validation of waw against real `wannier90.x`/`postw90.x` is
done during development (see `project_waw.md`'s per-tutorial memory entries
for the exact numbers) but is not preserved as a permanent fixture/test --
these notebooks are the pedagogical, waw-native, self-contained record.
