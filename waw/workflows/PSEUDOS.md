# Pseudopotentials for the waw workflow notebooks

Shared across `workflows/` (not specific to any one notebook set): any
notebook that runs DFT reads its pseudopotentials from here, so notebooks
stay reproducible without an external download. Most are from the
**PseudoDojo** ONCVPSP set (D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013);
pseudo-dojo.org), generated 08/16/2017, version 3.3.0, PBE functional. The
three fully-relativistic PAW files (`Pt-rel.upf`, `Ga-rel.upf`, `As-rel.upf`)
and `Fe.jry.pbe.UPF` come instead from Quantum ESPRESSO's own **PSlibrary**
(A. Dal Corso, *Comput. Mater. Sci.* **95**, 337 (2014)), as their headers
record. Both sets are redistributed here under their own upstream terms.

| File          | Element | Relativity                   | Valence                    | Used by (`workflows/w90tutorial/`)              |
|---------------|---------|-------------------------------|-----------------------------|--------------------------------------------------|
| `Ga.upf`      | Ga      | scalar-relativistic            | 3d + 4s4p (13 e⁻)           | `01_gaas_isolated`                                |
| `As.upf`      | As      | scalar-relativistic            | 4s4p (5 e⁻)                 | `01_gaas_isolated`                                |
| `Si.upf`      | Si      | scalar-relativistic            | 3s3p (4 e⁻)                 | `03_silicon_disentangle_bands`, `11_silicon_select_projections`, `16_silicon_thermoelectrics` |
| `Fe-sp_r.upf` | Fe      | **fully-relativistic** (SOC)    | 3s3p3d4s (16 e⁻, semicore)  | `17_iron_soc_spin_texture`, `18_iron_berry_ahc`   |
| `Fe.upf`      | Fe      | scalar-relativistic            | 3s3p3d4s (16 e⁻, semicore)  | `08_iron_spin_polarized` (collinear, no SOC needed) |
| `Cu.upf`      | Cu      | scalar-relativistic            | 3s3p3d4s (19 e⁻, semicore)  | `04_copper_fermi_surface`                         |
| `C.upf`       | C       | scalar-relativistic            | 2s2p (4 e⁻)                 | `05_diamond_wannier_functions`, `10_graphite_disentangle`, `12_benzene_gamma_only`, `13_cnt_transport` |
| `Al.upf`      | Al      | scalar-relativistic            | 3s3p (3 e⁻)                 | `bonus_aluminium_chain_transport`                 |
| `H.upf`       | H       | scalar-relativistic            | 1s (1 e⁻)                   | `07_silane_gamma_only`, `12_benzene_gamma_only`   |
| `Ba.upf`      | Ba      | scalar-relativistic            | 5s5p6s (10 e⁻, semicore)    | `09_batio3_bulk`                                  |
| `Ti.upf`      | Ti      | scalar-relativistic            | 3s3p3d4s (12 e⁻, semicore)  | `09_batio3_bulk`                                  |
| `O.upf`       | O       | scalar-relativistic            | 2s2p (6 e⁻)                 | `09_batio3_bulk`                                  |
| `Na.upf`      | Na      | scalar-relativistic            | 2s2p3s (9 e⁻, semicore)     | `14_na_chain_transport`                           |
| `Pb.upf`      | Pb      | scalar-relativistic            | 5d6s6p (14 e⁻, semicore)    | `02_lead_isolated`                                |
| `La.upf`      | La      | scalar-relativistic            | 5s5p5d6s (11 e⁻, semicore)  | `20_lavo3_dis_spheres`                            |
| `V.upf`       | V       | scalar-relativistic            | 3s3p3d4s (13 e⁻, semicore)  | `20_lavo3_dis_spheres`                            |
| `Te.upf`      | Te      | scalar-relativistic            | 4d5s5p (16 e⁻, semicore)    | `24_tellurium_gyrotropic`                         |
| `Pt-rel.upf`  | Pt      | **fully-relativistic** (SOC), PAW | 5d6s (16 e⁻)             | `29_platinum_spin_hall`                           |
| `Ga-rel.upf`  | Ga      | **fully-relativistic** (SOC), PAW | 3d4s4p (13 e⁻)           | `30_gaas_ac_spin_hall`                            |
| `As-rel.upf`  | As      | **fully-relativistic** (SOC), PAW | 4s4p (5 e⁻)              | `30_gaas_ac_spin_hall`                            |
| `W.upf`       | W       | scalar-relativistic            | 5s5p5d6s (14 e⁻, semicore)  | `32_tungsten_projectability_scdm`                 |
| `B.upf`       | B       | scalar-relativistic            | 2s2p (3 e⁻)                 | `33_bc2n_kdotp`                                   |
| `N.upf`       | N       | scalar-relativistic            | 2s2p (5 e⁻)                 | `33_bc2n_kdotp`                                   |
| `Pd-rel.upf`  | Pd      | **fully-relativistic** (SOC)    | 4s4p4d (18 e⁻, semicore)    | (superseded PdBi2 attempt, not currently used — kept for now in case that direction is revisited) |
| `Bi-rel.upf`  | Bi      | **fully-relativistic** (SOC)    | 5d6s6p (15 e⁻, semicore)    | (superseded PdBi2 attempt); `notebooks/03_bi2se3_topological_surface` |
| `Ru-rel.upf`  | Ru      | **fully-relativistic** (SOC)    | 4s4p4d (16 e⁻, semicore)    | `notebooks/07_fe3run_anomalous_nernst`            |
| `Se-rel.upf`  | Se      | **fully-relativistic** (SOC)    | 3d4s4p (16 e⁻, semicore)    | `notebooks/03_bi2se3_topological_surface`           |
| `Te-rel.upf`  | Te      | **fully-relativistic** (SOC)    | 4d5s5p (16 e⁻, semicore)    | `notebooks/spin_accumulation_tellurium` (new — SAC needs SOC, unlike `24_tellurium_gyrotropic`'s scalar-relativistic `Te.upf`) |
| `Co-rel.upf`  | Co      | **fully-relativistic** (SOC)    | 3s3p3d4s (17 e⁻, semicore)  | `notebooks/cosi_spin_hall_nernst` (new — CoSi chiral-fermion semimetal, PhysRevB.106.165102) |
| `Si-rel.upf`  | Si      | **fully-relativistic** (SOC)    | 3s3p (4 e⁻)                 | `notebooks/cosi_spin_hall_nernst` (noncolin+lspinorb requires ALL pseudopotentials in the run to be FR, even a light element like Si with negligible intrinsic SOC — cannot mix `Si.upf` (SR) with an FR Co pseudo in the same noncollinear calculation) |
| `Mg.upf`      | Mg      | scalar-relativistic            | 2p3s (10 e⁻, semicore)      | `notebooks/mgb2_phonon_fatbands` (new — first phonon capability, no SOC needed) |
| `Ca.upf`      | Ca      | scalar-relativistic            | 3s3p4s (10 e⁻, semicore)    | `notebooks/ca_elph_gw` (new — Ca's 3d manifold sits just above E_F and is misplaced by PBE, which is the target of the GW-corrected transport work; l_max = 2 so the pseudo carries the d channel that matters) |

* The scalar-relativistic pseudos (Ga, As, Si, Cu, C, Al) are the standard
  PseudoDojo `ONCVPSP-PBE-SR-PDv0.4` set.
* `Fe-sp_r.upf` is the fully-relativistic (spin-orbit) PseudoDojo
  `ONCVPSP-PBE-FR-PDv0.4` Fe pseudo — needed for the noncollinear + SOC Fe
  notebooks (a scalar-relativistic pseudo cannot carry spin-orbit). It is the
  same pseudo used by wannier90 tutorials 17/18's own reference data.
* `Pt-rel.upf`/`Ga-rel.upf`/`As-rel.upf` are the ONLY pseudos here NOT from
  PseudoDojo: they're QE pslibrary's fully-relativistic PAW pseudos
  (`Pt.rel-pbe-n-kjpaw_psl.0.1.UPF`, `Ga.rel-pbe-dn-kjpaw_psl.0.2.UPF`,
  `As.rel-pbe-n-kjpaw_psl.0.2.UPF`), used unmodified as bundled with wannier90
  tutorials 29/30's own reference data (PseudoDojo does not currently publish
  fully-relativistic entries for these elements). Note `Ga-rel.upf`/
  `As-rel.upf` are DIFFERENT pseudos from the already-committed scalar-
  relativistic `Ga.upf`/`As.upf` (tutorials 1/21/25/26) — a scalar-
  relativistic pseudo cannot carry spin-orbit coupling.
* `Pd-rel.upf`/`Bi-rel.upf`/`Se-rel.upf`/`Te-rel.upf`/`Co-rel.upf`/`Si-rel.upf`
  ARE from PseudoDojo (`ONCVPSP-PBE-FR-PDv0.4`, `nc-fr-04_pbe_standard` table,
  fetched 2026-07-15/16 from
  pseudo-dojo.org/pseudos/nc-fr-04_pbe_standard/{Pd,Bi,Se,Te,Co,Si}.upf.gz) —
  unlike Pt/Ga/As above, PseudoDojo does publish FR entries for these elements.

## Plane-wave cutoffs

Each notebook sets `ecutwfc` at or above the PseudoDojo "normal/high" hint for
its elements and is run to convergence (`conv_thr = 1e-10`):

| Notebook                        | `ecutwfc` (Ry) | Note                                       |
|----------------------------------|----------------|---------------------------------------------|
| `01_gaas_isolated`                | 70             | Ga 3d semicore needs a high cutoff          |
| `02_lead_isolated`                 | 50             | Pb 5d semicore                              |
| `03_silicon_disentangle_bands`    | 40             | well above Si normal (~36 Ry)               |
| `04_copper_fermi_surface`         | 55             | Cu 3s3p3d semicore                          |
| `05_diamond_wannier_functions`    | 50             | C, no semicore                              |
| `07_silane_gamma_only`             | 40             | matches the classic w90 tutorial's cutoff   |
| `08_iron_spin_polarized`           | 60             | Fe SR, same cutoff as `17_iron_soc_spin_texture` |
| `09_batio3_bulk`                   | 60             | Ba 5s5p + Ti 3s3p3d semicore                |
| `10_graphite_disentangle`         | 50             | C, same cutoff as `05_diamond_wannier_functions` |
| `11_silicon_select_projections`   | 40             | same Si system as `03_silicon_disentangle_bands` |
| `12_benzene_gamma_only`           | 30             | matches the classic w90 tutorial's cutoff   |
| `13_cnt_transport`                | 30             | matches the classic w90 tutorial's cutoff   |
| `14_na_chain_transport`           | 40             | Na 2s2p semicore                            |
| `16_silicon_thermoelectrics`      | 40             | same Si system as `03_silicon_disentangle_bands` |
| `17_iron_soc_spin_texture`        | 60             | Fe FR, above the ~45 Ry normal hint          |
| `18_iron_berry_ahc`               | 60             | same Fe system as `17_iron_soc_spin_texture` |
| `19_iron_orbital_magnetization`   | 60             | same Fe system as `17_iron_soc_spin_texture` |
| `20_lavo3_dis_spheres`            | 80             | La 5s5p + V 3s3p3d semicore                 |
| `bonus_aluminium_chain_transport` | 45             | Al, no semicore                             |
| `32_tungsten_projectability_scdm` | 82             | PseudoDojo "high" hint (41 Ha = 82 Ry) for W's 5s5p5d6s semicore; confirmed by direct convergence scan (total energy converged to 0.1 mRy between 70 and 80 Ry) |

## Provenance / license

* `Nb.upf` and `psml/Nb.psml` are the SAME PseudoDojo scalar-relativistic
  entry in two formats -- `nc-sr-04_pbe_standard`, generated 10/31/2017 --
  downloaded from
  pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard/Nb.{upf,psml}.gz. Both carry 13
  valence electrons (4s4p4d5s semicore), so the Quantum ESPRESSO and SIESTA
  routes to the Nb superfluid weight start from the same pseudopotential rather
  than from two different ones -- the point of that comparison being the basis
  (plane waves plus Wannierization versus numerical atomic orbitals), not the
  pseudopotential. A semicore set of this size wants ecutwfc well above the
  usual: run a convergence scan rather than assuming a value.

PseudoDojo pseudopotentials are distributed under a Creative-Commons license;
see pseudo-dojo.org. Redistributed here unmodified for reproducibility.
* `Ca.upf` is PseudoDojo `nc-sr-04_pbe_standard`, generated 10/31/2017 — the same
  table and vintage as `Nb.upf`, fetched from
  pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard/Ca.upf.gz. 10 valence electrons
  (3s3p semicore + 4s), `l_max = 2`, 6 projectors, with a nonlinear core
  correction. The d channel is not optional here: fcc Ca's empty 3d manifold
  sits a little above E_F and its position is what PBE gets wrong.

To refresh or extend the set, download from pseudo-dojo.org (choose the
`ONCVPSP-PBE-SR-PDv0.4` and `-FR-` tables) and drop the `.upf` files here.
