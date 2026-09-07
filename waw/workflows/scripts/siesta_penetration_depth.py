#!/usr/bin/env python
"""
CIF -> SIESTA (PBE) -> Loewdin H(R) -> quantum geometry -> London penetration
depth, with every convergence knob laddered rather than assumed.

    python siesta_penetration_depth.py STRUCTURE.cif --tc 9.25

WHAT IS COMPUTED, AND WHAT IT IS NOT

  The superfluid weight of Hiorth, Gutierrez-Amigo, Cavignac, Haule, Marques and
  Torma, arXiv:2603.10955, split into its conventional (band-velocity) and
  geometric (quantum-metric) parts, and lambda_L = 1/sqrt(mu_0 D_s).

  This is the clean-limit BAND-THEORY penetration depth. The London formula
  carries the band mass, and the band velocities here come from dH/dk, so the
  measured lambda is LARGER by the electron-phonon mass renormalisation,

      lambda_meas ~= lambda_band * sqrt(1 + lambda_ep),

  plus non-local (Pippard) corrections this local treatment does not contain.
  On Nb the two routes of notebook 22 give lambda_band = 21.3 nm against a
  measured 39-52 nm, and sqrt(1+1.20) closes most of that. Pass --lambda-ep to
  have the renormalised estimate printed alongside; without it, do not compare
  the output to a measurement.

  Assumptions inherited from the formulation: mean-field BCS with a SINGLE
  uniform gap on every orbital, no interband pairing, time-reversal symmetry.
  A two-gap superconductor (MgB2-like) needs one run per gap combined per Fermi
  sheet, which this script does not do -- it will happily print a single number
  that means little in that case.

  The tight-binding velocity omits intra-atomic position matrix elements
  <phi_a|r|phi_b>. For atom-centred NAOs these are not obviously small; on Nb
  the NAO and MLWF routes nevertheless agree to 1.1%, which is evidence but not
  a proof for other chemistries.

WHAT IS CONVERGED, AND HOW IT IS JUDGED

  1. SCF k-grid       -- ladder over k-spacing; watched through E_F and E_KS.
  2. Loewdin H(R) mesh -- the orthogonalised H(R) is longer-ranged than the NAO
                          one. Judged by OFF-MESH interpolation error against
                          SIESTA's own generalised eigenproblem, restricted to a
                          window around E_F (high NAO virtual states are basis-
                          tail artefacts and never interpolate; that is not a
                          failure of the fit). Independent of the SCF grid: the
                          NAO H(R) can be evaluated at any k, so this ladder
                          costs no further SIESTA runs.
  3. mu               -- the HSX file stores H - E_F*S, so the model's Fermi
                          level is identically 0. That is checked, not trusted:
                          mu is re-solved from the valence electron count on the
                          Loewdin model and the offset from 0 is reported.
  4. BZ mesh x sigma  -- D_conv is evaluated in the exact Delta -> 0 limit,
                          where its prefactor becomes 2*delta(eps) and the
                          integral converges like a DOS with a NUMERICAL width
                          sigma decoupled from the physical Delta. A plateau in
                          BOTH mesh and sigma is required; a plateau in one
                          alone is meaningless. D_geom is left exact at Delta.
  5. Delta            -- in the small-gap limit lambda depends on Delta only
                          through D_geom, so its sensitivity is a diagnostic of
                          how geometric the material is. Reported over Delta/2,
                          Delta, 2*Delta.

Every ladder writes a PASS/WARN verdict against an explicit tolerance, and the
JSON output carries the whole table so a series of compounds can be screened and
the doubtful ones revisited.

Requires: a PSML pseudopotential per element in --pseudo-dir, the SIESTA module
loaded (module load siesta/5.4.1-gcc-13.2.0-dfgltdg) and the project venv.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE
while not (REPO / "waw").exists() and REPO != REPO.parent:
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

import waw                                                    # noqa: E402
from waw.analysis.elph import band_eigensystem                 # noqa: E402
from waw.analysis.superfluid import (                          # noqa: E402
    superfluid_weight_small_gap, reduced_to_si, penetration_depth)
from waw.interfaces import siesta as sst                       # noqa: E402
from waw.interfaces.ase.structure import (                     # noqa: E402
    band_path, monkhorst_pack, real_lattice)
from waw.units import EV_TO_HARTREE, HARTREE_TO_EV             # noqa: E402

K_B_EV = 8.617333262e-5
BCS_GAP_OVER_KTC = 1.764


# ----------------------------------------------------------------- structure --
def load_structure(cif, primitive=True, symprec=1e-4):
    """ASE Atoms from a CIF, optionally reduced to the primitive cell.

    The primitive cell is not cosmetic here. D_s carries an explicit 1/V and the
    BZ average runs over the cell's own zone, so a conventional multi-formula
    cell gives the same physics only once the folded bands are fully converged --
    at several times the cost. Reduction is on by default and always reported.
    """
    from ase.io import read
    import spglib

    atoms = read(str(cif))
    n_in = len(atoms)
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    spg = f"{ds.international} (#{ds.number})" if ds is not None else "unknown"
    if primitive:
        prim = spglib.find_primitive(cell, symprec=symprec)
        if prim is not None and len(prim[2]) < n_in:
            from ase import Atoms
            atoms = Atoms(numbers=prim[2], scaled_positions=prim[1],
                          cell=prim[0], pbc=True)
    return atoms, spg, n_in


def check_pseudos(atoms, pseudo_dir):
    """Fail early and legibly on a missing PSML, listing what to fetch."""
    pseudo_dir = pathlib.Path(pseudo_dir)
    want = sorted(set(atoms.get_chemical_symbols()))
    missing = [s for s in want if not (pseudo_dir / f"{s}.psml").exists()]
    if missing:
        raise SystemExit(
            f"missing PSML pseudopotentials in {pseudo_dir}:\n"
            f"  {', '.join(missing)}\n\n"
            f"Fetch the PBE standard set from pseudo-dojo.org (nc-sr-04_pbe_standard,\n"
            f"PSML format) and drop <El>.psml there. Present: "
            f"{', '.join(sorted(p.stem for p in pseudo_dir.glob('*.psml'))) or '(none)'}")
    return want


def kgrid_for_spacing(atoms, spacing_inv_ang):
    """Monkhorst-Pack grid closest to a target k-spacing, per axis.

    A fixed N^3 is wrong for a series of compounds: the same N means a different
    sampling density for a 3 A cell and a 12 A one. This keeps the density fixed.
    """
    b = 2 * np.pi * np.linalg.inv(np.asarray(atoms.cell[:])).T
    n = np.maximum(1, np.ceil(np.linalg.norm(b, axis=1) / spacing_inv_ang))
    return tuple(int(x) for x in n)


# ------------------------------------------------------------------ stage 1 ---
def scf_ladder(atoms, workdir, label, args, elements):
    """SIESTA SCF at a ladder of k-spacings; returns the finest run's H_nao."""
    rows, H_nao, used = [], None, None
    for spacing in args.kspacing:
        kgrid = kgrid_for_spacing(atoms, spacing)
        tag = f"{label}_k{'x'.join(str(n) for n in kgrid)}"
        wd = workdir / tag
        wd.mkdir(parents=True, exist_ok=True)
        fdf = sst.write_fdf(
            wd / f"{tag}.fdf", atoms, label=tag, pseudo_dir=args.pseudo_dir,
            kgrid=kgrid, spin=args.spin, basis=args.basis,
            mesh_cutoff_ry=args.mesh_cutoff, xc="PBE",
            electronic_temperature_ev=args.electronic_temperature,
            extra=({"PAO.EnergyShift": f"{args.energy_shift_ry} Ry"}
                   if args.energy_shift_ry else None))
        t0 = time.time()
        out = sst.run_siesta(fdf, ncores=args.ncores)
        e_f = sst.read_fermi_level(wd, tag)
        e_ks = _read_etot(out)
        rows.append(dict(spacing=spacing, kgrid=list(kgrid), nk=int(np.prod(kgrid)),
                         e_fermi_ev=e_f, e_ks_ev=e_ks, seconds=round(time.time() - t0, 1)))
        print(f"  kspacing {spacing:.3f} 1/A -> {kgrid}  "
              f"E_F = {e_f:+9.4f} eV   E_KS = {e_ks:+14.5f} eV   "
              f"({rows[-1]['seconds']:.0f} s)", flush=True)
        H_nao, used = sst.load_hamiltonian(wd / f"{tag}.fdf"), tag

    verdict = "single point -- not converged, rerun with >= 2 --kspacing values"
    if len(rows) >= 2:
        d_ef = abs(rows[-1]["e_fermi_ev"] - rows[-2]["e_fermi_ev"]) * 1e3
        d_e = abs(rows[-1]["e_ks_ev"] - rows[-2]["e_ks_ev"]) * 1e3
        ok = d_ef < args.tol_scf_mev
        verdict = (f"{'PASS' if ok else 'WARN'}: last two rungs differ by "
                   f"{d_ef:.1f} meV in E_F and {d_e:.1f} meV in E_KS "
                   f"(tolerance {args.tol_scf_mev:.0f} meV on E_F)")
    print(f"  {verdict}")
    return H_nao, used, rows, verdict


def _read_etot(out_path):
    """Final `siesta: Total =` energy in eV (the consistency check is inside
    run_siesta; this is only for the convergence table)."""
    for line in reversed(pathlib.Path(out_path).read_text().splitlines()):
        if line.strip().startswith("siesta:") and "Total =" in line:
            return float(line.split("=")[-1])
    return float("nan")


# ------------------------------------------------------------------ stage 2 ---
def lowdin_ladder(H_nao, atoms, args):
    """Loewdin H(R) at increasing mesh; judged OFF the mesh near E_F."""
    rng = np.random.default_rng(0)
    off = rng.random((args.n_offmesh, 3)) * 0.5 + 0.017     # generic, off any mesh
    e_ref = np.array([H_nao.eigh(k) for k in off])          # eV, E_F = 0
    window = np.abs(e_ref) < args.fidelity_window
    if not window.any():
        raise SystemExit("no states within the fidelity window -- widen "
                         "--fidelity-window")
    rows, models = [], {}
    for n in args.lowdin_mesh:
        mesh = (n, n, n)
        t0 = time.time()
        hr = sst.lowdin_hamiltonian(H_nao, mesh)
        e_w = np.asarray(band_eigensystem(hr, off)[0]) * HARTREE_TO_EV
        err = np.abs(e_w - e_ref)[window]
        rows.append(dict(mesh=n, max_err_mev=float(err.max() * 1e3),
                         rms_err_mev=float(np.sqrt((err ** 2).mean()) * 1e3),
                         n_states=int(window.sum()),
                         seconds=round(time.time() - t0, 1)))
        models[n] = hr
        print(f"  {n:2d}^3: off-mesh error within +-{args.fidelity_window} eV of "
              f"E_F  rms {rows[-1]['rms_err_mev']:7.2f} meV   "
              f"max {rows[-1]['max_err_mev']:8.2f} meV", flush=True)
    best = rows[-1]
    # The verdict is on the RMS, not the max: the max is set by whichever single
    # off-mesh point happens to land worst and moves by tens of meV with the
    # sample, so it is a diagnostic rather than a criterion.
    ok = best["rms_err_mev"] < args.tol_fidelity_mev
    verdict = (f"{'PASS' if ok else 'WARN'}: finest mesh interpolates to "
               f"{best['rms_err_mev']:.1f} meV rms off-mesh "
               f"({best['max_err_mev']:.1f} max; tolerance "
               f"{args.tol_fidelity_mev:.0f} meV on the rms)")
    print(f"  {verdict}")
    return models[args.lowdin_mesh[-1]], rows, verdict


# ------------------------------------------------------------------ stage 3 ---
def check_mu(hr, H_nao, args):
    """Re-solve mu from the electron count instead of trusting mu = 0.

    SIESTA saves H - E_F*S, so the loaded model's Fermi level is 0 by
    construction. It is still worth verifying: the Loewdin model is exact only
    ON its mesh, so a nonzero offset here measures the same interpolation error
    that stage 2 bounds, now projected onto the quantity mu that D_s uses.
    """
    n_elec = float(sum(a.q0.sum() for a in H_nao.geometry.atoms))
    spin_factor = 1.0 if H_nao.spin.is_polarized else 2.0
    k = monkhorst_pack((args.mu_mesh,) * 3)
    eig = np.asarray(band_eigensystem(hr, k)[0])            # Hartree
    kT = max(args.electronic_temperature, 1e-3) * EV_TO_HARTREE

    def count(mu):
        return spin_factor * np.mean(
            np.sum(1.0 / (1.0 + np.exp((eig - mu) / kT)), axis=1))

    lo, hi = eig.min() - 1.0, eig.max() + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count(mid) < n_elec:
            lo = mid
        else:
            hi = mid
    mu = 0.5 * (lo + hi)
    off_mev = mu * HARTREE_TO_EV * 1e3
    ok = abs(off_mev) < args.tol_mu_mev
    verdict = (f"{'PASS' if ok else 'WARN'}: model mu from {n_elec:.0f} valence "
               f"electrons is {off_mev:+.1f} meV from the HSX zero "
               f"(tolerance {args.tol_mu_mev:.0f} meV)")
    print(f"  n_elec = {n_elec:.0f}, spin factor {spin_factor:.0f}, "
          f"{args.mu_mesh}^3 mesh")
    print(f"  {verdict}")
    return (mu if args.use_model_mu else 0.0), off_mev, verdict


# ------------------------------------------------------------------ stage 4 ---
def lam_nm(sw):
    """Diagonal lambda_L in nm from a SuperfluidWeight."""
    return penetration_depth(np.diag(reduced_to_si(sw.total))) * 1e9


def ds_ladder(hr, real, delta_ha, mu, args):
    """2-D ladder in (BZ mesh, numerical sigma). A plateau in both is required."""
    table, cache = [], {}
    hdr = f'{"mesh":>7s}' + "".join(f'{f"sig={s:g}":>13s}' for s in args.sigma)
    print("  lambda_L (nm, isotropic mean of the diagonal)")
    print("  " + hdr)
    for n in args.ds_mesh:
        row = []
        for sig in args.sigma:
            t0 = time.time()
            sw = superfluid_weight_small_gap(
                hr, real, delta=delta_ha, mu=mu,
                sigma=sig * EV_TO_HARTREE, mesh=(n, n, n))
            cache[(n, sig)] = sw
            row.append(float(lam_nm(sw).mean()))
            table.append(dict(mesh=n, sigma_ev=sig, lambda_nm=row[-1],
                              lambda_xyz_nm=[float(v) for v in lam_nm(sw)],
                              geom_share_pct=float(
                                  100 * np.trace(sw.geometric) / np.trace(sw.total)),
                              seconds=round(time.time() - t0, 1)))
        print(f"  {n:5d}^3" + "".join(f"{v:13.2f}" for v in row), flush=True)

    fine = args.ds_mesh[-1]
    lam = {s: [t["lambda_nm"] for t in table if t["sigma_ev"] == s] for s in args.sigma}
    at_fine = [t["lambda_nm"] for t in table if t["mesh"] == fine]
    sigma_spread = (max(at_fine) - min(at_fine)) / np.mean(at_fine) * 100
    if len(args.ds_mesh) >= 2:
        mesh_drift = max(abs(lam[s][-1] - lam[s][-2]) / lam[s][-1] * 100
                         for s in args.sigma)
        ok = mesh_drift < args.tol_plateau_pct and sigma_spread < args.tol_plateau_pct
        verdict = (f"{'PASS' if ok else 'WARN'}: at the finest mesh lambda varies "
                   f"{sigma_spread:.2f}% across sigma and {mesh_drift:.2f}% over "
                   f"the last mesh step (tolerance {args.tol_plateau_pct:.1f}% "
                   f"in both)")
    else:
        mesh_drift = None
        verdict = (f"WARN: lambda varies {sigma_spread:.2f}% across sigma, but a "
                   f"single --ds-mesh value cannot show a mesh plateau -- give "
                   f"at least two")
    print(f"  {verdict}")
    sw_best = cache[(fine, args.sigma[len(args.sigma) // 2])]
    return sw_best, table, verdict


# ------------------------------------------------------------------ stage 5 ---
def delta_sensitivity(hr, real, delta_ha, mu, args):
    """lambda vs Delta. In the small-gap limit only D_geom sees Delta, so a flat
    response means the material is conventional and a steep one means the
    geometric term matters -- which is the interesting case for this formalism."""
    n = args.ds_mesh[-1]
    sig = args.sigma[len(args.sigma) // 2]
    rows = []
    for f in (0.5, 1.0, 2.0):
        sw = superfluid_weight_small_gap(
            hr, real, delta=delta_ha * f, mu=mu,
            sigma=sig * EV_TO_HARTREE, mesh=(n, n, n))
        rows.append(dict(delta_factor=f,
                         delta_mev=delta_ha * f * HARTREE_TO_EV * 1e3,
                         lambda_nm=float(lam_nm(sw).mean()),
                         geom_share_pct=float(
                             100 * np.trace(sw.geometric) / np.trace(sw.total))))
        print(f"  Delta x {f:<4g} = {rows[-1]['delta_mev']:7.3f} meV  ->  "
              f"lambda {rows[-1]['lambda_nm']:8.2f} nm   "
              f"geometric share {rows[-1]['geom_share_pct']:8.4f}%")
    span = (max(r["lambda_nm"] for r in rows) - min(r["lambda_nm"] for r in rows))
    print(f"  lambda spans {span:.2f} nm over a factor 4 in Delta "
          f"({100*span/rows[1]['lambda_nm']:.2f}% of the central value)")
    return rows


# -------------------------------------------------------------------- report --
def bands_figure(H_nao, hr, atoms, path_png, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bp = band_path(atoms, npoints=200)
    x, xt, xl = bp.get_linear_kpoint_axis()
    e_nao = np.array([H_nao.eigh(k) for k in bp.kpts])
    e_low = np.asarray(band_eigensystem(hr, bp.kpts)[0]) * HARTREE_TO_EV
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(x, e_nao, ".", ms=2.5, color="0.45",
            label="SIESTA (generalised $HS$)")
    ax.plot(x, e_low, "-", lw=0.9, color="C3", label=r"Löwdin $H(R)$")
    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.axhspan(-window, window, color="C0", alpha=0.07,
               label=f"fidelity window $\\pm${window} eV")
    ax.set_xticks(xt)
    ax.set_xticklabels([l.replace("G", r"$\Gamma$") for l in xl])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-max(10.0, 2 * window), max(10.0, 2 * window))
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title(f"{atoms.get_chemical_formula()}: SIESTA vs interpolated $H(R)$",
                 fontsize=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:1] + h[-2:], l[:1] + l[-2:], fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cif", nargs="*", help="input CIF file(s); omit if --manifest")
    p.add_argument("--manifest", type=pathlib.Path, default=None,
                   help="CSV or JSON giving per-compound inputs, for a series "
                        "where one --tc would be wrong. Columns/keys: cif "
                        "(required) and any of tc, delta_mev, lambda_ep, spin. "
                        "Anything absent falls back to the command line.")
    p.add_argument("--outdir", default="penetration_depth_runs", type=pathlib.Path)
    p.add_argument("--pseudo-dir", type=pathlib.Path,
                   default=REPO / "workflows" / "pseudos" / "psml")
    # the gap
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tc", type=float,
                   help="Tc in K; Delta = 1.764 kB Tc (weak-coupling BCS)")
    g.add_argument("--delta-mev", type=float, help="gap in meV, measured or computed")
    p.add_argument("--lambda-ep", type=float, default=None,
                   help="electron-phonon coupling, to also report the "
                        "mass-renormalised lambda_band*sqrt(1+lambda_ep)")
    # SIESTA
    p.add_argument("--basis", default="DZP", choices=["SZ", "DZ", "SZP", "DZP", "TZP"])
    p.add_argument("--mesh-cutoff", type=float, default=400.0, help="Ry")
    p.add_argument("--spin", default="non-polarized",
                   choices=["non-polarized", "polarized", "spin-orbit"])
    p.add_argument("--electronic-temperature", type=float, default=0.025, help="eV")
    p.add_argument("--energy-shift-ry", type=float, default=0.01,
                   help="PAO.EnergyShift in Ry, written explicitly rather than "
                        "inherited (0 to omit the key). This is a BASIS-QUALITY "
                        "knob and it is not laddered: on Nb, 0.01 -> 0.02 Ry "
                        "moves lambda by 1.8%%, comparable to the convergence "
                        "tolerances, so treat --basis and this together as the "
                        "systematic they are. 0.01 is SIESTA's own default; ASE "
                        "writes 0.02")
    p.add_argument("--ncores", type=int, default=4,
                   help="MPI ranks; >=8 can silently corrupt this SIESTA build")
    p.add_argument("--kspacing", type=float, nargs="+", default=[0.25, 0.18],
                   help="SCF k-spacing ladder in 1/Angstrom, coarse to fine")
    p.add_argument("--conventional-cell", action="store_true",
                   help="skip the primitive reduction (slower, same physics)")
    # ladders
    p.add_argument("--lowdin-mesh", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--ds-mesh", type=int, nargs="+", default=[30, 40, 60])
    p.add_argument("--sigma", type=float, nargs="+", default=[0.02, 0.05, 0.10],
                   help="numerical Fermi-surface smearing in eV (NOT the gap)")
    p.add_argument("--mu-mesh", type=int, default=24)
    p.add_argument("--n-offmesh", type=int, default=12)
    p.add_argument("--fidelity-window", type=float, default=3.0,
                   help="eV around E_F over which interpolation is judged")
    p.add_argument("--use-model-mu", action="store_true",
                   help="use the electron-count mu instead of the HSX zero")
    # tolerances
    # Defaults calibrated on the bcc-Nb reference of notebook 22, which this
    # script reproduces to 0.1% (21.32 nm against 21.30): there, 16^3 Loewdin
    # gives 34 meV off-mesh and the 60^3 lambda plateau spans 0.9% over both
    # sigma and the last mesh step. Tolerances a shade looser than that, so the
    # reference passes and anything materially worse does not.
    p.add_argument("--tol-scf-mev", type=float, default=20.0)
    p.add_argument("--tol-fidelity-mev", type=float, default=50.0)
    p.add_argument("--tol-mu-mev", type=float, default=50.0)
    p.add_argument("--tol-plateau-pct", type=float, default=1.5)
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args(argv)

    waw.set_num_threads(args.threads)
    args.outdir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, p)
    results = []
    for cif, job in jobs:
        try:
            results.append(run_one(pathlib.Path(cif), job))
        except SystemExit as e:
            print(f"\n!! {cif}: {e}\n", flush=True)
            results.append(dict(cif=str(cif), status="failed", error=str(e)))
        except Exception as e:                                  # noqa: BLE001
            print(f"\n!! {cif}: {type(e).__name__}: {e}\n", flush=True)
            results.append(dict(cif=str(cif), status="failed",
                                error=f"{type(e).__name__}: {e}"))
    summary = args.outdir / "summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary}")
    if len(results) > 1:
        print(f'\n{"compound":<18s}{"lambda_band(nm)":>16s}{"geom %":>9s}{"verdict":>10s}')
        for r in results:
            if r.get("status") == "ok":
                print(f'{r["formula"]:<18s}{r["lambda_band_nm"]:16.2f}'
                      f'{r["geometric_share_pct"]:9.3f}'
                      f'{"PASS" if r["all_pass"] else "WARN":>10s}')
            else:
                print(f'{pathlib.Path(r["cif"]).stem:<18s}{"failed":>16s}')
    return 0


def build_jobs(args, parser):
    """[(cif, per-compound args)] from the command line and/or a manifest.

    A series will not share one Tc, so the manifest exists to carry the gap (and
    anything else that varies) per entry, while the numerical settings stay on
    the command line where they belong.
    """
    import copy

    rows = []
    if args.manifest:
        text = args.manifest.read_text()
        if args.manifest.suffix.lower() == ".json":
            rows = json.loads(text)
        else:
            import csv
            rows = list(csv.DictReader(text.splitlines()))
        if not isinstance(rows, list) or any("cif" not in r for r in rows):
            parser.error(f"{args.manifest}: needs a list of records each with a "
                         f"'cif' field")
    rows += [{"cif": c} for c in args.cif]
    if not rows:
        parser.error("give at least one CIF, or a --manifest")

    jobs = []
    for r in rows:
        a = copy.deepcopy(args)
        for key in ("tc", "delta_mev", "lambda_ep"):
            v = r.get(key, None)
            if v not in (None, ""):
                setattr(a, key, float(v))
        if r.get("spin"):
            a.spin = str(r["spin"])
        if a.tc is None and a.delta_mev is None:
            parser.error(f"{r['cif']}: no gap -- give --tc/--delta-mev or a "
                         f"tc/delta_mev field in the manifest")
        if r.get("delta_mev") not in (None, ""):
            a.tc = None                       # an explicit gap wins over Tc
        jobs.append((r["cif"], a))
    return jobs


def run_one(cif, args):
    t0 = time.time()
    atoms, spg, n_in = load_structure(cif, primitive=not args.conventional_cell)
    formula = atoms.get_chemical_formula()
    label = "".join(c for c in cif.stem if c.isalnum() or c == "_")[:40] or "cmpd"
    workdir = args.outdir / label
    workdir.mkdir(parents=True, exist_ok=True)
    elements = check_pseudos(atoms, args.pseudo_dir)

    delta_mev = (args.delta_mev if args.delta_mev is not None
                 else BCS_GAP_OVER_KTC * K_B_EV * args.tc * 1e3)
    delta_ha = delta_mev * 1e-3 * EV_TO_HARTREE

    print("=" * 78)
    print(f"{cif.name}  ->  {formula}   {spg}")
    print(f"  {n_in} atoms in the CIF, {len(atoms)} after "
          f"{'primitive reduction' if not args.conventional_cell else 'no reduction'}"
          f";  V = {atoms.get_volume():.2f} A^3")
    print(f"  PBE / {args.basis} / mesh {args.mesh_cutoff:.0f} Ry / {args.spin}")
    print(f"  Delta = {delta_mev:.3f} meV"
          + (f" (BCS from Tc = {args.tc} K)" if args.delta_mev is None else " (given)"))
    print("=" * 78)

    print("\n[1/5] SIESTA SCF, k-spacing ladder")
    H_nao, tag, scf_rows, scf_verdict = scf_ladder(atoms, workdir, label, args, elements)
    print(f"  {H_nao.no} NAOs, spin {H_nao.spin}")

    print("\n[2/5] Loewdin H(R) mesh, judged off-mesh near E_F")
    hr, low_rows, low_verdict = lowdin_ladder(H_nao, atoms, args)

    print("\n[3/5] chemical potential")
    mu, mu_off_mev, mu_verdict = check_mu(hr, H_nao, args)

    real = real_lattice(atoms)
    print("\n[4/5] superfluid weight: BZ mesh x numerical sigma")
    sw, ds_rows, ds_verdict = ds_ladder(hr, real, delta_ha, mu, args)

    print("\n[5/5] sensitivity to Delta")
    d_rows = delta_sensitivity(hr, real, delta_ha, mu, args)

    png = workdir / f"{label}_bands.png"
    bands_figure(H_nao, hr, atoms, png, args.fidelity_window)

    lam_xyz = lam_nm(sw)
    lam_iso = float(lam_xyz.mean())
    geom_pct = float(100 * np.trace(sw.geometric) / np.trace(sw.total))
    verdicts = [scf_verdict, low_verdict, mu_verdict, ds_verdict]
    all_pass = all(v.startswith("PASS") for v in verdicts)

    print("\n" + "-" * 78)
    print(f"RESULT  {formula}")
    print(f"  D_s (reduced a.u.), diagonal: "
          f"{np.array2string(np.diag(sw.total), precision=6)}")
    print(f"  geometric share of the trace: {geom_pct:.4f}%")
    print(f"  lambda_band per axis (nm):    {np.round(lam_xyz, 2)}")
    print(f"  lambda_band isotropic (nm):   {lam_iso:.2f}")
    if args.lambda_ep is not None:
        lam_meas = lam_iso * np.sqrt(1.0 + args.lambda_ep)
        print(f"  x sqrt(1+lambda_ep={args.lambda_ep}):        {lam_meas:.2f} nm "
              f"<- compare THIS to experiment, not the line above")
    else:
        print("  (band theory only: multiply by sqrt(1+lambda_ep) before comparing")
        print("   with a measurement -- pass --lambda-ep to have that done here)")
    print(f"  convergence: {'ALL PASS' if all_pass else 'SEE WARNINGS'}")
    for v in verdicts:
        print(f"    - {v}")
    print(f"  bands figure: {png}")
    print(f"  {time.time() - t0:.0f} s total")
    print("-" * 78)

    return dict(
        status="ok", cif=str(cif), formula=formula, spacegroup=spg,
        n_atoms=len(atoms), volume_ang3=float(atoms.get_volume()),
        n_orbitals=int(H_nao.no), basis=args.basis, xc="PBE",
        delta_mev=delta_mev, mu_hartree=float(mu), mu_offset_mev=float(mu_off_mev),
        lambda_band_nm=lam_iso,
        lambda_band_xyz_nm=[float(v) for v in lam_xyz],
        lambda_ep=args.lambda_ep,
        lambda_renormalised_nm=(None if args.lambda_ep is None
                                else lam_iso * float(np.sqrt(1 + args.lambda_ep))),
        geometric_share_pct=geom_pct,
        D_s_diag_reduced=[float(v) for v in np.diag(sw.total)],
        D_conv_diag_reduced=[float(v) for v in np.diag(sw.conventional)],
        D_geom_diag_reduced=[float(v) for v in np.diag(sw.geometric)],
        convergence=dict(scf=scf_rows, lowdin=low_rows, ds=ds_rows,
                         delta_sensitivity=d_rows),
        verdicts=verdicts, all_pass=bool(all_pass),
        seconds=round(time.time() - t0, 1),
    )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    raise SystemExit(main())
