"""Al el-ph at Miguel's converged settings: 20^3 electron k-mesh (also the
DFPT k-mesh), 10^3 coarse q via the certified dvscf star rotation (47
irreducible ph.x runs -> 1000 mesh points), fine interpolation to 40^3."""
import sys, time, pathlib
import numpy as np, torch

REPO = '/aims_data/miguel/nas-data001/claude/wannier'
sys.path.insert(0, REPO)
import waw
waw.set_num_threads(40)

from ase import Atoms
from waw.interfaces import quantum_espresso as qe
from waw.interfaces.ase.driver import wannierize
from waw.interfaces.projections import spd_projections
from waw.interfaces.ase.structure import (monkhorst_pack, irreducible_qpoints,
                                          crystal_symmetry_operations,
                                          real_lattice as rl, recip_lattice as rc)
from waw.interfaces.quantum_espresso.io import (write_pw_input, write_ph_input,
                                                write_ph_input_explicit_q, run_pw,
                                                write_q2r_input)
from waw.interfaces.quantum_espresso.phonon_io import read_force_constants
from waw.interfaces.quantum_espresso import dvscf_io
from waw.interfaces.quantum_espresso.upf import read_norm_conserving
from waw.interfaces.wannier90.io import read_unk
from waw.analysis import elph
from waw.analysis.phonon import apply_acoustic_sum_rule, interpolate_phonons
from waw.core.hamiltonian import HamiltonianR
from waw.units import (BOHR_TO_ANG, HARTREE_TO_EV, EV_TO_HARTREE,
                       CM1_TO_HARTREE, K_B_HARTREE)

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

W = pathlib.Path(REPO) / 'workflows/notebooks/runs/al_elph_dense'
W.mkdir(parents=True, exist_ok=True)
CACHE = pathlib.Path('/tmp/al_dense_cache'); CACHE.mkdir(exist_ok=True)
PSEUDO = pathlib.Path(REPO) / 'workflows/pseudos'

A_ANG = 4.0495
A_BOHR = A_ANG / BOHR_TO_ANG
cell = A_ANG * np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])
atoms = Atoms('Al', scaled_positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=True)
K_MESH, Q_MESH = (20, 20, 20), (10, 10, 10)
ECUTWFC, NBND = 52.0, 12
QE_IBRAV = {'ibrav': 2, 'celldm(1)': A_BOHR}
projections = spd_projections((0.0, 0.0, 0.0), 's;p')
NUM_WANN = len(projections)
real_lat = rl(atoms); recip = rc(atoms)
PSEUDOS = [read_norm_conserving(PSEUDO / 'Al.upf')]
TAU_FRAC = np.zeros((1, 3)); TYPES = np.array([0])
ECUT_RHO = 2.0 * ECUTWFC
NC = 24

log('electron scf/nscf/overlaps at 20^3 (write_unk) ...')
ov = qe.generate_overlaps(
    atoms, K_MESH, W, 'al', ecutwfc=ECUTWFC, scf_kpts=K_MESH, nbnd=NBND,
    num_wann=NUM_WANN, projections=projections,
    system_extra=dict(occupations='smearing', smearing='mp', degauss=0.02, **QE_IBRAV),
    pseudopotentials={'Al': 'Al.upf'}, pseudo_dir=PSEUDO, ncores=NC, write_unk=True,
)
kpts = ov['kpts']
log(f"QE Fermi = {ov['fermi_energy']:.4f} eV")

hrf = CACHE / 'hr.npz'
if hrf.exists():
    z = np.load(hrf)
    hr = HamiltonianR(H_R=torch.from_numpy(z['H_R']), R_vectors=z['R'],
                      degen=z['degen'], nw=int(z['nw']))
    Wg = z['W']; log('loaded cached Wannierisation')
else:
    log('wannierising (4 sp MLWFs at 20^3) ...')
    res = wannierize(atoms, K_MESH, kpts, mmn=ov['mmn'], amn=ov['amn'], eig=ov['eig'],
                     nnkpts=ov['nnkpts'], g_vectors=ov['g_vectors'], nw=NUM_WANN,
                     outer_window=(-1e3, 1e3),
                     frozen_window=(ov['fermi_energy'] - 2.0, ov['fermi_energy'] + 2.0),
                     guiding_centres=True, optimizer='cg', n_restarts=2,
                     dis_n_iter=2000, n_iter=2000, verbose=False)
    hr = res.hr
    Wg = (torch.einsum('kbw,kwn->kbn', res.dis.V, res.spread.U_final).detach().cpu().numpy()
          if res.dis is not None else res.spread.U_final.detach().cpu().numpy())
    np.savez(hrf, H_R=hr.H_R.detach().cpu().numpy(), R=hr.R_vectors,
             degen=hr.degen, nw=hr.nw, W=Wg)
    log(f'Omega_total = {res.omega_final * BOHR_TO_ANG**2:.4f} Ang^2')

# ---- phonons: ldisp 10^3 (force constants) with the 20^3 DFPT k-mesh ------
if not (W / 'al.fc').exists():
    log('DFPT scf at 20^3 k ...')
    scf_system = {'ecutwfc': ECUTWFC, 'occupations': 'smearing', 'smearing': 'mp',
                  'degauss': 0.02, **QE_IBRAV}
    write_pw_input(W / 'al.ph_scf.in', atoms,
                   control={'calculation': 'scf', 'prefix': 'al', 'outdir': './out_ph',
                            'pseudo_dir': str(PSEUDO)},
                   system=scf_system, electrons={'conv_thr': 1.0e-10},
                   pseudopotentials={'Al': 'Al.upf'}, kpoints=('automatic', K_MESH, (0, 0, 0)))
    run_pw(W / 'al.ph_scf.in', W / 'al.ph_scf.out', ncores=NC)
    log('ldisp DFPT on the 10^3 wedge (dyn for q2r) ...')
    write_ph_input(W / 'al.ph1.in', prefix='al', outdir='./out_ph', fildyn='al.elph.dyn',
                   nq=Q_MESH, tr2_ph=1.0e-14, extra={})
    run_pw(W / 'al.ph1.in', W / 'al.ph1.out', ncores=NC, pw='ph.x')
    write_q2r_input(W / 'al.q2r.in', fildyn='al.elph.dyn', flfrc='al.fc', zasr='crystal')
    run_pw(W / 'al.q2r.in', W / 'al.q2r.out', ncores=1, pw='q2r.x')
log('al.fc ready')
fc = apply_acoustic_sum_rule(read_force_constants(W / 'al.fc'))

# ---- dvscf: explicit-q run over OUR wedge, then star-rotate to the mesh ---
q_irr, _qw = irreducible_qpoints(atoms, Q_MESH)
log(f'{len(q_irr)} irreducible q of the 10^3 mesh')
if not (W / 'out_ph_irr' / '_ph0' / 'al.phsave' / f'patterns.{len(q_irr)}.xml').exists():
    scf_system = {'ecutwfc': ECUTWFC, 'occupations': 'smearing', 'smearing': 'mp',
                  'degauss': 0.02, **QE_IBRAV}
    write_pw_input(W / 'al.irr.scf.in', atoms,
                   control={'calculation': 'scf', 'prefix': 'al', 'outdir': './out_ph_irr',
                            'pseudo_dir': str(PSEUDO)},
                   system=scf_system, electrons={'conv_thr': 1.0e-10},
                   pseudopotentials={'Al': 'Al.upf'}, kpoints=('automatic', K_MESH, (0, 0, 0)))
    run_pw(W / 'al.irr.scf.in', W / 'al.irr.scf.out', ncores=NC)
    q_tpiba = q_irr @ recip * A_BOHR / (2 * np.pi)
    write_ph_input_explicit_q(W / 'al.ph_irr.in', prefix='al', outdir='./out_ph_irr',
                              fildyn='al.irr.dyn', qpoints_tpiba=q_tpiba,
                              tr2_ph=1.0e-14, extra={'fildvscf': 'dvscf'})
    log('wedge DFPT with fildvscf (47 q at 20^3 k) ...')
    run_pw(W / 'al.ph_irr.in', W / 'al.ph_irr.out', ncores=NC, pw='ph.x')
log('wedge dvscf ready')

sym = crystal_symmetry_operations(atoms)
routes = dvscf_io.dvscf_star_routes(q_irr, Q_MESH, sym)
n_tr = int((routes[:, 2] > 0).sum())
log(f'routes: {len(routes)} mesh points ({n_tr} via time reversal)')

gR_f = CACHE / 'gR.npz'
if gR_f.exists():
    z = np.load(gR_f)
    g_R, R_e, degen_e, R_q, degen_q = (z['g_R'], z['R_e'], z['degen_e'],
                                       z['R_q'], z['degen_q'])
    log('loaded cached g(Re,Rq)')
else:
    log('reading UNK files (8000 k) ...')
    u_all = np.stack([read_unk(W / f'UNK{ik+1:05d}.1')['u_nk'] for ik in range(len(kpts))])
    dv_grid = u_all.shape[2:]
    log(f'u_all {u_all.shape} ({u_all.nbytes/1e9:.1f} GB), dvscf grid {dv_grid}')
    _cache = {}
    def read_dvscf_q(iq):
        i, isym, tr = routes[iq]
        if i not in _cache:
            _cache[i] = dvscf_io.read_dvscf(W / 'out_ph_irr', 'al', int(i) + 1,
                                            dv_grid, nat=1)
        dv, _ = dvscf_io.rotate_dvscf(_cache[i], q_irr[i], int(isym), sym,
                                      real_lat, recip, TAU_FRAC,
                                      time_reversal=bool(tr))
        return dv
    qpts = monkhorst_pack(Q_MESH)
    log(f'building g(Re,Rq) over {len(qpts)} q x {len(kpts)} k (the long step) ...')
    g_R, R_e, degen_e, R_q, degen_q = elph.wannier_transform_elph(
        u_all, Wg, kpts, qpts, read_dvscf_q, K_MESH, Q_MESH, real_lat,
        pseudos=PSEUDOS, tau_frac=TAU_FRAC, types=TYPES, ecut_rho=ECUT_RHO)
    np.savez(gR_f, g_R=g_R, R_e=R_e, degen_e=degen_e, R_q=R_q, degen_q=degen_q)
    del u_all
    log(f'g_R {g_R.shape}')

# ---- fine-mesh sweep, 16^3 ... 40^3 ---------------------------------------
def phonon_fn(qf):
    b = interpolate_phonons(fc, real_lat, TAU_FRAC, qf)
    return CM1_TO_HARTREE * b.freq_cm1, b.eigvecs

SIGMA_E = elph.epw_degauss_to_sigma(0.05 * EV_TO_HARTREE)
eig_d, _ = elph.band_eigensystem(hr, monkhorst_pack((100, 100, 100)))
EF = elph.fermi_level_from_electron_count(eig_d, 3.0, SIGMA_E)
del eig_d
log(f'EF_MODEL = {EF * HARTREE_TO_EV:.4f} eV')
omq, _ = phonon_fn(monkhorst_pack(Q_MESH))
og = np.linspace(1e-5, omq.max() * 1.2, 300)
FSTHICK = 12.0 * SIGMA_E + omq.max()
rows, a2f_fine = [], []
for n in (16, 20, 24, 28, 32, 36, 40):
    mesh = (n,) * 3
    kf = monkhorst_pack(mesh)
    qi, qw = irreducible_qpoints(atoms, mesh)
    eig_f, U_f = elph.band_eigensystem(hr, kf)
    om_f, ev_f = phonon_fn(qi)
    n_ef = elph.fermi_surface_dos(eig_f, EF, SIGMA_E)
    a2f_f, lam_qnu = elph.alpha2f(
        eig_f, U_f, g_R, R_e, degen_e, R_q, degen_q, kf, qi, None, om_f, ev_f,
        fc['masses_amu'], fc['types'], fermi_energy=EF, dos_at_ef=n_ef,
        omega_grid=og, sigma_e=SIGMA_E, hr=hr, fsthick=FSTHICK, q_weights=qw,
        return_qnu=True)
    lam_f = float((qw[:, None] * np.clip(lam_qnu, 0, None)).sum())
    lc, wc = elph.eliashberg_moments(a2f_f, og)
    w2 = elph.eliashberg_omega_2(a2f_f, og)
    dog = og[1] - og[0]
    lowm = og < (5e-3 * EV_TO_HARTREE)
    lam_low = float(2 * np.sum(np.where(og > 0, a2f_f / np.where(og > 0, og, 1), 0)[lowm]) * dog)
    rows.append((n, lam_f, n_ef, wc[-1], w2))
    a2f_fine.append(a2f_f)
    log(f'{n}^3: nq_irr={len(qi):4d} N(eF)={n_ef:7.4f} lambda={lam_f:.4f} '
        f'coupling={lam_f / n_ef:.4f} wlog={wc[-1] * HARTREE_TO_EV * 1e3:5.2f} meV '
        f'w2={w2 * HARTREE_TO_EV * 1e3:5.2f} lam(<5meV)={lam_low:.4f}')
rows = np.array(rows)
dos_rows = []
for n in (64, 96, 128, 160, 192):
    e_d, _ = elph.band_eigensystem(hr, monkhorst_pack((n,) * 3))
    dos_rows.append((n, elph.fermi_surface_dos(e_d, EF, SIGMA_E)))
    log(f'[dos] {n}^3: N(eF)={dos_rows[-1][1]:.5f}')
dos_rows = np.array(dos_rows)

Ns, lam_N, nef_N, wlog_N, w2_N = rows.T
coupling = lam_N / nef_N; h = 1.0 / Ns
cs, rho = np.polyfit(h, coupling, 1)
resid = coupling - (rho + cs * h)
nefi = dos_rows[1:, 1].mean(); nefe = dos_rows[1:, 1].std(ddof=1)
lam = rho * nefi
lam_err = np.sqrt((np.std(resid, ddof=1) * nefi) ** 2 + (rho * nefe) ** 2)
wlog = wlog_N[-3:].mean(); wloge = wlog_N[-3:].std(ddof=1); w2m = w2_N[-3:].mean()
log(f'coupling -> {rho:.4f} Ha (rms {np.std(resid, ddof=1):.4f});  '
    f'N(eF) -> {nefi:.4f} +- {nefe:.4f} Ha^-1')
log(f'lambda = {lam:.4f} +- {lam_err:.4f}')
log(f'wlog = {wlog * HARTREE_TO_EV * 1e3:.2f} +- {wloge * HARTREE_TO_EV * 1e3:.2f} meV; '
    f'w2 = {w2m * HARTREE_TO_EV * 1e3:.2f} meV')
for mu in (0.10, 0.11, 0.12, 0.13):
    log(f'Tc(mu*={mu:.2f}) = {elph.allen_dynes_tc(lam, wlog, w2m, mu) / K_B_HARTREE:.2f} K')
np.savez(CACHE / 'al_dense_result.npz', rows=rows, dos_rows=dos_rows,
         a2f_fine=np.array(a2f_fine), og=og, EF=EF)
log('ALL DONE')
