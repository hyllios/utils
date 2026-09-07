"""
Tests for waw.core.spread's Pipek-Mezey (PM) atomic-locality functional
(`compute_pm_spread`) and its companion atom-to-column bookkeeping
(`interfaces.quantum_espresso.upf.atom_proj_column_atoms`).

Scope note: `use_pm_functional`/`Aat`/`atom_index` are wired into
`core.optim.minimize_spread` only (all six plain optimizers), not into the
higher-level `core.pipeline.wannierize`/`interfaces.ase.driver.wannierize`
-- same scope decision already made for `lbfgs`/`diis`/`rtr` not
extending to the SLWF/symmetrized variants. A comparison notebook calls
`minimize_spread` directly (same pattern as the optimizer-comparison
notebook), not the full pipeline wrapper.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from waw.core.spread import compute_pm_spread
from waw.core.optim import minimize_spread
from waw.core.init import svd_init
from waw.core.disentangle import disentangle
from waw.core.spread import rotate_overlaps
from waw.interfaces.quantum_espresso.upf import atom_proj_column_atoms
from waw.units import BOHR_TO_ANG


class _FakeAtoms:
    """Minimal stand-in for ase.Atoms -- atom_proj_column_atoms only calls
    get_chemical_symbols()."""
    def __init__(self, symbols):
        self._symbols = list(symbols)

    def get_chemical_symbols(self):
        return self._symbols


def test_atom_proj_column_atoms_matches_hand_worked_example():
    """Two Si atoms (s+p channels each) then one Ge atom (s+p+d) --
    ordering must be atom-by-atom (structure order), then that atom's
    species channels in ascending l, then 2l+1 columns per channel."""
    atoms = _FakeAtoms(["Si", "Si", "Ge"])
    species_to_radial = {
        "Si": {0: (np.array([1.0]), np.array([1.0])), 1: (np.array([1.0]), np.array([1.0]))},
        "Ge": {0: (np.array([1.0]), np.array([1.0])), 1: (np.array([1.0]), np.array([1.0])),
              2: (np.array([1.0]), np.array([1.0]))},
    }
    idx = atom_proj_column_atoms(atoms, species_to_radial)
    # Si0: s(1) + p(3) = 4 columns; Si1: 4 columns; Ge: s(1)+p(3)+d(5) = 9 columns
    expected = np.array([0]*4 + [1]*4 + [2]*9)
    np.testing.assert_array_equal(idx, expected)


def test_atom_proj_column_atoms_respects_l_sort_order_not_dict_order():
    """The species dict's key insertion order must NOT matter -- channels
    are always taken in ascending-l order (matching write_atom_proj_ext's
    own `sorted(radial)` and pw2wannier90's file-order read-back)."""
    atoms = _FakeAtoms(["X"])
    unsorted = {"X": {1: (np.array([1.0]), np.array([1.0])), 0: (np.array([1.0]), np.array([1.0]))}}
    idx = atom_proj_column_atoms(atoms, unsorted)
    # l=0 (1 column) must come before l=1 (3 columns) regardless of dict order
    np.testing.assert_array_equal(idx, np.array([0, 0, 0, 0]))


def _random_unitary(nk, nw, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(nk, nw, nw, dtype=torch.complex128, generator=g)
    return torch.linalg.qr(A)[0]


def test_pm_omega_is_maximized_not_minimized_convention():
    """Omega_PM = -sum Q^2 must be more negative (Omega_PM smaller) the
    MORE concentrated the Mulliken charges are on fewer atoms -- a
    fully-localized-on-one-atom Wannier function must give a lower
    (more negative) Omega_PM than one split evenly across two atoms."""
    nk, nw = 3, 1
    U = torch.ones(nk, 1, 1, dtype=torch.complex128)   # trivial 1x1 "unitary" phase
    atom_index = torch.tensor([0, 1])

    # fully on atom 0: Aat[:, :, 0] = 1, Aat[:, :, 1] = 0
    Aat_localized = torch.zeros(nk, 1, 2, dtype=torch.complex128)
    Aat_localized[:, 0, 0] = 1.0
    Omega_localized, Q_localized = compute_pm_spread(U, Aat_localized, atom_index)

    # split evenly: overlap 1/sqrt(2) with each atom's orbital
    Aat_split = torch.zeros(nk, 1, 2, dtype=torch.complex128)
    Aat_split[:, 0, 0] = 1.0 / 2 ** 0.5
    Aat_split[:, 0, 1] = 1.0 / 2 ** 0.5
    Omega_split, Q_split = compute_pm_spread(U, Aat_split, atom_index)

    assert Omega_localized.item() < Omega_split.item()
    assert Q_localized[0].sum().item() == pytest.approx(Q_split[0].sum().item(), abs=1e-10)


def test_pm_spread_is_differentiable():
    """Autodiff correctness -- no hand-derived gradient, same as every
    other functional in this module."""
    torch.manual_seed(0)
    nk, nw, natproj = 4, 2, 5
    U = _random_unitary(nk, nw, seed=1).detach().requires_grad_(True)
    Aat = torch.randn(nk, nw, natproj, dtype=torch.complex128)
    atom_index = torch.tensor([0, 0, 1, 1, 2])

    Omega, Q = compute_pm_spread(U, Aat, atom_index)
    Omega.backward()
    assert U.grad is not None
    assert torch.isfinite(U.grad).all()


SI_EXT_DIR = Path(__file__).parent.parent / "workflows" / "w90tutorial" / "runs" / "si_ext_proj"
HAS_SI_EXT = (SI_EXT_DIR / "si_ext.mmn").exists()


@pytest.mark.skipif(not HAS_SI_EXT, reason="cached atom_proj_ext Si overlaps not found")
def test_pm_localizes_more_atomically_than_mv_on_real_data():
    """
    End-to-end validation on real (already-committed, DFT-computed)
    atom_proj_ext Si overlap data (`w90tutorial/35_silicon_atom_proj_ext`):
    starting from the IDENTICAL disentangled U_init, optimizing the PM
    functional instead of MV must concentrate more Mulliken charge on
    fewer atoms (higher per-WF max atomic charge) than the MV optimum
    does, at the cost of a worse (higher) ordinary MV spread -- the real,
    expected localization-criterion tradeoff, not a coincidence of this
    one run.
    """
    from ase.build import bulk
    from waw.core.init import svd_init
    from waw.interfaces.ase.driver import build_wannier_data
    from waw.interfaces.ase.structure import recip_lattice
    from waw.interfaces.quantum_espresso.upf import read_pswfc
    from waw.interfaces.wannier90.io import read_mmn, read_amn, read_nnkp
    from waw.units import EV_TO_HARTREE

    Mmn_np, _ = read_mmn(SI_EXT_DIR / "si_ext.mmn")
    Amn_np = read_amn(SI_EXT_DIR / "si_ext.amn")
    nnkp = read_nnkp(SI_EXT_DIR / "si_ext.nnkp")
    eig_np = np.loadtxt(SI_EXT_DIR / "si_ext.eig")
    nbnd = int(eig_np[:, 0].max()); nk = int(eig_np[:, 1].max())
    eig_np = eig_np[:, 2].reshape(nk, nbnd)

    atoms = bulk("Si", "diamond", a=5.43)
    radial = read_pswfc(Path(__file__).parent.parent / "workflows" / "pseudos" / "Si.upf")
    atom_index = torch.tensor(atom_proj_column_atoms(atoms, {"Si": radial}))

    wdata = build_wannier_data(recip_lattice(atoms), nnkp["kpoints"], Mmn_np, Amn_np, eig_np,
                              nnkp["nnkpts"], nnkp["g_vectors"])
    dis = disentangle(wdata.Mmn, wdata.eig, wdata.wb, wdata.kb_idx, nw=8, Amn=wdata.Amn,
                      proj_min=0.01, proj_max=0.95, frozen_window=(-1.0e6, 8.7 * EV_TO_HARTREE),
                      n_iter=1000, conv_tol=1e-10)

    Mmn_opt = rotate_overlaps(dis.V, wdata.Mmn, wdata.kb_idx)
    Aat_sub = torch.einsum("kmi,kmj->kij", dis.V.conj(), wdata.Amn)
    U_init = svd_init(Aat_sub)

    res_mv = minimize_spread(U_init, Mmn_opt, wdata.wb, wdata.bvecs, wdata.kb_idx,
                             optimizer="cg", lr=1.0, n_iter=500, conv_tol=1e-10, conv_window=5)
    _, Q_mv = compute_pm_spread(res_mv.U_final, Aat_sub, atom_index)

    res_pm = minimize_spread(U_init, Mmn_opt, wdata.wb, wdata.bvecs, wdata.kb_idx,
                             optimizer="cg", lr=1.0, n_iter=500, conv_tol=1e-10, conv_window=5,
                             use_pm_functional=True, Aat=Aat_sub, atom_index=atom_index)
    _, Q_pm = compute_pm_spread(res_pm.U_final, Aat_sub, atom_index)

    max_charge_mv = Q_mv.max(dim=1).values.mean().item()
    max_charge_pm = Q_pm.max(dim=1).values.mean().item()
    assert max_charge_pm > max_charge_mv, (
        f"PM should concentrate charge more than MV: PM={max_charge_pm:.4f}, MV={max_charge_mv:.4f}"
    )

    omega_mv_ang2 = res_mv.Omega * BOHR_TO_ANG**2
    omega_pm_ang2 = res_pm.Omega * BOHR_TO_ANG**2
    assert omega_pm_ang2 > omega_mv_ang2, (
        "PM's ordinary MV-reported spread should be worse (higher) than MV's own optimum "
        f"(PM={omega_pm_ang2:.4f}, MV={omega_mv_ang2:.4f} Ang^2)"
    )
