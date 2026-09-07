"""Which face of the stack the decimation exposes, and which cell it reports.

Reference is an explicitly built long chain, diagonalised: the surface Green's
function of a semi-infinite chain has to equal the LDOS on the terminal cell of
a chain long enough that the far end cannot be felt. An SSH chain is used
because its two terminations are physically distinguishable -- cutting the weak
bond leaves a zero-energy edge state, cutting the strong bond does not -- so a
mix-up cannot hide in a symmetric spectrum.

Two things this pins down:

  * `SurfaceLayers.exposed`. The principal-layer blocks run in order of
    increasing layer index and H01 couples upward, so for face="top" the exposed
    cell is the LAST block. Reading block 0 (as the code did) returns the first
    BURIED cell whenever num_pl > 1 -- a silent error, since the result still
    looks like a surface spectral function.
  * `face=`. Both stable terminations of a polar crystal live at opposite ends
    of the stack; without this the second one is unreachable except by
    re-running the electronic structure on a mirrored cell.
"""
import numpy as np
import pytest
import torch

from waw.core.hamiltonian import HamiltonianR
from waw.analysis.surface import build_surface_layers, surface_spectral_function

V, W, X, C, D = 0.4, 1.0, 0.25, 2.0, 0.7      # weak, strong, 2-cell hop; cell, A-B
LAT = np.diag([1.0, 1.0, C])
CEN = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, D]])       # A at 0, B at D
E = np.linspace(-2.4, 2.4, 49)
ETA = 0.05
K0 = np.zeros((1, 2))


def model(two_cell=0.0):
    """SSH along z. Intracell A-B = V; intercell B(n) -> A(n+1) = W.
    `two_cell` adds B(n) -> A(n+2), which pushes num_pl to 2."""
    Rs = [[0, 0, 0], [0, 0, 1], [0, 0, -1]]
    if two_cell:
        Rs += [[0, 0, 2], [0, 0, -2]]
    H = np.zeros((len(Rs), 2, 2), dtype=complex)
    H[0] = [[0.0, V], [V, 0.0]]
    H[1][1, 0] = W
    H[2][0, 1] = W
    if two_cell:
        H[3][1, 0] = two_cell
        H[4][0, 1] = two_cell
    return HamiltonianR(H_R=torch.tensor(H), R_vectors=np.array(Rs),
                        degen=np.ones(len(Rs), dtype=int), nw=2)


def chain_ldos(first_bond_weak, sites, two_cell=0.0, n=3000):
    """LDOS on `sites` of a long chain, indexed from its terminal site inward.

    `first_bond_weak` says whether the bond leaving the terminal site is V.
    The 2-cell hop joins B(n) to A(n+2); counting inward from a B-terminated
    end that is index i -> i - 3, so it only exists once 3 sites are behind.
    """
    b0, b1 = (V, W) if first_bond_weak else (W, V)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i + 1] = H[i + 1, i] = b0 if i % 2 == 0 else b1
    if two_cell:
        for i in range(0, n, 2):
            if i - 3 >= 0:
                H[i, i - 3] = H[i - 3, i] = two_cell
    ev, U = np.linalg.eigh(H)
    g = (np.abs(U[list(sites)]) ** 2).sum(axis=0)
    return np.array([(g * ETA / np.pi / ((e - ev) ** 2 + ETA ** 2)).sum() for e in E])


def A_surf(hr, termination, face, two_cell=0.0):
    return surface_spectral_function(
        hr, LAT, (0, 0, 1), K0, E, eta=ETA, termination=termination,
        wf_centres=CEN, face=face).A_surface[0]


# --- num_pl = 1: the whole machinery against the exact chain ---------------

@pytest.mark.parametrize("termination,weak_first", [(1, True), (0, False)])
def test_single_principal_layer_reproduces_the_exact_chain(termination, weak_first):
    """termination 1 exposes B (severing its W bond, leaving V inward);
    termination 0 exposes A (severing V, leaving W inward)."""
    got = A_surf(model(), termination, "top")
    ref = chain_ldos(weak_first, (0, 1))
    assert np.abs(got - ref).max() / ref.max() < 1e-12


def test_the_two_terminations_differ_by_an_edge_state():
    """Not a symmetry: cutting the strong bond kills the zero-energy state.
    Without this the test above could pass on a spectrum that is termination
    independent, which would prove nothing."""
    mid = len(E) // 2
    assert abs(E[mid]) < 1e-12
    weak_cut = A_surf(model(), 1, "top")[mid]
    strong_cut = A_surf(model(), 0, "top")[mid]
    assert weak_cut > 50 * strong_cut


# --- num_pl = 2: the exposed cell must be the outermost one ----------------

def test_exposed_cell_is_the_outermost_not_the_buried_one():
    """The regression that motivated `SurfaceLayers.exposed`: with num_pl = 2,
    taking block 0 reproduced the SECOND cell instead of the surface cell."""
    hr = model(X)
    layers = build_surface_layers(hr, LAT, (0, 0, 1), termination=1,
                                  wf_centres=CEN)
    assert layers.num_pl == 2
    got = A_surf(hr, 1, "top", two_cell=X)
    outer = chain_ldos(True, (0, 1), two_cell=X)
    second = chain_ldos(True, (2, 3), two_cell=X)
    assert np.abs(got - outer).max() / outer.max() < 1e-12
    # and it is genuinely a different curve, so the check has teeth
    assert np.abs(outer - second).max() / outer.max() > 0.1


def test_exposed_slice_tracks_face_and_num_pl():
    hr = model(X)
    top = build_surface_layers(hr, LAT, (0, 0, 1), termination=1,
                               wf_centres=CEN, face="top")
    bot = build_surface_layers(hr, LAT, (0, 0, 1), termination=1,
                               wf_centres=CEN, face="bottom")
    assert top.exposed == slice((top.num_pl - 1) * top.nw, top.num_pl * top.nw)
    assert bot.exposed == slice(0, bot.nw)


# --- face= : the other end of the same crystal -----------------------------

def test_bottom_face_exposes_the_other_end_of_the_stack():
    """face="bottom" on the SAME H(R) must give the surface at the far end.

    Exposing sublayer 1 (B) downward severs B's V bond, so the bond leaving the
    terminal site inward is W -- the opposite of what face="top" gives, and the
    edge state must therefore disappear.
    """
    got = A_surf(model(), 1, "bottom")
    ref = chain_ldos(False, (0, 1))
    assert np.abs(got - ref).max() / ref.max() < 1e-12
    mid = len(E) // 2
    assert got[mid] < A_surf(model(), 1, "top")[mid] / 50


def test_face_bottom_of_one_sublayer_equals_face_top_of_its_neighbour():
    """Cutting the same bond from either side exposes the same pair of faces:
    the bond between sublayers t-1 and t is severed by face="top" on t-1 and by
    face="bottom" on t, so those two runs must agree."""
    a = A_surf(model(), 0, "top")        # cut above A  (severs V)
    b = A_surf(model(), 1, "bottom")     # cut below B  (severs the same V)
    assert np.abs(a - b).max() / max(a.max(), b.max()) < 1e-12


def test_bulk_projection_is_face_independent():
    """A_bulk restores both self-energies, so it cannot depend on which side is
    called the surface."""
    kw = dict(eta=ETA, termination=1, wf_centres=CEN)
    t = surface_spectral_function(model(X), LAT, (0, 0, 1), K0, E, face="top", **kw)
    b = surface_spectral_function(model(X), LAT, (0, 0, 1), K0, E, face="bottom", **kw)
    assert np.abs(t.A_bulk - b.A_bulk).max() / t.A_bulk.max() < 1e-12


def test_face_is_validated():
    with pytest.raises(ValueError, match="face must be"):
        build_surface_layers(model(), LAT, (0, 0, 1), face="sideways")


# --- item 5: hr_cutoff quietly deciding num_pl ------------------------------

def test_cutoff_that_truncates_the_hamiltonian_warns():
    """A cutoff above a real coupling drops a whole principal layer, and the
    spectral function changes materially without saying so."""
    hr = model(X)
    with pytest.warns(RuntimeWarning, match="discarded couplings reaching layer"):
        coarse = build_surface_layers(hr, LAT, (0, 0, 1), hr_cutoff=2 * X,
                                      termination=1, wf_centres=CEN)
    assert coarse.num_pl == 1
    fine = build_surface_layers(hr, LAT, (0, 0, 1), hr_cutoff=X / 100,
                                termination=1, wf_centres=CEN)
    assert fine.num_pl == 2


def test_cutoff_below_every_coupling_is_silent(recwarn):
    build_surface_layers(model(X), LAT, (0, 0, 1), hr_cutoff=1e-12,
                         termination=1, wf_centres=CEN)
    assert not [w for w in recwarn if "discarded couplings" in str(w.message)]
