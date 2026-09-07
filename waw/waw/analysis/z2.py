"""
Z2 topological invariants via the Wilson-loop / hybrid Wannier charge
center (WCC) evolution method (Yu, Qi, Bernevig, Fang, Dai, PRB 84,
075119 (2011); Soluyanov & Vanderbilt, PRB 83, 235401 (2011) "largest
gap" tracking), applied directly to a Wannier tight-binding H(k). Not a
Wannier90/postw90 feature (that's what standalone tools like Z2Pack/
WannierTools do) -- genuinely new capability for this project.

Method: for a time-reversal-invariant-momentum (TRIM) plane (one crystal
direction fixed at 0 or 1/2), sweep the other two crystal directions'
Wilson loop over half the 2D Brillouin zone (0 to 1/2, using the k -> -k
symmetry at a TRIM plane). At each step, track the position of the
*largest gap* between sorted WCC (recomputed fresh at every step, not
fixed), and count a WCC as having "crossed" between consecutive steps if
it now lies strictly between the previous and current gap positions. The
parity of the total crossing count is the Z2 invariant of the plane. Six
such planes (kx/ky/kz = 0, 1/2) give the strong index nu0 and the three
weak indices nu1, nu2, nu3.

Validated against Z2Pack (Gresch et al., Comput. Phys. Commun. 224, 165
(2018)) on both a 2-band (time-reversal-doubled Qi-Wu-Zhang) and a 4-band
(Clifford-algebra Dirac lattice, the standard strong-TI model) synthetic
model -- exact per-plane agreement including the full nu0=1 strong-TI
case, see tests/test_analysis_z2.py and tests/test_analysis_z2_vs_z2pack.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.hamiltonian import HamiltonianR
from .topology import _eigvecs_at


def _overlap_matrix(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """
    Unitarized overlap (parallel-transport link) between two bases of
    occupied-subspace eigenvectors: V1^dagger V2 with its singular values
    discarded (U @ Vh from an SVD) -- the matrix generalization of
    `topology._link_variable`'s determinant-only U(1) link.
    """
    U, _, Vh = np.linalg.svd(V1.conj().T @ V2)
    return U @ Vh


def wilson_loop_wcc(
    hr: HamiltonianR, band_group: tuple, plane: tuple,
    fixed_index: int, fixed_value: float, loop_value: float,
    n_loop: int = 41,
) -> np.ndarray:
    """
    Wannier charge centers (WCC), as angles in (-pi, pi], of the Wilson
    loop for `band_group` closing over `plane[1]` (0 -> 1, periodic), at
    `plane[0] = loop_value` and the third crystal direction
    `fixed_index = fixed_value`.

    Returns (len(band_group),) angles, unordered.
    """
    a, b = plane
    V = []
    for j in range(n_loop):
        k = np.zeros(3)
        k[a] = loop_value
        k[b] = j / n_loop
        k[fixed_index] = fixed_value
        V.append(_eigvecs_at(hr, k, band_group))
    V.append(V[0])   # close the loop (b=1 is the same physical k as b=0)

    n = len(band_group)
    W = np.eye(n, dtype=np.complex128)
    for j in range(n_loop):
        # Dagger on the *destination* eigenvectors, not the source: this is
        # what makes the loop product telescope to a similarity transform
        # (gauge-invariant spectrum) under an arbitrary V_j -> V_j @ U_j
        # regauging at every point, including within an exactly-degenerate
        # band_group subspace. The other dagger order looks fine on a
        # well-separated 2-band model (small residual gauge drift doesn't
        # flip the parity) but silently breaks on a genuinely degenerate
        # multi-band subspace.
        W = _overlap_matrix(V[j + 1], V[j]) @ W

    return np.angle(np.linalg.eigvals(W))


def _largest_gap(wcc_frac: np.ndarray) -> float:
    """
    Position (fractional, in [0, 1)) of the largest gap between sorted
    `wcc_frac` (fractional WCC, mod 1) -- same convention and algorithm as
    Z2Pack's `_utils._gapfind`.
    """
    w = np.sort(wcc_frac % 1.0)
    ext = np.concatenate([w, [w[0] + 1.0]])
    gaps = np.diff(ext)
    idx = int(np.argmax(gaps))
    return float((w[idx] + gaps[idx] / 2) % 1.0)


def _crossed(gap_prev: float, gap_curr: float, x: float) -> bool:
    """
    Whether fractional WCC `x` lies strictly between `gap_prev` and
    `gap_curr` -- the Soluyanov-Vanderbilt sign function for one step of
    largest-gap WCC tracking (Z2Pack's `_utils._sgng`).
    """
    return min(gap_prev, gap_curr) < x < max(gap_prev, gap_curr)


def z2_invariant_plane(
    hr: HamiltonianR, band_group: tuple, plane: tuple,
    fixed_index: int, fixed_value: float,
    mesh: int = 101, n_loop: int = 41,
) -> int:
    """
    Z2 invariant of the TRIM plane `fixed_index = fixed_value` (0 or 0.5),
    via WCC evolution over `plane[0]` from 0 to 1/2 (the other TRIM-plane
    coordinate is swept by the Wilson loop `plane[1]` inside
    `wilson_loop_wcc`) -- only half the range is needed since a TRIM
    plane's WCC spectrum at `plane[0]=t` and `plane[0]=-t` coincide.

    At each step, the largest gap of the current WCC spectrum is located
    fresh (not held fixed), and each WCC in the next step is checked for
    having crossed between the previous and current gap position; the
    parity of the total crossing count is the invariant (Soluyanov &
    Vanderbilt 2011).
    """
    t_values = np.linspace(0.0, 0.5, mesh)

    def _wcc_frac(t):
        angles = wilson_loop_wcc(hr, band_group, plane, fixed_index, fixed_value,
                                 loop_value=t, n_loop=n_loop)
        return angles / (2 * np.pi)   # radians -> fractional, mod 1 in _largest_gap

    wcc_prev = _wcc_frac(t_values[0])
    gap_prev = _largest_gap(wcc_prev)

    inv = 1
    for t in t_values[1:]:
        wcc_curr = _wcc_frac(t)
        gap_curr = _largest_gap(wcc_curr)
        for x in wcc_curr % 1.0:
            if _crossed(gap_prev, gap_curr, x):
                inv *= -1
        gap_prev = gap_curr

    return 0 if inv == 1 else 1


@dataclass
class Z2Result:
    """Strong (nu0) and weak (nu1, nu2, nu3) Z2 indices of a 3D insulator."""
    nu0: int
    nu1: int
    nu2: int
    nu3: int
    z2_planes: dict   # {'x0': int, 'x1': int, 'y0': int, 'y1': int, 'z0': int, 'z1': int}
    consistent: bool  # whether all three normal directions agree on nu0


def z2_invariants_3d(
    hr: HamiltonianR, band_group: tuple, mesh: int = 101, n_loop: int = 41,
) -> Z2Result:
    """
    The four Z2 invariants (nu0; nu1 nu2 nu3) of a 3D insulator (Fu-Kane-
    Mele classification), from the Wilson-loop invariant of the six TRIM
    planes kx/ky/kz = 0, 1/2 (`z2_invariant_plane`).

    nu0 = z2(k_a=0) XOR z2(k_a=1/2) for any crystal direction a -- all
    three should agree (`consistent`); nu1/nu2/nu3 are conventionally the
    kx=1/2 / ky=1/2 / kz=1/2 plane's own invariant.
    """
    planes = {
        "x0": ((1, 2), 0, 0.0), "x1": ((1, 2), 0, 0.5),
        "y0": ((2, 0), 1, 0.0), "y1": ((2, 0), 1, 0.5),
        "z0": ((0, 1), 2, 0.0), "z1": ((0, 1), 2, 0.5),
    }
    z2 = {
        name: z2_invariant_plane(hr, band_group, plane, fixed_index, fixed_value,
                                 mesh=mesh, n_loop=n_loop)
        for name, (plane, fixed_index, fixed_value) in planes.items()
    }

    nu0_x = z2["x0"] ^ z2["x1"]
    nu0_y = z2["y0"] ^ z2["y1"]
    nu0_z = z2["z0"] ^ z2["z1"]
    consistent = (nu0_x == nu0_y == nu0_z)

    return Z2Result(nu0=nu0_z, nu1=z2["x1"], nu2=z2["y1"], nu3=z2["z1"],
                    z2_planes=z2, consistent=consistent)
