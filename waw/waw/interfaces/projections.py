"""
Analytic projection specs (angular-momentum trial orbitals).

Shared by `interfaces.wannier90.io.write_nnkp` and
`interfaces.quantum_espresso.generate_overlaps(projections=...)` -- neither
reads nor writes a file itself, so it belongs to no single ecosystem-specific
interface.
"""

_SHELL_MR = {0: (1,), 1: (1, 2, 3), 2: (1, 2, 3, 4, 5), 3: (1, 2, 3, 4, 5, 6, 7)}
_SHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3}


def spd_projections(centre_frac, shells="s;p;d", *, r=1, zona=1.0,
                    zaxis=(0.0, 0.0, 1.0), xaxis=(1.0, 0.0, 0.0)):
    """Build analytic projection specs for `write_nnkp`'s `projections=` arg.

    Angular momentum shells (Wannier90 order) at a single centre, e.g.
    `spd_projections((0,0,0), "s;p;d")` -> 9 specs (1 s + 3 p + 5 d) with
    the standard mr ordering (p: pz,px,py; d: dz2,dxz,dyz,dx2-y2,dxy). Using
    the same shells/axes for two materials pins a common cubic-harmonic
    frame for orbital-by-orbital correspondence in alloy/CPA averaging.
    """
    c = tuple(float(x) for x in centre_frac)
    zx = tuple(float(x) for x in zaxis)
    xx = tuple(float(x) for x in xaxis)
    specs = []
    for tok in str(shells).replace(",", ";").split(";"):
        tok = tok.strip().lower()
        if not tok:
            continue
        l = _SHELL_L[tok]
        for mr in _SHELL_MR[l]:
            specs.append((c, l, mr, int(r), zx, xx, float(zona)))
    return specs
