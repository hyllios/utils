"""
WannierData — central data container for the optimization engine.

Holds every tensor the core numerics need, in **atomic units**: lengths are
Bohr (b-vectors Bohr^-1, shell weights Bohr^2) and energies are Hartree.
Nothing in this module reads files or touches physical units; the Wannier90
loader (interfaces.wannier90.loader) and the ASE driver build instances of this
class and own all unit conversion (eV<->Hartree, Angstrom<->Bohr).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class WannierData:
    """
    All tensors needed by the Wannier optimization engine.

    Shapes
    ------
    Mmn   : (nk, nnb, nb, nb)  complex  overlap matrices M^(k,b)_{mn}
    Amn   : (nk, nb, nw)       complex  projection matrices A^(k)_{mn}
    eig   : (nk, nb)           real     eigenvalues in Hartree
    kpts  : (nk, 3)            real     k-points in crystal coordinates
    bvecs : (nk, nnb, 3)       real     b-vectors in Cartesian (Bohr^-1), k-specific
    wb    : (nnb,)             real     shell weights
    kb_idx: (nk, nnb)          long     index of k+b in kpts (0-based)
    params: dict               raw key-value pairs from .win

    Derived
    -------
    nk, nb, nw, nnb: dimensions extracted from tensor shapes.
    """

    Mmn:    torch.Tensor
    Amn:    torch.Tensor
    eig:    torch.Tensor
    kpts:   torch.Tensor
    bvecs:  torch.Tensor
    wb:     torch.Tensor
    kb_idx: torch.Tensor
    params: dict = field(default_factory=dict)

    # Convenience properties so callers never have to index .shape directly.

    @property
    def nk(self) -> int:
        return self.Mmn.shape[0]

    @property
    def nnb(self) -> int:
        return self.Mmn.shape[1]

    @property
    def nb(self) -> int:
        return self.Mmn.shape[2]

    @property
    def nw(self) -> int:
        return self.Amn.shape[2]

    def __repr__(self) -> str:
        return (
            f"WannierData(nk={self.nk}, nb={self.nb}, "
            f"nw={self.nw}, nnb={self.nnb}, "
            f"device={self.Mmn.device})"
        )
