"""
Initialization of the gauge matrices U(k) before spread minimization.

Two strategies:

  svd_init(Amn)
    Construct U(k) as the closest unitary to the projection matrix A(k)
    via the polar decomposition: SVD(A) = P Σ Q† → U = P Q†.
    This is how Wannier90 initialises its minimisation and is the default.

  random_unitary(nk, nw)
    Sample a Haar-random unitary via QR decomposition of a complex Gaussian
    matrix.  Used for random restarts in the global optimizer (Part 7).
"""

import torch
from torch import Tensor


def svd_init(Amn: Tensor) -> Tensor:
    """
    Initialise U(k) from the projection matrices A(k) = <psi_mk | g_n>.

    The closest unitary to A (in Frobenius norm) is given by the polar
    decomposition: if SVD(A) = P Σ Q† then U = P Q†.

    For the isolated-band case (nb == nw), U is square unitary.
    For the entangled case (nb > nw), A is (nb, nw) and U is (nb, nw)
    — a semi-unitary matrix (U† U = I_nw).  This is handled correctly
    by torch.linalg.svd with full_matrices=False.

    Args:
      Amn: (nk, nb, nw) complex  projection matrices from .amn file

    Returns:
      U  : (nk, nb, nw) complex  semi-unitary (or unitary if nb == nw)
    """
    # full_matrices=False gives P: (nk, nb, nw), S: (nk, nw), Qh: (nk, nw, nw)
    P, _singular_values, Qh = torch.linalg.svd(Amn, full_matrices=False)
    U = torch.matmul(P, Qh)
    return U


def random_unitary(
    nk: int,
    nw: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.complex128,
    generator: torch.Generator | None = None,
) -> Tensor:
    """
    Sample Haar-random unitary matrices via QR decomposition.

    Drawing A ~ CN(0,1) and computing QR(A) = Q R, then Q is Haar-distributed
    on U(nw) (after fixing the sign/phase of R's diagonal, but for our purposes
    of random restarts the phase correction is not critical).

    Args:
      nk       : number of k-points
      nw       : number of Wannier functions
      device   : torch device
      dtype    : complex dtype
      generator: optional RNG for reproducibility

    Returns:
      U: (nk, nw, nw) complex unitary matrices
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32

    A_real = torch.randn(nk, nw, nw, dtype=real_dtype, device=device,
                         generator=generator)
    A_imag = torch.randn(nk, nw, nw, dtype=real_dtype, device=device,
                         generator=generator)
    A = torch.complex(A_real, A_imag)
    Q, R = torch.linalg.qr(A)

    # Fix the phase of each column so the diagonal of R is real positive.
    # This makes the distribution exactly Haar-uniform.
    phases = R.diagonal(dim1=-2, dim2=-1)        # (nk, nw)
    phases = phases / phases.abs()               # unit complex numbers
    Q = Q * phases.unsqueeze(-2)                 # broadcast over rows

    return Q
