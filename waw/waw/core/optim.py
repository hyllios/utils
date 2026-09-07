"""
Spread functional minimization on the unitary manifold.

U(k) lives in the unitary group U(nw), a Riemannian manifold.  Naive
Euclidean gradient steps would immediately leave the manifold, so we use
Riemannian optimization: project the Euclidean gradient onto the tangent
space and retract the result back onto the manifold after each step.

Tangent space at U ∈ U(nw):
    T_U U(nw) = { U S : S skew-Hermitian }

Riemannian gradient (projection of Euclidean gradient G onto T_U):
    G_riem = G - U @ sym(U† G),   sym(A) = (A + A†) / 2

Retraction (QR-based, canonical choice):
    retract(U, Δ) = Q   from   QR(U + Δ),  with phase fix on R's diagonal.

Six optimizers are implemented natively (no geoopt — geoopt does not support
complex128 and corrupts the retraction for complex matrices):

  SGD   : Riemannian steepest descent with Armijo backtracking line search.
          Robust; matches wannier90's own algorithm in spirit.
  CG    : Riemannian nonlinear conjugate gradients (Fletcher-Reeves, periodic
          steepest-descent restarts, parabolic line search), matching
          wannier90's own minimizer. Tends to reach a tighter Omega than
          Adam with fewer iterations.
  Adam  : Riemannian Adam with QR retraction (Bécigneul & Ganea, ICLR 2019).
          Fewer hyperparameters to tune, robust default for a first pass.
  LBFGS : Riemannian limited-memory BFGS (two-loop recursion), same parabolic
          line search as CG. A quasi-Newton method: curvature pairs (s, y)
          approximate the inverse Hessian action on the gradient instead of
          just combining gradient directions (CG) or rescaling them
          elementwise (Adam).
  DIIS  : Pulay-mixing (DIIS) extrapolation of the iterate sequence, using
          each iterate's Riemannian gradient as its "error vector" (the
          standard DIIS residual choice for a fixed-point/minimization
          problem, where the residual vanishes at a stationary point).
          Falls back to an Armijo-backtracked steepest-descent step
          whenever the extrapolated point doesn't improve on it, since raw
          Pulay mixing has no intrinsic descent guarantee.
  RTR   : Riemannian trust-region (Absil, Baker & Gallivan, Found. Comput.
          Math. 7, 303 (2007)) -- the only second-order method here. Each
          iteration builds a local quadratic model from the Riemannian
          gradient and an (approximate, projection-based) Riemannian
          Hessian-vector product -- via double backward through the same
          autodiff'd Omega(U) every other optimizer already uses for its
          gradient -- approximately minimizes it within a trust region via
          truncated CG (Steihaug-Toint), and accepts/rejects the retracted
          step while growing/shrinking the trust radius from the ratio of
          actual to predicted decrease. Substantially more expensive per
          outer iteration (up to `tcg_max_iter` Hessian-vector products,
          each its own double-backward pass) than the gradient-only
          methods, in exchange for genuine curvature information.

CG, Adam, LBFGS and DIIS's momentum/curvature-pair/history terms are carried
across iterations as raw tangent-space matrices without parallel transport
between the successive retraction points (U_t is close to U_{t+1}, so this is
the same "trivialized tangent space" approximation every one of them relies
on, not an extra approximation specific to any single method). RTR carries no
state between outer iterations besides the trust radius (a scalar) and does
not need this approximation for its own curvature information, though its
Hessian is itself only an approximation to the exact Riemannian one (see
`_riemannian_hvp`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .spread import (
    compute_spread, compute_spread_from_M_tilde, rotate_overlaps,
    refine_guiding_centres, compute_slwf_spread, compute_slwf_spread_from_M_tilde,
    compute_ss_spread, compute_ss_spread_from_M_tilde, compute_pm_spread,
)
from .sitesym import SiteSymmetry, broadcast_matrix, reduce_gradient_to_irr, symmetrize_u_irr


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SpreadResult:
    """Convergence history and final state from a minimization run."""

    U_final   : Tensor
    centres   : Tensor            # (nw, 3)
    Omega     : float
    Omega_I   : float
    Omega_D   : float
    Omega_OD  : float
    history   : list[float] = field(default_factory=list)
    converged : bool = False
    rguide    : Tensor | None = None   # (nw, 3), set when guiding_centres=True;
                                       # reuse for later re-evaluation of
                                       # Omega/centres at U_final to stay
                                       # consistent with the guided phase.
    Omega_IOD : float | None = None    # SLWF only: Omega_I and Omega_OD are
                                       # not separately decomposable when
                                       # slwf_num < nw (see
                                       # core.spread._slwf_spread_from_M_tilde);
                                       # Omega_I is set to this same combined
                                       # value and Omega_OD to 0.0, so plain
                                       # Omega_I+Omega_D+Omega_OD callers still
                                       # get the right total, and Omega_IOD is
                                       # the one to read for the breakdown.
    Omega_nu  : float | None = None    # SLWF centre-constraint penalty
                                       # (slwf_constrain=True only), 0.0 else.


# ---------------------------------------------------------------------------
# Riemannian geometry utilities
# ---------------------------------------------------------------------------

def _riemannian_gradient(U: Tensor, G_euc: Tensor) -> Tensor:
    """
    Project the Euclidean gradient G_euc onto the tangent space of U(nw) at U.

        G_riem = G - U @ sym(U† G),    sym(A) = (A + A†) / 2

    Result satisfies U† G_riem + G_riem† U = 0.
    """
    UhG = torch.matmul(U.conj().transpose(-1, -2), G_euc)
    sym = (UhG + UhG.conj().transpose(-1, -2)) / 2
    return G_euc - torch.matmul(U, sym)


def _qr_retract(U: Tensor, direction: Tensor) -> Tensor:
    """
    QR retraction: project U + direction back onto the unitary manifold.
    The sign fix on R's diagonal makes the retraction smooth and canonical.
    """
    Q, R = torch.linalg.qr(U + direction)
    phases = R.diagonal(dim1=-2, dim2=-1).sgn()   # ±1 per column
    return Q * phases.unsqueeze(-2)


def _omega_and_grad(
    U: Tensor, Mmn, wb, bvecs, kb_idx, rguide: Tensor | None = None,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False, Aat: Tensor | None = None, atom_index: Tensor | None = None,
):
    """
    Compute Omega and its Euclidean gradient in one forward+backward pass.

    ``use_ss_functional``: use `compute_ss_spread` (Stengel-Spaldin, PRB 73,
    075121, 2006) instead of the ordinary MV `compute_spread` -- a
    different, fully-differentiable forward Omega
    (`core.spread._ss_spread_from_M_tilde`), no hand-derived gradient
    needed either way. ``rguide`` is ignored in this mode (the SS Omega_D
    formula needs no guiding-centre/branch-cut treatment).

    ``use_pm_functional``: use `compute_pm_spread` (Pipek-Mezey) instead --
    Mmn/wb/bvecs/kb_idx/rguide are all unused in this mode (PM references
    only ``Aat``/``atom_index``), but every optimizer still threads them
    through unconditionally so `_final_result` can always report the
    ordinary MV spread/centres regardless of which functional drove the
    optimization.
    """
    U = U.detach().requires_grad_(True)
    if use_pm_functional:
        Omega = compute_pm_spread(U, Aat, atom_index)[0]
    elif use_ss_functional:
        Omega = compute_ss_spread(U, Mmn, wb, bvecs, kb_idx)[0]
    else:
        Omega = compute_spread(U, Mmn, wb, bvecs, kb_idx, rguide)[0]
    Omega.backward()
    return Omega.item(), U.grad.detach()


def _omega_at(
    U, Mmn, wb, bvecs, kb_idx, rguide=None, use_ss_functional: bool = False,
    use_pm_functional: bool = False, Aat: Tensor | None = None, atom_index: Tensor | None = None,
) -> float:
    """Omega only (no grad), for line-search trial evaluations -- SS/PM-aware
    counterpart of the bare `compute_spread(...)[0].item()` calls."""
    with torch.no_grad():
        if use_pm_functional:
            return compute_pm_spread(U, Aat, atom_index)[0].item()
        if use_ss_functional:
            return compute_ss_spread(U, Mmn, wb, bvecs, kb_idx)[0].item()
        return compute_spread(U, Mmn, wb, bvecs, kb_idx, rguide)[0].item()


def _final_result(
    U, Mmn, wb, bvecs, kb_idx, history, converged, rguide: Tensor | None = None,
    use_ss_functional: bool = False,
) -> SpreadResult:
    """
    Evaluate spread at U_final and build SpreadResult.

    Computing Omega at U_final (after the last retraction) ensures
    SpreadResult.Omega is always consistent with SpreadResult.U_final.
    global_minimize_spread relies on this for correct winner selection.

    ``use_ss_functional``: report the SS Omega_I/Omega_D/Omega_OD instead
    of the ordinary MV ones (Omega_I/Omega_OD are numerically identical
    either way, see `_ss_spread_from_M_tilde`). ``centres`` always comes
    from the ordinary MV formula (`compute_spread_from_M_tilde`), since the
    SS Omega_D formula doesn't produce a centre of its own.
    """
    with torch.no_grad():
        M_tilde = rotate_overlaps(U, Mmn, kb_idx)
        Omega, OI, OD, OOD, centres = compute_spread_from_M_tilde(M_tilde, wb, bvecs, rguide)
        if use_ss_functional:
            Omega, OI, OD, OOD = compute_ss_spread_from_M_tilde(M_tilde, wb, bvecs)
    return SpreadResult(
        U_final=U, centres=centres,
        Omega=Omega.item(), Omega_I=OI.item(),
        Omega_D=OD.item(), Omega_OD=OOD.item(),
        history=history, converged=converged, rguide=rguide,
    )


def _omega_and_grad_slwf(
    U: Tensor, Mmn, wb, bvecs, kb_idx, slwf_num: int, constrain: bool,
    target_centres: Tensor | None, lambda_: float, rguide: Tensor | None = None,
):
    """SLWF counterpart of `_omega_and_grad` -- see `compute_slwf_spread`."""
    U = U.detach().requires_grad_(True)
    Omega = compute_slwf_spread(U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_, rguide)[0]
    Omega.backward()
    return Omega.item(), U.grad.detach()


def _final_result_slwf(
    U, Mmn, wb, bvecs, kb_idx, slwf_num: int, constrain: bool,
    target_centres: Tensor | None, lambda_: float, history, converged,
    rguide: Tensor | None = None,
) -> SpreadResult:
    """SLWF counterpart of `_final_result` -- see `SpreadResult.Omega_IOD`/
    `Omega_nu` for how the SLWF-specific breakdown is stored."""
    with torch.no_grad():
        Omega, OIOD, OD, Onu, centres = compute_slwf_spread_from_M_tilde(
            rotate_overlaps(U, Mmn, kb_idx), wb, bvecs, slwf_num, constrain, target_centres, lambda_, rguide,
        )
    return SpreadResult(
        U_final=U, centres=centres,
        Omega=Omega.item(), Omega_I=OIOD.item(),
        Omega_D=OD.item(), Omega_OD=0.0,
        history=history, converged=converged, rguide=rguide,
        Omega_IOD=OIOD.item(), Omega_nu=Onu.item(),
    )


# ---------------------------------------------------------------------------
# Guiding centres (Wannier90's guiding_centres): periodic re-anchoring to
# prevent the MLWF "runaway centre" branch-cut pathology -- see
# spread.py::_guided_phase / refine_guiding_centres for the mechanism.
# ---------------------------------------------------------------------------

def _bootstrap_rguide(U: Tensor, Mmn, wb, bvecs, kb_idx) -> Tensor:
    """Initial guiding centres: naive centres, then one robust refinement pass."""
    with torch.no_grad():
        M_tilde = rotate_overlaps(U, Mmn, kb_idx)
        centres0 = compute_spread_from_M_tilde(M_tilde, wb, bvecs)[4]
        M_diag = torch.diagonal(M_tilde, dim1=-2, dim2=-1)
        return refine_guiding_centres(M_diag, bvecs, centres0)


def _refresh_rguide(U: Tensor, Mmn, wb, bvecs, kb_idx, rguide_prev: Tensor) -> Tensor:
    """Re-anchor the guiding centres to the current gauge (periodic refresh)."""
    with torch.no_grad():
        M_tilde = rotate_overlaps(U, Mmn, kb_idx)
        M_diag = torch.diagonal(M_tilde, dim1=-2, dim2=-1)
        return refine_guiding_centres(M_diag, bvecs, rguide_prev)


# ---------------------------------------------------------------------------
# Steepest descent with Armijo backtracking line search
# ---------------------------------------------------------------------------

def _minimize_sgd(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat           : Tensor | None = None,
    atom_index    : Tensor | None = None,
) -> SpreadResult:
    """
    Riemannian steepest descent with Armijo backtracking line search.

    The line search ensures Omega decreases at every step by halving the
    step size up to 50 times if needed.  This makes SGD robust regardless
    of the initial learning rate.
    """
    U         = U_init.clone()
    history   = []
    converged = False

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad(U, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        step = lr
        for _ in range(50):
            U_trial = _qr_retract(U, -step * G_riem)
            with torch.no_grad():
                omega_trial = _omega_at(U_trial, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
            if omega_trial < omega_val:
                break
            step *= 0.5
        U = U_trial.detach()

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result(U, Mmn, wb, bvecs, kb_idx, history, converged, rguide, use_ss_functional)



# ---------------------------------------------------------------------------
# Riemannian nonlinear conjugate gradients (Fletcher-Reeves), matching
# Wannier90's own wannierise.F90 algorithm.
# ---------------------------------------------------------------------------

def _real_inner(A: Tensor, B: Tensor) -> Tensor:
    """Re Tr[A^dagger B], summed over any leading (k) batch dims -- the real
    inner product two stacks of tangent-space matrices are combined with
    throughout (Fletcher-Reeves ratio, directional derivative, line search)."""
    return (A.conj() * B).real.sum()


def _minimize_cg(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    num_cg_steps  : int   = 5,
    cg_max_ratio  : float = 3.0,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat           : Tensor | None = None,
    atom_index    : Tensor | None = None,
) -> SpreadResult:
    """
    Riemannian nonlinear CG: Fletcher-Reeves direction + parabolic line
    search, matching wannier90's own minimizer.

      * direction d_t = -G_riem_t + beta_t * d_{t-1}, beta_t = ||G_t||^2 /
        ||G_{t-1}||^2 (Fletcher-Reeves), reset to plain steepest descent
        (beta=0) every `num_cg_steps` iterations or whenever beta exceeds
        `cg_max_ratio` (default 3) -- both guard against a bad CG direction
        far from the manifold's origin.
      * if d_t is not a descent direction (directional derivative >= 0),
        fall back to steepest descent and, failing that, reverse the
        direction.
      * step length: take one trial step of length `lr` along d_t, then fit
        the unique parabola through (0, Omega), slope (directional
        derivative) at 0, and (lr, Omega_trial); its minimum gives the
        optimal step analytically. Falls back to the trial step itself if
        the parabola is degenerate or predicts a maximum instead of a
        minimum along d_t.

    The direction/momentum carried between iterations lives in the tangent
    space at the previous U and is reused at the new U without parallel
    transport (the same trivialized-tangent-space approximation Adam uses).
    """
    U         = U_init.clone()
    history   = []
    converged = False

    d_prev    = None            # previous search direction (tangent-ish)
    gcnorm0   = 0.0              # ||G_riem||^2 from the previous iteration
    ncg       = 0                # steps since the last steepest-descent reset

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad(U, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)
        gcnorm1 = _real_inner(G_riem, G_riem).item()

        # Fletcher-Reeves coefficient, with Wannier90's own reset rules.
        if t == 0 or ncg >= num_cg_steps:
            beta = 0.0
            ncg = 0
        elif gcnorm0 > torch.finfo(torch.float64).eps:
            beta = gcnorm1 / gcnorm0
            if beta > cg_max_ratio:
                beta = 0.0
                ncg = 0
            else:
                ncg += 1
        else:
            beta = 0.0
            ncg = 0
        gcnorm0 = gcnorm1

        direction = -G_riem if beta == 0.0 else -G_riem + beta * d_prev
        doda0 = _real_inner(G_riem, direction).item()   # directional derivative

        if doda0 >= 0.0:
            # Not a descent direction (can happen since d_prev's momentum was
            # carried without transport) -- reset to steepest descent first...
            direction = -G_riem
            beta, ncg = 0.0, 0
            doda0 = _real_inner(G_riem, direction).item()
            if doda0 >= 0.0:
                # ...and if even -G_riem isn't downhill (should not happen;
                # -||G||^2 <= 0 always unless G is exactly zero), reverse it.
                direction = -direction
                doda0 = -doda0

        # Parabolic line search: trial step at length `lr`, fit through
        # (0, omega_val), slope doda0, and (lr, omega_trial).
        U_trial = _qr_retract(U, lr * direction)
        with torch.no_grad():
            omega_trial = _omega_at(U_trial, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)

        denom = omega_trial - omega_val
        c = (denom - doda0 * lr) / lr**2 if abs(lr) > 0 else 0.0
        use_quadratic = abs(c) > torch.finfo(torch.float64).eps
        if use_quadratic:
            alpha = -0.5 * doda0 / c
            if doda0 * alpha >= 0.0:
                use_quadratic = False   # parabola predicts an uphill point

        if use_quadratic:
            U = _qr_retract(U, alpha * direction).detach()
        else:
            U = U_trial.detach()   # Wannier90's own fallback: keep the trial step

        d_prev = direction

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result(U, Mmn, wb, bvecs, kb_idx, history, converged, rguide, use_ss_functional)


# ---------------------------------------------------------------------------
# Riemannian Adam (Bécigneul & Ganea, ICLR 2019)
# ---------------------------------------------------------------------------

def _minimize_adam(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    beta1         : float = 0.9,
    beta2         : float = 0.999,
    eps           : float = 1e-8,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat           : Tensor | None = None,
    atom_index    : Tensor | None = None,
) -> SpreadResult:
    """
    Riemannian Adam with QR retraction for complex unitary matrices.

    The moment tensors are maintained in the tangent space (complex-valued
    first moment, real-valued second moment tracking element-wise |grad|^2).
    After each moment update the update direction is retracted back to U(nw).

    This is a native implementation — geoopt is not used because it does not
    support complex128 and silently breaks the retraction.
    """
    U         = U_init.clone()
    history   = []
    converged = False

    m = torch.zeros_like(U)
    v = torch.zeros(U.shape, dtype=torch.float64, device=U.device)

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(1, n_iter + 1):
        if guiding_centres and t > 1 and (t - 1) % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad(U, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        m = beta1 * m + (1 - beta1) * G_riem
        v = beta2 * v + (1 - beta2) * G_riem.abs().pow(2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        U = _qr_retract(U, -lr * m_hat / (v_hat.sqrt() + eps))

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result(U, Mmn, wb, bvecs, kb_idx, history, converged, rguide, use_ss_functional)


# ---------------------------------------------------------------------------
# Riemannian limited-memory BFGS (two-loop recursion)
# ---------------------------------------------------------------------------

def _minimize_lbfgs(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    memory        : int   = 10,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat           : Tensor | None = None,
    atom_index    : Tensor | None = None,
) -> SpreadResult:
    """
    Riemannian L-BFGS: standard two-loop recursion for the search direction,
    using the same parabolic line search as `_minimize_cg`.

    Curvature pairs (s_i, y_i) are the ambient differences of consecutive
    retracted iterates and their Riemannian gradients, s_i = U_{i+1} - U_i,
    y_i = G_riem_{i+1} - G_riem_i -- like CG's conjugate direction, these are
    reused untransported at later iterates (no parallel transport). A pair
    is only kept if it satisfies the curvature condition s_i^T y_i > 0
    (otherwise the implied inverse-Hessian approximation would not be
    positive-definite); the most recent `memory` valid pairs are kept.

    If the resulting direction isn't downhill (can happen since the
    two-loop recursion's implicit Hessian approximation relies on the same
    untransported-history approximation as CG), reset to steepest descent
    and clear the stored history.
    """
    U         = U_init.clone()
    history   = []
    converged = False

    s_hist: list[Tensor] = []
    y_hist: list[Tensor] = []
    rho_hist: list[float] = []
    U_prev, G_riem_prev = None, None

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None
    eps = torch.finfo(torch.float64).eps

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad(U, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        if U_prev is not None:
            s = U.detach() - U_prev
            y = G_riem - G_riem_prev
            sy = _real_inner(s, y).item()
            if sy > eps:
                s_hist.append(s); y_hist.append(y); rho_hist.append(1.0 / sy)
                if len(s_hist) > memory:
                    s_hist.pop(0); y_hist.pop(0); rho_hist.pop(0)

        # two-loop recursion
        q = G_riem.clone()
        alphas = []
        for s_i, y_i, rho_i in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            alpha_i = rho_i * _real_inner(s_i, q).item()
            q = q - alpha_i * y_i
            alphas.append(alpha_i)
        alphas.reverse()

        if s_hist:
            gamma = _real_inner(s_hist[-1], y_hist[-1]).item() / max(
                _real_inner(y_hist[-1], y_hist[-1]).item(), eps)
        else:
            gamma = 1.0
        r = gamma * q
        for s_i, y_i, rho_i, alpha_i in zip(s_hist, y_hist, rho_hist, alphas):
            beta_i = rho_i * _real_inner(y_i, r).item()
            r = r + (alpha_i - beta_i) * s_i

        direction = -r
        doda0 = _real_inner(G_riem, direction).item()
        if doda0 >= 0.0:
            direction = -G_riem
            doda0 = _real_inner(G_riem, direction).item()
            s_hist.clear(); y_hist.clear(); rho_hist.clear()

        # Parabolic line search, identical to _minimize_cg's.
        U_trial = _qr_retract(U, lr * direction)
        with torch.no_grad():
            omega_trial = _omega_at(U_trial, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)

        denom = omega_trial - omega_val
        c = (denom - doda0 * lr) / lr**2 if abs(lr) > 0 else 0.0
        use_quadratic = abs(c) > eps
        if use_quadratic:
            alpha = -0.5 * doda0 / c
            if doda0 * alpha >= 0.0:
                use_quadratic = False

        if use_quadratic:
            U_new = _qr_retract(U, alpha * direction)
            with torch.no_grad():
                omega_new = _omega_at(U_new, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        else:
            U_new, omega_new = U_trial, omega_trial

        if omega_new >= omega_val:
            # Neither the quadratic-fit step nor the raw trial step actually
            # decreased Omega -- the two-loop recursion's Hessian scaling
            # (gamma) can occasionally produce a badly-scaled direction on a
            # messy landscape (unlike CG, where the direction is always a
            # bounded combination of unit-ish gradients). Fall back to an
            # Armijo-backtracked steepest-descent step (as `_minimize_sgd`
            # does) and discard the curvature-pair history, since it fed
            # into the direction that just failed.
            step = lr
            for _ in range(50):
                U_new = _qr_retract(U, -step * G_riem)
                with torch.no_grad():
                    omega_new = _omega_at(U_new, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
                if omega_new < omega_val:
                    break
                step *= 0.5
            s_hist.clear(); y_hist.clear(); rho_hist.clear()

        U_prev, G_riem_prev = U.detach(), G_riem.detach()
        U = U_new.detach()

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result(U, Mmn, wb, bvecs, kb_idx, history, converged, rguide, use_ss_functional)


# ---------------------------------------------------------------------------
# DIIS / Pulay mixing (subspace acceleration)
# ---------------------------------------------------------------------------

def _diis_coefficients(G_hist: list[Tensor]) -> Tensor | None:
    """
    Solve the standard DIIS linear system for mixing coefficients c that
    minimize ||sum_i c_i * e_i||^2 subject to sum_i c_i = 1, e_i = G_hist[i]:

        [ B   1 ] [c]   [0]
        [ 1^T 0 ] [l] = [1],   B_ij = <e_i, e_j> (real inner product)

    Returns None if the solve fails or the coefficients are unreasonably
    large (a hallmark of DIIS numerical instability from a near-singular B).
    """
    m = len(G_hist)
    device = G_hist[0].device
    B = torch.empty((m, m), dtype=torch.float64, device=device)
    for i in range(m):
        for j in range(i, m):
            B[i, j] = B[j, i] = _real_inner(G_hist[i], G_hist[j])

    A = torch.zeros((m + 1, m + 1), dtype=torch.float64, device=device)
    A[:m, :m] = B
    A[:m, m] = 1.0
    A[m, :m] = 1.0
    rhs = torch.zeros(m + 1, dtype=torch.float64, device=device)
    rhs[m] = 1.0

    try:
        sol = torch.linalg.solve(A, rhs)
    except RuntimeError:
        return None
    c = sol[:m]
    if not torch.isfinite(c).all() or c.abs().max() > 10.0:
        return None
    return c


def _minimize_diis(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    memory        : int   = 8,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat           : Tensor | None = None,
    atom_index    : Tensor | None = None,
) -> SpreadResult:
    """
    DIIS (Pulay-mixing) accelerated Riemannian steepest descent.

    Each iteration: store (U_t, G_riem_t) (up to `memory` most recent),
    solve for the DIIS coefficients minimizing the mixed gradient's norm
    (`_diis_coefficients`), and form the extrapolated ambient combination
    sum_i c_i U_i -- U(nw) isn't a vector space, so this is generally not
    unitary and is re-projected via the nearest-unitary (polar) retraction
    (SVD-unitarize, the same operation `analysis.z2`'s Wilson-loop links
    use). This extrapolated point is only accepted if it beats a safeguarded
    Armijo-backtracked steepest-descent step from the same U_t -- raw Pulay
    mixing has no intrinsic descent guarantee, unlike a line search.
    """
    U         = U_init.clone()
    history   = []
    converged = False

    U_hist: list[Tensor] = []
    G_hist: list[Tensor] = []

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad(U, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        U_hist.append(U.detach()); G_hist.append(G_riem.detach())
        if len(U_hist) > memory:
            U_hist.pop(0); G_hist.pop(0)

        # Armijo-backtracked steepest-descent candidate (same as _minimize_sgd).
        step = lr
        for _ in range(50):
            U_sd = _qr_retract(U, -step * G_riem)
            with torch.no_grad():
                omega_sd = _omega_at(U_sd, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
            if omega_sd < omega_val:
                break
            step *= 0.5

        U_new = U_sd
        if len(U_hist) >= 2:
            c = _diis_coefficients(G_hist)
            if c is not None:
                U_mix = sum((ci.item() * Ui for ci, Ui in zip(c, U_hist)))
                Us, _, Vhs = torch.linalg.svd(U_mix)
                U_diis = Us @ Vhs
                with torch.no_grad():
                    omega_diis = _omega_at(U_diis, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
                if omega_diis < omega_sd:
                    U_new = U_diis

        U = U_new.detach()

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result(U, Mmn, wb, bvecs, kb_idx, history, converged, rguide, use_ss_functional)


# ---------------------------------------------------------------------------
# Riemannian trust-region (RTR), Absil, Baker & Gallivan, Found. Comput.
# Math. 7, 303 (2007) -- the only optimizer here using second-order
# (curvature) information rather than gradient-only directions.
# ---------------------------------------------------------------------------

def _euclidean_hvp(
    U: Tensor, Mmn, wb, bvecs, kb_idx, eta: Tensor,
    rguide: Tensor | None = None, use_ss_functional: bool = False,
    use_pm_functional: bool = False, Aat: Tensor | None = None, atom_index: Tensor | None = None,
) -> Tensor:
    """
    Euclidean Hessian-vector product D(grad Omega)(U)[eta], via double
    backward through the same autodiff'd Omega(U) every other optimizer
    uses. Validated against a central finite difference of the Euclidean
    gradient (matches to the expected O(step^2) truncation error on real
    DFT overlap data) -- see the RTR entry in `tests/test_init_optim.py`.
    """
    U = U.detach().requires_grad_(True)
    if use_pm_functional:
        Omega = compute_pm_spread(U, Aat, atom_index)[0]
    elif use_ss_functional:
        Omega = compute_ss_spread(U, Mmn, wb, bvecs, kb_idx)[0]
    else:
        Omega = compute_spread(U, Mmn, wb, bvecs, kb_idx, rguide)[0]
    G_euc = torch.autograd.grad(Omega, U, create_graph=True)[0]
    Hv_euc = torch.autograd.grad(G_euc, U, grad_outputs=eta)[0]
    return Hv_euc.detach()


def _riemannian_hvp(
    U: Tensor, Mmn, wb, bvecs, kb_idx, eta: Tensor,
    rguide: Tensor | None = None, use_ss_functional: bool = False,
    use_pm_functional: bool = False, Aat: Tensor | None = None, atom_index: Tensor | None = None,
) -> Tensor:
    """
    Approximate Riemannian Hessian-vector product Hess Omega(U)[eta]: the
    standard projection-based approximation for embedded submanifolds
    (Absil, Mahony & Sepulchre, "Optimization Algorithms on Matrix
    Manifolds", Sec. 5.3) -- project the ambient directional derivative of
    the Euclidean gradient (`_euclidean_hvp`) onto the tangent space with
    the same operator `_riemannian_gradient` uses for the gradient itself.
    This omits the Weingarten-map curvature-correction term the exact
    Riemannian Hessian on U(nw) would need -- the same level of
    approximation as this module's QR retraction (not the true exponential
    map) and its untransported tangent-space state (CG/Adam/L-BFGS/DIIS),
    not a new approximation specific to RTR.
    """
    return _riemannian_gradient(U, _euclidean_hvp(U, Mmn, wb, bvecs, kb_idx, eta, rguide, use_ss_functional, use_pm_functional, Aat, atom_index))


def _boundary_tau(ee: float, ed: float, dd: float, trust_radius: float) -> float:
    """
    Positive root tau of ||eta + tau*d||^2 = trust_radius^2, i.e. of the
    quadratic dd*tau^2 + 2*ed*tau + (ee - trust_radius^2) = 0 -- how far
    truncated CG must travel along d to reach the trust-region boundary.
    """
    disc = max(ed * ed - dd * (ee - trust_radius ** 2), 0.0)
    return (-ed + disc ** 0.5) / dd


def _truncated_cg(
    G_riem: Tensor, hvp_fn, trust_radius: float, max_iter: int, kappa: float,
) -> tuple[Tensor, bool]:
    """
    Steihaug-Toint truncated CG: approximately solves the trust-region
    subproblem

        min_eta  <G_riem, eta> + 0.5 <Hess[eta], eta>   s.t.  ||eta|| <= trust_radius

    starting from eta=0 (so the model's own gradient there is G_riem).
    Stops early (returning the trust-region-boundary point along the
    current CG direction) on non-positive curvature or once the boundary
    is reached; otherwise runs an ordinary linear-CG on the model gradient
    until it drops to a `kappa` fraction of its initial norm or `max_iter`
    is hit.

    Returns (eta, hit_boundary) -- `hit_boundary` feeds the outer loop's
    trust-radius growth rule (Absil-Baker-Gallivan only grow the radius
    when the accepted step actually reached the current boundary).
    """
    eta = torch.zeros_like(G_riem)
    r = G_riem.clone()
    d = -r
    r0_norm = _real_inner(r, r).sqrt().item()
    if r0_norm < 1e-14:
        return eta, False

    rr = r0_norm ** 2
    for _ in range(max_iter):
        Hd = hvp_fn(d)
        dHd = _real_inner(d, Hd).item()

        if dHd <= 0.0:
            ee = _real_inner(eta, eta).item()
            ed = _real_inner(eta, d).item()
            dd = _real_inner(d, d).item()
            tau = _boundary_tau(ee, ed, dd, trust_radius)
            return eta + tau * d, True

        alpha = rr / dHd
        eta_next = eta + alpha * d

        if _real_inner(eta_next, eta_next).sqrt().item() >= trust_radius:
            ee = _real_inner(eta, eta).item()
            ed = _real_inner(eta, d).item()
            dd = _real_inner(d, d).item()
            tau = _boundary_tau(ee, ed, dd, trust_radius)
            return eta + tau * d, True

        r_next = r + alpha * Hd
        rr_next = _real_inner(r_next, r_next).item()
        if rr_next ** 0.5 <= r0_norm * kappa:
            return eta_next, False

        beta = rr_next / rr
        d = -r_next + beta * d
        eta, r, rr = eta_next, r_next, rr_next

    return eta, False


def _minimize_rtr(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    tcg_max_iter  : int   = 30,
    tcg_kappa     : float = 0.1,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat           : Tensor | None = None,
    atom_index    : Tensor | None = None,
) -> SpreadResult:
    """
    Riemannian trust-region method (RTR). Each iteration: build the local
    quadratic model from the Riemannian gradient and (approximate)
    Hessian-vector product (`_riemannian_hvp`), approximately minimize it
    within a trust region of radius `Delta` via truncated CG
    (`_truncated_cg`), retract the candidate step, and accept/reject +
    grow/shrink `Delta` from the ratio of actual to predicted decrease
    (standard Absil-Baker-Gallivan rule: shrink below rho=0.25, grow above
    rho=0.75 when the step reached the trust-region boundary, accept
    whenever rho > 0.1).

    `lr` sets the initial and maximum trust-region radius
    (Delta_0 = lr, Delta_max = 10*lr) -- an analogous "how big a step to
    try" role to the other optimizers' `lr`.

    The only optimizer here that uses curvature (second-order) information;
    substantially more expensive per outer iteration (up to `tcg_max_iter`
    Hessian-vector products, each its own double-backward pass) than the
    gradient-only methods.
    """
    U         = U_init.clone()
    history   = []
    converged = False

    Delta     = lr
    Delta_max = 10.0 * lr

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad(U, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        def hvp_fn(v, U=U, rguide=rguide):
            return _riemannian_hvp(U, Mmn, wb, bvecs, kb_idx, v, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)

        eta, hit_boundary = _truncated_cg(G_riem, hvp_fn, Delta, tcg_max_iter, tcg_kappa)

        H_eta = hvp_fn(eta)
        pred_decrease = -(_real_inner(G_riem, eta).item() + 0.5 * _real_inner(H_eta, eta).item())

        U_trial = _qr_retract(U, eta)
        with torch.no_grad():
            omega_trial = _omega_at(U_trial, Mmn, wb, bvecs, kb_idx, rguide, use_ss_functional, use_pm_functional, Aat, atom_index)
        actual_decrease = omega_val - omega_trial

        rho = actual_decrease / pred_decrease if abs(pred_decrease) > 1e-14 else -1.0

        if rho < 0.25:
            Delta = 0.25 * Delta
        elif rho > 0.75 and hit_boundary:
            Delta = min(2.0 * Delta, Delta_max)

        if rho > 0.1:
            U = U_trial.detach()
        # else reject the step -- U (and Delta, already shrunk above) carry
        # over to the next iteration, which retries from the same point.

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result(U, Mmn, wb, bvecs, kb_idx, history, converged, rguide, use_ss_functional)


# ---------------------------------------------------------------------------
# Selectively localized Wannier functions (SLWF, tutorial26, `slwf_num`/
# `slwf_constrain`): only the first `slwf_num` of `nw` Wannier functions are
# localized (core.spread.compute_slwf_spread) -- unlike site_symmetry, the
# optimization VARIABLE is still the full (nk, nw, nw) unitary gauge (no
# irreducible-k restriction), only the Omega FUNCTION differs, so each
# variant below is otherwise identical to its plain sibling.
# ---------------------------------------------------------------------------

def _minimize_sgd_slwf(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    slwf_num      : int,
    constrain     : bool,
    target_centres: Tensor | None,
    lambda_       : float,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
) -> SpreadResult:
    """SLWF counterpart of `_minimize_sgd` -- see the SLWF section docstring."""
    U         = U_init.clone()
    history   = []
    converged = False

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad_slwf(
            U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_, rguide)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        step = lr
        for _ in range(50):
            U_trial = _qr_retract(U, -step * G_riem)
            with torch.no_grad():
                omega_trial = compute_slwf_spread(
                    U_trial, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_, rguide)[0].item()
            if omega_trial < omega_val:
                break
            step *= 0.5
        U = U_trial.detach()

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result_slwf(U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_,
                              history, converged, rguide)


def _minimize_cg_slwf(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    slwf_num      : int,
    constrain     : bool,
    target_centres: Tensor | None,
    lambda_       : float,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    num_cg_steps  : int   = 5,
    cg_max_ratio  : float = 3.0,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
) -> SpreadResult:
    """SLWF counterpart of `_minimize_cg` -- see the SLWF section docstring."""
    U         = U_init.clone()
    history   = []
    converged = False

    d_prev    = None
    gcnorm0   = 0.0
    ncg       = 0

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad_slwf(
            U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_, rguide)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)
        gcnorm1 = _real_inner(G_riem, G_riem).item()

        if t == 0 or ncg >= num_cg_steps:
            beta = 0.0
            ncg = 0
        elif gcnorm0 > torch.finfo(torch.float64).eps:
            beta = gcnorm1 / gcnorm0
            if beta > cg_max_ratio:
                beta = 0.0
                ncg = 0
            else:
                ncg += 1
        else:
            beta = 0.0
            ncg = 0
        gcnorm0 = gcnorm1

        direction = -G_riem if beta == 0.0 else -G_riem + beta * d_prev
        doda0 = _real_inner(G_riem, direction).item()

        if doda0 >= 0.0:
            direction = -G_riem
            beta, ncg = 0.0, 0
            doda0 = _real_inner(G_riem, direction).item()
            if doda0 >= 0.0:
                direction = -direction
                doda0 = -doda0

        U_trial = _qr_retract(U, lr * direction)
        with torch.no_grad():
            omega_trial = compute_slwf_spread(
                U_trial, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_, rguide)[0].item()

        denom = omega_trial - omega_val
        c = (denom - doda0 * lr) / lr**2 if abs(lr) > 0 else 0.0
        use_quadratic = abs(c) > torch.finfo(torch.float64).eps
        if use_quadratic:
            alpha = -0.5 * doda0 / c
            if doda0 * alpha >= 0.0:
                use_quadratic = False

        if use_quadratic:
            U = _qr_retract(U, alpha * direction).detach()
        else:
            U = U_trial.detach()

        d_prev = direction

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result_slwf(U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_,
                              history, converged, rguide)


def _minimize_adam_slwf(
    U_init        : Tensor,
    Mmn           : Tensor,
    wb            : Tensor,
    bvecs         : Tensor,
    kb_idx        : Tensor,
    slwf_num      : int,
    constrain     : bool,
    target_centres: Tensor | None,
    lambda_       : float,
    lr            : float,
    n_iter        : int,
    conv_tol      : float,
    conv_window   : int,
    beta1         : float = 0.9,
    beta2         : float = 0.999,
    eps           : float = 1e-8,
    guiding_centres: bool = False,
    guide_refresh : int   = 10,
) -> SpreadResult:
    """SLWF counterpart of `_minimize_adam` -- see the SLWF section docstring."""
    U         = U_init.clone()
    history   = []
    converged = False

    m = torch.zeros_like(U)
    v = torch.zeros(U.shape, dtype=torch.float64, device=U.device)

    rguide = _bootstrap_rguide(U, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(1, n_iter + 1):
        if guiding_centres and t > 1 and (t - 1) % guide_refresh == 0:
            rguide = _refresh_rguide(U, Mmn, wb, bvecs, kb_idx, rguide)

        omega_val, G_euc = _omega_and_grad_slwf(
            U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_, rguide)
        history.append(omega_val)
        G_riem = _riemannian_gradient(U, G_euc)

        m = beta1 * m + (1 - beta1) * G_riem
        v = beta2 * v + (1 - beta2) * G_riem.abs().pow(2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        U = _qr_retract(U, -lr * m_hat / (v_hat.sqrt() + eps))

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    return _final_result_slwf(U, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_,
                              history, converged, rguide)


def minimize_spread_slwf(
    U_init         : Tensor,
    Mmn            : Tensor,
    wb             : Tensor,
    bvecs          : Tensor,
    kb_idx         : Tensor,
    slwf_num       : int,
    constrain      : bool  = False,
    target_centres : Tensor | None = None,
    lambda_        : float = 1.0,
    optimizer      : str   = "cg",
    lr             : float = 3e-2,
    n_iter         : int   = 1000,
    conv_tol       : float = 1e-10,
    conv_window    : int   = 5,
    guiding_centres: bool  = False,
    guide_refresh  : int   = 10,
) -> SpreadResult:
    """
    Minimize the SLWF spread functional (Wannier90 tutorial26, `slwf_num`/
    `slwf_constrain`/`slwf_lambda`) -- see `core.spread.compute_slwf_spread`
    for the formula and `minimize_spread`'s docstring for the shared
    optimizer/convergence arguments (identical meaning here).

    Args (beyond `minimize_spread`'s):
      slwf_num      : J' <= nw, number of objective Wannier functions
      constrain     : whether to fix the OWF centres (`slwf_constrain`)
      target_centres: (J', 3) Bohr, required when constrain=True
      lambda_       : Lagrange multiplier (`slwf_lambda`, default 1.0);
                      ignored when constrain=False

    Returns:
      SpreadResult with `Omega_IOD`/`Omega_nu` populated (see
      `SpreadResult`'s docstring for how the SLWF breakdown maps onto its
      shared `Omega_I`/`Omega_D`/`Omega_OD` fields).
    """
    if optimizer == "sgd":
        return _minimize_sgd_slwf(
            U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_,
            lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh,
        )
    elif optimizer == "cg":
        return _minimize_cg_slwf(
            U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_,
            lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh,
        )
    elif optimizer == "adam":
        return _minimize_adam_slwf(
            U_init, Mmn, wb, bvecs, kb_idx, slwf_num, constrain, target_centres, lambda_,
            lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh,
        )
    else:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. Choose 'sgd', 'cg' or 'adam'."
        )


# ---------------------------------------------------------------------------
# Symmetry-adapted spread minimization (site_symmetry, tutorial21): only the
# irreducible k-wedge is an independent variable; every other k's U is
# derived via the crystal's point-group action (core.sitesym).
# ---------------------------------------------------------------------------

def _symmetrize_and_broadcast(
    U_irr: Tensor, sitesym: SiteSymmetry, d_left: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Re-symmetrize U at every irreducible k under its own little-group
    stabilizer, then broadcast to the full mesh. Shared by every
    symmetrized optimizer's retraction step.

    U's left index lives in whatever space U's rows actually represent:
    `sitesym.d_matrix_wann` (default) after genuine disentanglement (both
    of U's indices are Wannier-gauge), but `sitesym.d_matrix_band` for
    isolated bands (nb == nw, `core.disentangle.disentangle` never ran --
    U's left index is still the raw candidate-band manifold, a different
    representation from the Wannier-gauge one on the right, even though
    same-dimensional). The correct covariance relation is
    U(Rk) = D_band(R,k) U(k) D_wann(R,k)^dagger for isolated bands (not
    D_wann on both sides) -- see `core.pipeline.wannierize`'s docstring.

    Returns (U_irr_symmetrized, U_full).
    """
    if d_left is None:
        d_left = sitesym.d_matrix_wann
    d_wann = sitesym.d_matrix_wann
    U_irr = torch.stack([
        symmetrize_u_irr(U_irr[ir], sitesym, ir, d_left=d_left)
        for ir in range(sitesym.nkptirr)
    ])
    U_full = broadcast_matrix(U_irr, d_left, d_wann, sitesym.kptsym, sitesym.ir2ik, sitesym.num_kpts)
    return U_irr, U_full


def _minimize_sgd_symmetrized(
    U_irr_init: Tensor, sitesym: SiteSymmetry,
    Mmn: Tensor, wb: Tensor, bvecs: Tensor, kb_idx: Tensor,
    lr: float, n_iter: int, conv_tol: float, conv_window: int,
    guiding_centres: bool = False, guide_refresh: int = 10,
    d_left: Tensor | None = None,
) -> SpreadResult:
    """
    Symmetry-adapted Riemannian steepest descent (`site_symmetry = .true.`):
    the same Armijo-backtracking steepest descent as `_minimize_sgd`, but
    the only independent variable is U at the irreducible k-points
    (`sitesym.ir2ik`) -- every other k's U is slaved to an irreducible
    representative via the point-group action.

    Each iteration: broadcast U_irr to the full mesh (needed since Mmn
    couples k to its own neighbours k+b, which may not be irreducible),
    evaluate Omega/gradient over the full mesh as usual, reduce the
    Riemannian gradient back to the irreducible wedge
    (`core.sitesym.reduce_gradient_to_irr`), take a steepest-descent step on
    U_irr only, then re-symmetrize under the little-group stabilizer
    (`core.sitesym.symmetrize_u_irr`) before the next broadcast -- this
    keeps the whole-mesh U point-group-covariant at every iteration.

    `d_left` : see `_symmetrize_and_broadcast` -- `d_matrix_wann` (default)
    after genuine disentanglement, `d_matrix_band` for isolated bands.

    `guiding_centres` works as in `_minimize_sgd`: the `nw` real-space
    Wannier-centre positions are bootstrapped/refreshed from the broadcast
    full-mesh U/M_tilde; only the optimization step is restricted to the
    irreducible wedge.
    """
    U_irr = U_irr_init.clone()
    nk = sitesym.num_kpts
    history = []
    converged = False

    _, U_full0 = _symmetrize_and_broadcast(U_irr, sitesym, d_left)
    rguide = _bootstrap_rguide(U_full0, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        _, U_full = _symmetrize_and_broadcast(U_irr, sitesym, d_left)
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U_full, Mmn, wb, bvecs, kb_idx, rguide)
        omega_val, G_euc_full = _omega_and_grad(U_full, Mmn, wb, bvecs, kb_idx, rguide)
        history.append(omega_val)

        G_riem_full = _riemannian_gradient(U_full, G_euc_full)
        G_riem_irr = reduce_gradient_to_irr(G_riem_full, sitesym, d_left)

        step = lr
        for _ in range(50):
            U_irr_trial, U_full_trial = _symmetrize_and_broadcast(
                _qr_retract(U_irr, -step * G_riem_irr), sitesym, d_left)
            with torch.no_grad():
                omega_trial = compute_spread(U_full_trial, Mmn, wb, bvecs, kb_idx, rguide)[0].item()
            if omega_trial < omega_val:
                break
            step *= 0.5
        U_irr = U_irr_trial.detach()

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    _, U_full = _symmetrize_and_broadcast(U_irr.detach(), sitesym, d_left)
    return _final_result(U_full, Mmn, wb, bvecs, kb_idx, history, converged, rguide)


def _minimize_cg_symmetrized(
    U_irr_init: Tensor, sitesym: SiteSymmetry,
    Mmn: Tensor, wb: Tensor, bvecs: Tensor, kb_idx: Tensor,
    lr: float, n_iter: int, conv_tol: float, conv_window: int,
    num_cg_steps: int = 5, cg_max_ratio: float = 3.0,
    guiding_centres: bool = False, guide_refresh: int = 10,
    d_left: Tensor | None = None,
) -> SpreadResult:
    """
    Symmetry-adapted Riemannian CG: `_minimize_cg`'s Fletcher-Reeves
    direction + parabolic line search, restricted to the irreducible
    k-wedge -- see `_minimize_sgd_symmetrized` for the broadcast/reduce
    pattern (`d_left`, `guiding_centres` handling) shared by every
    symmetrized optimizer. The conjugate direction `d_prev` is carried on
    the irreducible-k tensor only.
    """
    U_irr = U_irr_init.clone()
    history, converged = [], False
    d_prev, gcnorm0, ncg = None, 0.0, 0

    _, U_full0 = _symmetrize_and_broadcast(U_irr, sitesym, d_left)
    rguide = _bootstrap_rguide(U_full0, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(n_iter):
        _, U_full = _symmetrize_and_broadcast(U_irr, sitesym, d_left)
        if guiding_centres and t > 0 and t % guide_refresh == 0:
            rguide = _refresh_rguide(U_full, Mmn, wb, bvecs, kb_idx, rguide)
        omega_val, G_euc_full = _omega_and_grad(U_full, Mmn, wb, bvecs, kb_idx, rguide)
        history.append(omega_val)
        G_riem_full = _riemannian_gradient(U_full, G_euc_full)
        G_riem = reduce_gradient_to_irr(G_riem_full, sitesym, d_left)
        gcnorm1 = _real_inner(G_riem, G_riem).item()

        if t == 0 or ncg >= num_cg_steps:
            beta = 0.0
            ncg = 0
        elif gcnorm0 > torch.finfo(torch.float64).eps:
            beta = gcnorm1 / gcnorm0
            if beta > cg_max_ratio:
                beta = 0.0
                ncg = 0
            else:
                ncg += 1
        else:
            beta = 0.0
            ncg = 0
        gcnorm0 = gcnorm1

        direction = -G_riem if beta == 0.0 else -G_riem + beta * d_prev
        doda0 = _real_inner(G_riem, direction).item()

        if doda0 >= 0.0:
            direction = -G_riem
            beta, ncg = 0.0, 0
            doda0 = _real_inner(G_riem, direction).item()
            if doda0 >= 0.0:
                direction = -direction
                doda0 = -doda0

        U_irr_trial, U_full_trial = _symmetrize_and_broadcast(_qr_retract(U_irr, lr * direction), sitesym, d_left)
        with torch.no_grad():
            omega_trial = compute_spread(U_full_trial, Mmn, wb, bvecs, kb_idx, rguide)[0].item()

        denom = omega_trial - omega_val
        c = (denom - doda0 * lr) / lr**2 if abs(lr) > 0 else 0.0
        use_quadratic = abs(c) > torch.finfo(torch.float64).eps
        if use_quadratic:
            alpha = -0.5 * doda0 / c
            if doda0 * alpha >= 0.0:
                use_quadratic = False

        if use_quadratic:
            U_irr, _ = _symmetrize_and_broadcast(_qr_retract(U_irr, alpha * direction), sitesym, d_left)
            U_irr = U_irr.detach()
        else:
            U_irr = U_irr_trial.detach()

        d_prev = direction

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    _, U_full = _symmetrize_and_broadcast(U_irr.detach(), sitesym, d_left)
    return _final_result(U_full, Mmn, wb, bvecs, kb_idx, history, converged, rguide)


def _minimize_adam_symmetrized(
    U_irr_init: Tensor, sitesym: SiteSymmetry,
    Mmn: Tensor, wb: Tensor, bvecs: Tensor, kb_idx: Tensor,
    lr: float, n_iter: int, conv_tol: float, conv_window: int,
    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
    guiding_centres: bool = False, guide_refresh: int = 10,
    d_left: Tensor | None = None,
) -> SpreadResult:
    """
    Symmetry-adapted Riemannian Adam: exactly `_minimize_adam`'s moment
    update, restricted to the irreducible k-wedge -- see
    `_minimize_sgd_symmetrized` for the broadcast/reduce pattern (`d_left`)
    shared by every symmetrized optimizer. Both moment tensors live on the
    irreducible-k tensor only.
    """
    U_irr = U_irr_init.clone()
    history, converged = [], False
    m = torch.zeros_like(U_irr)
    v = torch.zeros(U_irr.shape, dtype=torch.float64, device=U_irr.device)

    _, U_full0 = _symmetrize_and_broadcast(U_irr, sitesym, d_left)
    rguide = _bootstrap_rguide(U_full0, Mmn, wb, bvecs, kb_idx) if guiding_centres else None

    for t in range(1, n_iter + 1):
        _, U_full = _symmetrize_and_broadcast(U_irr, sitesym, d_left)
        if guiding_centres and t > 1 and (t - 1) % guide_refresh == 0:
            rguide = _refresh_rguide(U_full, Mmn, wb, bvecs, kb_idx, rguide)
        omega_val, G_euc_full = _omega_and_grad(U_full, Mmn, wb, bvecs, kb_idx, rguide)
        history.append(omega_val)
        G_riem_full = _riemannian_gradient(U_full, G_euc_full)
        G_riem = reduce_gradient_to_irr(G_riem_full, sitesym, d_left)

        m = beta1 * m + (1 - beta1) * G_riem
        v = beta2 * v + (1 - beta2) * G_riem.abs().pow(2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        U_irr, _ = _symmetrize_and_broadcast(
            _qr_retract(U_irr, -lr * m_hat / (v_hat.sqrt() + eps)), sitesym, d_left)
        U_irr = U_irr.detach()

        if len(history) >= conv_window and abs(history[-conv_window] - history[-1]) < conv_tol:
            converged = True
            break

    _, U_full = _symmetrize_and_broadcast(U_irr.detach(), sitesym, d_left)
    return _final_result(U_full, Mmn, wb, bvecs, kb_idx, history, converged, rguide)


def minimize_spread_symmetrized(
    U_irr_init: Tensor,
    sitesym:    SiteSymmetry,
    Mmn:        Tensor,
    wb:         Tensor,
    bvecs:      Tensor,
    kb_idx:     Tensor,
    optimizer:  str   = "cg",
    lr:         float = 3e-2,
    n_iter:     int   = 1000,
    conv_tol:   float = 1e-10,
    conv_window: int  = 5,
    guiding_centres: bool = False,
    guide_refresh:   int  = 10,
    d_left: Tensor | None = None,
) -> SpreadResult:
    """
    Symmetry-adapted spread minimization (`site_symmetry = .true.`): the
    only independent variable is U at the irreducible k-points
    (`sitesym.ir2ik`); every other k's U is derived from an irreducible
    representative via the point-group action, not independently
    optimized.

    `optimizer` selects the same three algorithms `minimize_spread` does
    ("sgd" | "cg" | "adam"), each restricted to the irreducible wedge with
    the momentum/conjugate-direction state also living there only; see
    `_minimize_sgd_symmetrized` for the shared broadcast/re-symmetrize
    pattern.

    `d_left` : which representation U's left index transforms under --
    `sitesym.d_matrix_wann` (default, both of U's indices are Wannier-gauge)
    after genuine disentanglement, but `sitesym.d_matrix_band` for isolated
    bands (`core.pipeline.wannierize`'s `nb == nw` branch, where U's left
    index is still the raw candidate-band manifold, a different
    representation from the Wannier-gauge one even though same-dimensional).
    `core.pipeline.wannierize` sets this correctly for you; only pass it
    explicitly if calling this function directly with an isolated-bands
    `U_irr_init`.

    `guiding_centres`, `guide_refresh` : wannier90-style guiding centres
    (see `minimize_spread`'s docstring for the mechanism). The `nw` guiding
    centres are real-space Wannier-centre positions (not a per-k quantity),
    bootstrapped/refreshed from the broadcast full mesh; only the
    optimization step itself is restricted to the irreducible wedge.

    Default is "cg" (matches wannier90's own minimizer; Adam can get stuck
    in a bad local minimum on some systems).

    Args:
      U_irr_init : (nkptirr, nw, nw) complex, initial gauge AT THE
                   IRREDUCIBLE k-points only
      sitesym    : SiteSymmetry
      Mmn, wb, bvecs, kb_idx : as `minimize_spread`, on the FULL mesh
      optimizer, lr, n_iter, conv_tol, conv_window : as `minimize_spread`

    Returns:
      SpreadResult with U_final on the FULL mesh (nk, nw, nw) -- broadcast
      from the converged irreducible-k solution, so downstream code
      (H(R), spreads, …) needs no special-casing.
    """
    if optimizer == "sgd":
        return _minimize_sgd_symmetrized(
            U_irr_init, sitesym, Mmn, wb, bvecs, kb_idx,
            lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh, d_left=d_left,
        )
    elif optimizer == "cg":
        return _minimize_cg_symmetrized(
            U_irr_init, sitesym, Mmn, wb, bvecs, kb_idx,
            lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh, d_left=d_left,
        )
    elif optimizer == "adam":
        return _minimize_adam_symmetrized(
            U_irr_init, sitesym, Mmn, wb, bvecs, kb_idx,
            lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh, d_left=d_left,
        )
    else:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. Choose 'sgd', 'cg' or 'adam'."
        )

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def minimize_spread(
    U_init         : Tensor,
    Mmn            : Tensor,
    wb             : Tensor,
    bvecs          : Tensor,
    kb_idx         : Tensor,
    optimizer      : str   = "cg",
    lr             : float = 3e-2,
    n_iter         : int   = 1000,
    conv_tol       : float = 1e-10,
    conv_window    : int   = 5,
    guiding_centres: bool  = False,
    guide_refresh  : int   = 10,
    use_ss_functional: bool = False,
    use_pm_functional: bool = False,
    Aat            : Tensor | None = None,
    atom_index     : Tensor | None = None,
) -> SpreadResult:
    """
    Minimize the spread functional Ω(U) over unitary matrices U(k) ∈ U(nw).

    Default is "cg" (Riemannian conjugate-gradient, Fletcher-Reeves) --
    wannier90's own minimizer; tends to converge faster and to a
    more-symmetric gauge than Adam.

    Args:
      U_init     : (nk, nw, nw) complex  initial gauge (e.g. from svd_init)
      Mmn        : (nk, nnb, nw, nw) complex  overlap matrices
      wb         : (nnb,)  real  shell weights
      bvecs      : (nk, nnb, 3) real  Cartesian b-vectors
      kb_idx     : (nk, nnb) long  neighbour k-index table
      optimizer  : "sgd" | "cg" | "adam" | "lbfgs" | "diis" | "rtr"
      lr         : learning rate (Adam/SGD) or line-search trial step (CG)
      n_iter     : maximum number of iterations
      conv_tol   : convergence threshold: |Ω[t] - Ω[t-conv_window]| < tol
      conv_window: window size for convergence check
      guiding_centres: enable wannier90-style guiding centres (periodic
                   branch-consistent re-anchoring of the Ω_D phase; see
                   spread.py::_guided_phase). Off by default -- only needed
                   for systems prone to the MLWF "runaway centre" pathology
                   (small/few-k periodic metals).
      guide_refresh: iterations between guiding-centre refreshes when enabled.
      use_ss_functional: use the Stengel-Spaldin alternative localization
                   functional (`use_ss_functional`) instead of the ordinary
                   Marzari-Vanderbilt one -- see
                   `core.spread._ss_spread_from_M_tilde`. `guiding_centres`
                   is ignored when this is set (the SS Omega_D formula
                   needs no branch-cut treatment); `centres` in the
                   returned `SpreadResult` still comes from the ordinary
                   MV formula.
      use_pm_functional: use the Pipek-Mezey atomic-locality functional
                   instead (`core.spread.compute_pm_spread`) -- requires
                   `Aat`/`atom_index`. Mutually exclusive with
                   `use_ss_functional`; `guiding_centres` is ignored (PM has
                   no branch-cut logarithm at all). `Omega`/`Omega_I`/
                   `Omega_D`/`Omega_OD`/`centres` in the returned
                   `SpreadResult` still come from the ordinary MV formula
                   evaluated at the PM-optimized `U_final` -- PM has no
                   spread decomposition of its own, so this is how to see
                   what localizing for atomic character cost (or gained)
                   in ordinary spatial spread.
      Aat        : (nk, nw, n_orbitals) complex, required when
                   `use_pm_functional=True` -- atomic pseudo-orbital
                   overlaps already projected into the nw-dimensional
                   Wannier-gauge subspace (see `compute_pm_spread`).
      atom_index : (n_orbitals,) long, required when `use_pm_functional=True`
                   -- 0-based atom owning each `Aat` column (see
                   `interfaces.quantum_espresso.upf.atom_proj_column_atoms`).

    Returns:
      SpreadResult with final U, centres, spread components, history.
      SpreadResult.Omega is always evaluated at U_final (post-retraction).
    """
    kwargs = dict(
        lr=lr, n_iter=n_iter, conv_tol=conv_tol, conv_window=conv_window,
        guiding_centres=guiding_centres, guide_refresh=guide_refresh,
        use_ss_functional=use_ss_functional,
        use_pm_functional=use_pm_functional, Aat=Aat, atom_index=atom_index,
    )
    if optimizer == "sgd":
        return _minimize_sgd(U_init, Mmn, wb, bvecs, kb_idx, **kwargs)
    elif optimizer == "cg":
        return _minimize_cg(U_init, Mmn, wb, bvecs, kb_idx, **kwargs)
    elif optimizer == "adam":
        return _minimize_adam(U_init, Mmn, wb, bvecs, kb_idx, **kwargs)
    elif optimizer == "lbfgs":
        return _minimize_lbfgs(U_init, Mmn, wb, bvecs, kb_idx, **kwargs)
    elif optimizer == "diis":
        return _minimize_diis(U_init, Mmn, wb, bvecs, kb_idx, **kwargs)
    elif optimizer == "rtr":
        return _minimize_rtr(U_init, Mmn, wb, bvecs, kb_idx, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}. Choose 'sgd', 'cg', 'adam', 'lbfgs', 'diis' or 'rtr'."
        )
