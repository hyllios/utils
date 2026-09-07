"""
Global minimization of the Wannier spread functional.

Local optimizers (SGD, CG, Adam) converge to the nearest local minimum, which for
entangled or low-symmetry systems may not be the global one.  This module
provides two strategies on top of the local optimizer:

  random_restarts
    Run the local optimizer multiple times with independent random initial
    gauges, keep the result with the lowest final Omega.  The first run
    always uses the SVD-initialized U_init (the W90 standard starting point).
    Restarts are run in parallel using a thread pool (each restart has its
    own U tensor; shared Mmn/wb/bvecs are read-only).

  basin_hopping
    After each local convergence, apply a random unitary perturbation and
    re-minimize.  Accept the jump if the new basin is lower.  Combined with
    random restarts this covers a wide region of the landscape.

Both strategies are unified in `global_minimize_spread`.
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass

import torch
from torch import Tensor

from .init import random_unitary
from .optim import minimize_spread, SpreadResult, _qr_retract


# ---------------------------------------------------------------------------
# Perturbation for basin hopping
# ---------------------------------------------------------------------------

def _perturb_U(U: Tensor, strength: float, generator: torch.Generator) -> Tensor:
    """
    Add a random skew-Hermitian perturbation to U and retract to U(nw).

    The noise is drawn in the Lie algebra direction (skew-Hermitian tangent
    vector), so the perturbation is a local random rotation on the manifold.
    `strength` is the Frobenius norm of the perturbation direction.
    """
    nk, nw = U.shape[0], U.shape[1]
    dtype  = U.dtype
    real_t = torch.float64 if dtype == torch.complex128 else torch.float32

    noise_r = torch.randn(nk, nw, nw, dtype=real_t, generator=generator)
    noise_i = torch.randn(nk, nw, nw, dtype=real_t, generator=generator)
    noise   = torch.complex(noise_r, noise_i)

    S = (noise - noise.conj().transpose(-1, -2)) / 2
    S = S * (strength / (S.norm() / (nk * nw) ** 0.5 + 1e-12))
    return _qr_retract(U, torch.matmul(U, S))


# ---------------------------------------------------------------------------
# Per-restart configuration (replaces fragile positional tuple)
# ---------------------------------------------------------------------------

@dataclass
class _RestartConfig:
    U_init:       Tensor
    Mmn:          Tensor
    wb:           Tensor
    bvecs:        Tensor
    kb_idx:       Tensor
    i_restart:    int
    seed:         int
    n_hops:       int
    hop_strength: float
    optimizer:    str
    lr:           float
    n_iter:       int
    conv_tol:     float
    conv_window:  int
    guiding_centres: bool = False
    guide_refresh:   int  = 10
    use_ss_functional: bool = False


def _restart_worker(cfg: _RestartConfig) -> SpreadResult:
    """
    Run one complete restart: local minimization followed by basin hops.

    Each restart is independent — its own U clone, its own Generator seeds,
    read-only access to the shared Mmn/wb/bvecs tensors.  Safe to run
    concurrently from a ThreadPoolExecutor.
    """
    nk, nw = cfg.U_init.shape[:2]

    if cfg.i_restart == 0:
        U_start = cfg.U_init.clone()
    else:
        g = torch.Generator().manual_seed(cfg.seed + cfg.i_restart * 1000)
        U_start = random_unitary(nk, nw, generator=g).to(dtype=cfg.U_init.dtype)

    result = minimize_spread(
        U_start, cfg.Mmn, cfg.wb, cfg.bvecs, cfg.kb_idx,
        optimizer=cfg.optimizer, lr=cfg.lr,
        n_iter=cfg.n_iter, conv_tol=cfg.conv_tol, conv_window=cfg.conv_window,
        guiding_centres=cfg.guiding_centres, guide_refresh=cfg.guide_refresh,
        use_ss_functional=cfg.use_ss_functional,
    )

    for i_hop in range(cfg.n_hops):
        g_hop = torch.Generator().manual_seed(cfg.seed + cfg.i_restart * 1000 + i_hop + 1)
        U_perturbed = _perturb_U(result.U_final, cfg.hop_strength, g_hop)
        hop_result  = minimize_spread(
            U_perturbed, cfg.Mmn, cfg.wb, cfg.bvecs, cfg.kb_idx,
            optimizer=cfg.optimizer, lr=cfg.lr,
            n_iter=cfg.n_iter, conv_tol=cfg.conv_tol, conv_window=cfg.conv_window,
            guiding_centres=cfg.guiding_centres, guide_refresh=cfg.guide_refresh,
            use_ss_functional=cfg.use_ss_functional,
        )
        if hop_result.Omega < result.Omega:
            result = hop_result

    return result


# ---------------------------------------------------------------------------
# Global minimizer
# ---------------------------------------------------------------------------

def global_minimize_spread(
    U_init:       Tensor,
    Mmn:          Tensor,
    wb:           Tensor,
    bvecs:        Tensor,
    kb_idx:       Tensor,
    n_restarts:   int         = 5,
    n_hops:       int         = 0,
    hop_strength: float       = 0.3,
    optimizer:    str         = "cg",
    lr:           float       = 3e-2,
    n_iter:       int         = 1000,
    conv_tol:     float       = 1e-10,
    conv_window:  int         = 5,
    seed:         int         = 0,
    n_workers:    int | None  = None,
    guiding_centres: bool     = False,
    guide_refresh:   int      = 10,
    use_ss_functional: bool   = False,
    score:        "callable | None" = None,
) -> SpreadResult:
    """
    Global minimization of Omega via random restarts and optional basin hopping.

    *** OPEN BUG, HIGH PRIORITY (2026-07-30): argmin(Omega) IS NOT A SAFE
    SELECTION CRITERION FOR AN ENTANGLED MODEL. ***

    This returns `min(results, key=lambda r: r.Omega)` -- the restart with the
    lowest spread -- and nothing in that criterion knows anything about how the
    resulting model INTERPOLATES. Measured on bcc Nb (notebook 22), 8^3 mesh,
    identical data and window, varying only n_restarts:

        n_restarts   Omega_I   Omega    on-mesh   off-mesh max@E_F
             1        9.030    10.424   0.00 meV        73 meV
             3        9.030    10.423   0.00 meV      3797 meV

    The 3-restart run found a spread 0.001 Ang^2 LOWER and a model 52x worse at
    E_F. The minima are near-degenerate in Omega and wildly different in
    interpolation quality, so the selection cannot tell them apart -- and more
    restarts means more chances to pick a bad one. It is silent: `mesh_fidelity`
    reads 0.00 meV either way, because a Wannier model reproduces its own mesh
    by construction. Notebook 22's Nb superfluid weight came out 23x too large
    this way (lambda_L 4.39 nm against 21.30 from SIESTA and ~39 measured).

    MITIGATION IN PLACE: pass `score`. `pipeline.wannierize` supplies one that
    ranks restarts by `hamiltonian.hopping_range` -- the second moment of H(R),
    which is precisely the quantity Omega is blind to and which separates the
    two Nb minima by 14x where Omega separates them by 1.00006. Selection only:
    every candidate is still a genuine minimum of Omega, nothing is perturbed,
    and the returned model is one of the ones that was computed anyway. Pass
    `score=None` for the legacy argmin(Omega).

    Still treat any n_restarts > 1 model as unvalidated unless
    `analysis.bands.band_path_fidelity` has been run on it off-mesh: a better
    selection criterion does not prove the pool contained a good model.
    n_restarts = 1 is not a fix either, only a smaller exposure -- the same
    degeneracy is there, one draw instead of several.

    What needs deciding: whether to score restarts by something interpolation-
    aware (H(R) decay, or an off-mesh residual against a held-out k-set) instead
    of by Omega alone; whether the low-Omega/bad-interpolation minima are a
    genuine feature of the disentanglement or an artefact of how the subspace is
    frozen. See the memory note `project_waw_n_restarts_bug`.

    The first restart always starts from U_init (typically the SVD projection).
    Subsequent restarts use Haar-random unitaries.  After each local minimum,
    `n_hops` basin-hopping jumps are attempted; a jump is accepted only if the
    new basin yields a lower Omega.

    Restarts run in parallel using a thread pool.  PyTorch releases the GIL
    during C++ operations, so threads overlap on compute-heavy workloads.
    Each restart has its own U clone; the shared tensors (Mmn, wb, bvecs,
    kb_idx) are accessed read-only and are thread-safe.

    Args:
      U_init       : (nk, nw, nw)  initial gauge (from svd_init)
      Mmn          : (nk, nnb, nb, nb) complex  overlap matrices
      wb           : (nnb,)  real  shell weights
      bvecs        : (nk, nnb, 3) real  Cartesian b-vectors
      kb_idx       : (nk, nnb)  long  neighbour k-index table
      n_restarts   : total number of independent starts (including U_init)
      n_hops       : basin-hopping jumps per restart (0 = pure restarts)
      hop_strength : approximate Frobenius norm of the random perturbation
      optimizer    : "sgd" | "cg" | "adam"  (passed to minimize_spread)
      lr           : learning rate
      n_iter       : max iterations per local run
      conv_tol     : convergence tolerance per local run
      conv_window  : convergence window size per local run
      seed         : base RNG seed; restart i uses seed + i * 1000
      score        : optional f(SpreadResult) -> float, minimized instead of
                     Omega when choosing among restarts. Ties are broken by
                     Omega. None reproduces the legacy argmin(Omega).
      n_workers    : number of parallel threads.
                     None (default) = min(n_restarts, cpu_count).
                     1 = sequential (no thread pool).
      guiding_centres, guide_refresh : passed to minimize_spread (see there).
      use_ss_functional : passed to minimize_spread (see there; tutorial36).

    Returns:
      SpreadResult with the lowest Omega found across all restarts and hops.
    """
    _cpu = min(os.cpu_count() or 1, 16)
    if n_workers is None:
        n_workers = min(n_restarts, _cpu)
    else:
        n_workers = max(1, min(n_workers, n_restarts, _cpu))

    configs = [
        _RestartConfig(
            U_init=U_init, Mmn=Mmn, wb=wb, bvecs=bvecs, kb_idx=kb_idx,
            i_restart=i, seed=seed, n_hops=n_hops, hop_strength=hop_strength,
            optimizer=optimizer, lr=lr, n_iter=n_iter,
            conv_tol=conv_tol, conv_window=conv_window,
            guiding_centres=guiding_centres, guide_refresh=guide_refresh,
            use_ss_functional=use_ss_functional,
        )
        for i in range(n_restarts)
    ]

    if n_workers <= 1 or n_restarts <= 1:
        results = [_restart_worker(cfg) for cfg in configs]
    else:
        # Cap per-worker BLAS threads so n_workers concurrent restarts don't
        # oversubscribe the cores (each restart is compute-heavy torch/BLAS).
        from ..parallel import intraop_threads_for_pool
        saved_threads = torch.get_num_threads()
        torch.set_num_threads(intraop_threads_for_pool(n_workers))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_restart_worker, configs))
        finally:
            torch.set_num_threads(saved_threads)

    if score is None:
        return min(results, key=lambda r: r.Omega)
    scored = [(score(r), r.Omega, i) for i, r in enumerate(results)]
    return results[min(scored)[2]]
