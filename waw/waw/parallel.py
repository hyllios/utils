"""
CPU thread configuration for waw.

waw's numerics are batched torch linear algebra (``torch.linalg.eigh`` /
``bmm`` over stacks of k-points), whose parallelism comes from the
underlying BLAS/OpenMP thread pool. Many HPC environments pin
``OMP_NUM_THREADS=1``; call :func:`set_num_threads` to use more cores.

Two kinds of parallelism coexist:

* **intra-op** — a single torch op split across threads, controlled via
  ``torch.set_num_threads`` (the runtime control) plus the
  ``*_NUM_THREADS`` env vars (read by numpy's BLAS and child processes).
* **task** — independent Python-level tasks run concurrently, e.g. the
  wannierize restart pool in :mod:`waw.core.global_optim`. That pool caps
  its per-worker intra-op threads (via :func:`intraop_threads_for_pool`)
  so ``n_workers * intra_threads <= get_num_threads()`` — no
  oversubscription.
"""

from __future__ import annotations

import os

import torch

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def default_num_threads() -> int:
    """
    Thread count to use when the caller does not ask for a specific one.

    An explicitly-set ``OMP_NUM_THREADS`` wins: batch systems and HPC modules
    pin it deliberately (this project's Quantum ESPRESSO module sets it to 1 so
    that MPI ranks don't each spawn a full thread pool), and silently
    overriding that is how you get a 16-rank job trying to run 16x16 threads.
    Falls back to the CPUs actually available to the process.

    Note this is only the DEFAULT -- ``set_num_threads(16)`` is an explicit
    request and still overrides the environment, which is what a single-process
    analysis run wants (QE/VASP children are separately pinned back to one
    thread per rank by the interfaces' own ``_mpi_env``/``_vasp_env``).
    """
    env = os.environ.get("OMP_NUM_THREADS", "").strip()
    if env:
        try:
            if int(env) > 0:
                return int(env)
        except ValueError:
            pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:            # not available on all platforms
        return os.cpu_count() or 1


def set_num_threads(n: int | None = None) -> int:
    """
    Set the CPU thread count for waw's batched linear algebra.

    ``n=None`` uses every core available to the process. Returns the value set.

    Three things are set, because no single one covers both backends:

    * ``torch.set_num_threads`` -- torch's intra-op pool, effective at once;
    * the OMP/MKL/OpenBLAS env vars -- for any child process we spawn, and for
      BLAS libraries that read them at first use;
    * the ALREADY-LOADED BLAS thread count, via ``threadpoolctl`` -- this is
      the one that matters for numpy.

    That last step exists because the env vars are read by OpenBLAS/MKL when
    the library is *loaded*, i.e. at ``import numpy``. If the environment
    pinned ``OMP_NUM_THREADS=1`` before then -- which HPC modules routinely do
    (this project's own Quantum ESPRESSO module does exactly that) -- setting
    the variable afterwards is silently ignored and every numpy ``@``/GEMM runs
    single-threaded, no matter what this function was asked for. Measured on a
    2000x2000 complex GEMM under a module-pinned ``OMP_NUM_THREADS=1``: 1.22 s
    before, 0.10 s after (12x). ``threadpoolctl`` reaches into the loaded
    library and re-sets it at runtime, so the call works regardless of when it
    happens. If ``threadpoolctl`` is unavailable the rest still applies and
    numpy-side BLAS keeps whatever the environment gave it.
    """
    if n is None:
        n = default_num_threads()
    n = max(1, int(n))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(n)
    torch.set_num_threads(n)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass
    return n


def get_num_threads() -> int:
    """torch's current intra-op thread count."""
    return torch.get_num_threads()


def intraop_threads_for_pool(n_workers: int) -> int:
    """
    Per-worker intra-op thread budget for a task pool of ``n_workers``, so the
    product stays within the process-wide count set by :func:`set_num_threads`
    (avoids BLAS oversubscription when restarts run concurrently).
    """
    return max(1, get_num_threads() // max(1, n_workers))
