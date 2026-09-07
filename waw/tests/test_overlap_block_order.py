"""Test the .mmn/.nnkp block-order guard."""
import numpy as np
import pytest

from waw.interfaces.quantum_espresso.pipeline import _check_overlap_block_order


def _fake(nk=8, nnb=4, seed=0):
    rng = np.random.default_rng(seed)
    nnkpts = np.stack([rng.permutation(nk)[:nnb] for _ in range(nk)])
    g = rng.integers(-1, 2, size=(nk, nnb, 3))
    kpb = [(ik, int(nnkpts[ik, p]), *[int(x) for x in g[ik, p]])
           for ik in range(nk) for p in range(nnb)]
    return {"nnkpts": nnkpts, "g_vectors": g}, kpb


def test_guard_accepts_matching_order(tmp_path):
    nnkp, kpb = _fake()
    _check_overlap_block_order(kpb, nnkp, tmp_path, "s")   # must not raise


def test_guard_rejects_permuted_neighbours(tmp_path):
    nnkp, kpb = _fake()
    kpb[0], kpb[1] = kpb[1], kpb[0]      # swap two neighbours of k=0
    with pytest.raises(ValueError, match="neighbour block order disagrees"):
        _check_overlap_block_order(kpb, nnkp, tmp_path, "s")


def test_guard_rejects_wrong_block_count(tmp_path):
    nnkp, kpb = _fake()
    with pytest.raises(ValueError, match="blocks but the"):
        _check_overlap_block_order(kpb[:-1], nnkp, tmp_path, "s")
