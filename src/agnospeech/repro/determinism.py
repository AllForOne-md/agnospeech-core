"""Determinism harness. Reproducibility is scoped to a stated numeric tolerance
on TO, never bit-equality (DP sampling + BLAS nondeterminism make bit-exact a
false promise). Call ``pin()`` before any eval run."""

from __future__ import annotations

import os
import random

GLOBAL_SEED = 20260617
TO_TOLERANCE = 0.01  # absolute tolerance on reported TO numbers across reruns


def pin(seed: int = GLOBAL_SEED) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
