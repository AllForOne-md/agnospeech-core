"""Frozen general-purpose semantic similarity (replaces the per-corpus TF-IDF refit).

``quality.mean_semantic_similarity`` refits a char-ngram TF-IDF vocabulary on the
eval corpus EVERY call, which is itself a per-corpus-fit artifact. This swaps in a
frozen, pretrained, numpy-only embedding (model2vec ``potion-base-8M``, MIT) whose
vocabulary is fixed a priori and corpus-independent, behind the same
``(originals, rewrites) -> mean cosine`` interface, so curve.py / quality consumers
swap with no change.

Optional dep: ``pip install model2vec`` plus a one-time model fetch (~30MB; pin to a
local HF snapshot for offline use). ``available()`` checks only that the LIBRARY
imports - the model snapshot is fetched lazily on first ``_load`` (and raises there
if offline with no snapshot), so a dispatcher should fall back to the TF-IDF proxy
on that load error, not assume available() implies a usable snapshot.

Honesty: this carries the embedding's pretraining bias like any learned component;
it is a meaning-preservation proxy, not a guarantee. The counterspeech-parity report
mode (all-MiniLM-L6-v2) is a separate, heavier option.
"""

from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}
DEFAULT_MODEL = "minishlab/potion-base-8M"


def available() -> bool:
    try:
        import model2vec  # noqa: F401
        return True
    except Exception:
        return False


def _load(name: str):
    if name not in _MODEL_CACHE:
        from model2vec import StaticModel
        _MODEL_CACHE[name] = StaticModel.from_pretrained(name)
    return _MODEL_CACHE[name]


def model2vec_semantic_similarity(originals: list[str], rewrites: list[str],
                                  model_name: str = DEFAULT_MODEL) -> float:
    """Mean row-wise cosine between each original and its rewrite, using a frozen
    pretrained static embedding. Empty / zero-norm rows score 0.0."""
    if not originals:
        return 0.0
    model = _load(model_name)
    a = np.asarray(model.encode(list(originals)), dtype=float)
    b = np.asarray(model.encode([t if t else " " for t in rewrites]), dtype=float)
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = an * bn
    dots = np.sum(a * b, axis=1)
    sims = np.where(denom > 1e-12, dots / np.maximum(denom, 1e-12), 0.0)
    return float(np.mean(sims))
