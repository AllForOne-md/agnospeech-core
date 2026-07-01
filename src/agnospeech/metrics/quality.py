"""Text-quality floors for the Utility-vs-Privacy curve.

A privatized text that is unreadable noise can post a high TO (the degenerate
optimum). The curve is dishonest without floors, so we compute readability and
semantic-similarity proxies and refuse any operating point that breaks a floor.

CPU-only proxies (no model):
- readability     : fraction of output tokens that are real word tokens (not the
                    neutral '·' blanking token, not bare punctuation). Falls as
                    the rewrite blanks more content.
- semantic_sim    : char-n-gram TF-IDF cosine between original and rewritten,
                    averaged over the corpus. A meaning-preservation proxy.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_WORD = re.compile(r"[A-Za-z]{2,}")
_TOK = re.compile(r"\S+")


def readability(text: str) -> float:
    toks = _TOK.findall(text)
    if not toks:
        return 0.0
    words = sum(1 for t in toks if _WORD.fullmatch(t))
    return words / len(toks)


def mean_readability(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return float(np.mean([readability(t) for t in texts]))


def mean_semantic_similarity(originals: list[str], rewrites: list[str]) -> float:
    """Average char-n-gram TF-IDF cosine between each original and its rewrite."""
    if not originals:
        return 0.0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    # Fit on the union so both sides share a vocabulary.
    vec.fit(originals + rewrites)
    a = vec.transform(originals)
    b = vec.transform(rewrites)
    sims = [float(cosine_similarity(a[i], b[i])[0, 0]) for i in range(len(originals))]
    return float(np.mean(sims))
