"""Pinned revisions. Every reported number must trace to a pinned component.

The CPU-only spine pins library versions and seeds. The event build adds model
SHAs (HF revision hashes), the llama.cpp build hash, and quantization level here
behind the same record, so anyone reruns from a clean clone and gets the
exact numbers within the stated tolerance."""

from __future__ import annotations

import platform
import sys


def runtime_revisions() -> dict:
    rev = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hsd_head": "tfidf(word1-2 + char2-5) + logreg(C=4, balanced)  [spine]",
        "hsd_head_event_target": "cardiffnlp/twitter-roberta-base-hate-latest",
        "authorship_attacker": "tfidf(char_wb 3-5) + linearSVC(C=1)  [spine]",
        "authorship_attacker_event_target": "DeBERTa-v3 + char-ngram SVM",
        "l1": "regex + gazetteer  [spine] / Presidio + GLiNER  [event]",
        "l3": "rationale-anchored non-DP rewrite [spine] / DP-Prompt logit-clip [event]",
    }
    for mod in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            rev[mod] = __import__(mod).__version__
        except Exception:
            rev[mod] = "missing"
    return rev
