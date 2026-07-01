"""L2 Distill: rationale-only harm-span extraction.

Keep the harm-bearing spans, drop the surrounding stylistic tissue. The detector
still fires on the retained harm content (utility preserved), while a large part
of the authorship fingerprint (function words, idiosyncratic phrasing) is thrown
away (privacy improves over L1).

Invariant (hard-rule guard): L2 runs on already-L1-redacted text, so distillation
can never resurrect an identifier the harm span carried.

``L2LearnedDistill`` keeps the tokens the *trained* HSD
head relies on (detector-grounded saliency: linear for the TF-IDF head, occlusion
for a transformer head). The harm signal is whatever the detector learned on this
corpus (or, with a pretrained subword head, what it learned in general) — so the
kept spans follow the detector's coverage and transfer as far as the detector does.
"""

from __future__ import annotations

from .base import Privatizer
from .l1_redact import L1Redact
from .saliency import (
    LinearWordAttributor,
    join_kept,
    keep_mask,
    occlusion_salience,
    token_spans,
)


class L2LearnedDistill(Privatizer):
    """Lexicon-free L2: keep the tokens the trained HSD head actually relies on.

    De-fixed harm-span extraction. ``method``:
      - ``linear``    : signed contribution to the hate logit (fast, TF-IDF head)
      - ``occlusion`` : leave-one-token-out P(hate) drop (ERASER comprehensiveness,
                        faithful but slow; works on any head incl. a transformer)

    No slur list, no target-word list: the harm signal is whatever the detector
    learned, so the kept spans follow the detector's own coverage. The fitted
    ``HsdHead`` is threaded in by the harness / scripts.

    ``keep_frac`` is one global knob (the defensible operating point is ~0.6,
    where utility_ratio stays ~1.0; pushing it higher invites the same-head
    sufficiency-gaming Goodhart effect, so attribute here and measure utility with
    a *different* detector — see metrics/faithfulness.py).
    """

    level = "L2"

    def __init__(self, hsd, method: str = "linear", keep_frac: float = 0.6,
                 window: int = 1):
        if method not in ("linear", "occlusion"):
            raise ValueError(f"unknown L2 method {method!r}")
        self.hsd = hsd
        self.method = method
        self.keep_frac = keep_frac
        self.window = window
        self.name = f"rationale_distill_{method}_learned"
        self._l1 = L1Redact()
        self._attr = LinearWordAttributor(hsd) if method == "linear" else None

    def _salience_and_spans(self, text: str):
        spans = token_spans(text)
        if not spans:
            return spans, None
        if self.method == "occlusion":
            sal = occlusion_salience(self.hsd, text, spans)
        else:
            sal = self._attr.salience(text, spans)
        return spans, sal

    def harm_spans(self, text: str) -> list[tuple[int, int]]:
        """(start, end) offsets of kept harm tokens on the L1-redacted text.

        The detector-grounded anchor L3 segments on (segment-and-skip): rewrite
        everything outside these spans, pass the harm spans through.
        """
        text = self._l1.apply(text)
        spans, sal = self._salience_and_spans(text)
        if sal is None:
            return []
        keep = keep_mask(sal, self.keep_frac, self.window)
        return [(s, e) for (s, e, _t), k in zip(spans, keep) if k]

    def apply(self, text: str) -> str:
        text = self._l1.apply(text)  # invariant: L1 first
        spans, sal = self._salience_and_spans(text)
        if sal is None:
            return text
        keep = keep_mask(sal, self.keep_frac, self.window)
        if not any(keep):
            # detector found no harm-bearing token: fall back to L1 (no emptying).
            return text
        return join_kept([t for _s, _e, t in spans], keep)
