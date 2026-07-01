"""L3 Rewrite: rationale-anchored local style rewrite.

Goal: destroy the authorship fingerprint (capitalization quirks, character
elongation, punctuation habits, lexical choices, function-word patterns) while
anchoring the harm rationale so the detector still fires. An ``intensity`` knob
(epsilon proxy) trades readability for privacy.

No hand lists. The harm anchor is detector-grounded (the same learned spans
``L2LearnedDistill`` keeps — linear for a TF-IDF head, occlusion for a transformer
head, or a precomputed cache). The style canonicalization is NLTK stopwords +
WordNet dominant-synset substitution, both prebuilt and corpus-independent.

HONESTY: this is a clearly-labeled NON-DP local rewrite. ``intensity`` is not a
privacy budget. Output is "DP-grounded (pending the logit spike)", never "DP".
Optional DP-grounded neural rewriters (DP-MLM / PrivFill) are not part of this minimal build.
"""

from __future__ import annotations

import hashlib
import random
import re

from .base import Privatizer
from .l1_redact import L1Redact
from .saliency import LinearWordAttributor, keep_mask, occlusion_salience, token_spans

_TOKEN = re.compile(r"\[[A-Z]+\]|[A-Za-z]+'?[A-Za-z]*|\d+|[^\w\s]")


class L3Rewrite(Privatizer):
    level = "L3"
    name = "rationale_anchored_rewrite_nondp_detector_nltk"

    def __init__(self, intensity: float = 0.6, seed: int = 0, hsd=None,
                 keep_frac: float = 0.6, method: str = "auto",
                 anchor_words: dict[str, set[str]] | None = None):
        if hsd is None and anchor_words is None:
            raise ValueError("L3Rewrite needs a fitted hsd (detector anchor) or a "
                             "precomputed anchor_words cache")
        from . import style as _style
        if not _style.available():
            raise ValueError("L3Rewrite needs nltk + stopwords/wordnet/omw-1.4 corpora "
                             "(python -m nltk.downloader stopwords wordnet omw-1.4)")
        self.intensity = float(intensity)
        self.seed = seed
        self.keep_frac = keep_frac
        self.hsd = hsd
        self._l1 = L1Redact()
        # Precomputed {L1-applied text -> anchor words}: lets the harness compute
        # the (possibly expensive, e.g. transformer-occlusion) attribution ONCE
        # per text and reuse it across the whole intensity sweep + the ladder.
        self._anchor_cache = anchor_words
        # Attribution method: 'auto' = linear when the head exposes the sklearn
        # word channel (TF-IDF), else occlusion (proba-only) so a transformer head
        # and future GPU models work too.
        self._method = None
        self._attr = None
        if anchor_words is None:
            m = method
            if m == "auto":
                m = ("linear" if hasattr(getattr(hsd, "model", None), "named_steps")
                     else "occlusion")
            if m not in ("linear", "occlusion"):
                raise ValueError(f"unknown L3 anchor method {m!r} (linear|occlusion)")
            self._method = m
            self._attr = LinearWordAttributor(hsd) if m == "linear" else None

    def _anchor_words(self, text: str) -> set[str]:
        """Lowercased detector-salient tokens to pass through verbatim as the harm
        rationale (the same primitive L2LearnedDistill uses)."""
        if self._anchor_cache is not None:
            return self._anchor_cache.get(text, set())
        spans = token_spans(text)
        if not spans:
            return set()
        if self._method == "occlusion":
            sal = occlusion_salience(self.hsd, text, spans)
        else:
            sal = self._attr.salience(text, spans)
        keep = keep_mask(sal, self.keep_frac, window=0)
        return {t.lower() for (_s, _e, t), k in zip(spans, keep) if k}

    def _canon(self, low: str) -> tuple[str, bool]:
        """(replacement, handled). WordNet dominant-synset lemma — a prebuilt,
        corpus-independent canonicalization (no hand map)."""
        from .style import wordnet_canonical
        rep = wordnet_canonical(low)
        return (rep, True) if rep != low else (low, False)

    def _is_function(self, low: str) -> bool:
        from .style import function_words
        return low in function_words()

    def apply(self, text: str) -> str:
        text = self._l1.apply(text)            # never resurrect PII
        salient = self._anchor_words(text)
        toks = _TOKEN.findall(text)            # tokenize BEFORE casing changes,
        # so L1's uppercase [TYPE] placeholders survive as single tokens.
        digest = hashlib.md5(text.encode("utf-8")).digest()
        rng = random.Random((self.seed * 1_000_003) ^ int.from_bytes(digest[:4], "big"))
        out: list[str] = []
        for t in toks:
            if t.startswith("[") and t.endswith("]") and t[1:-1].isupper():
                out.append(t)
                continue
            low = re.sub(r"(.)\1{2,}", r"\1\1", t.lower())
            if low in salient:
                out.append(low)                # anchor the harm rationale
                continue
            rep, handled = self._canon(low)    # canonicalize lexical choice
            if handled:
                if rep:
                    out.append(rep)
                continue
            if rng.random() < self.intensity:
                if self._is_function(low):
                    continue
                if low.isalpha():
                    out.append("·")       # neutral content placeholder
                continue
            out.append(low)
        s = " ".join(out)
        s = re.sub(r"\s+([.,!?;:])", r"\1", s)        # tidy spacing before punct
        s = re.sub(r"([!?.,])\1{1,}", r"\1", s)       # !!! -> !
        s = re.sub(r"\s+", " ", s)
        return s.strip()
