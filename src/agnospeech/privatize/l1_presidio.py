"""Presidio + GLiNER L1 redaction drop-in (the doc's P2 L1 upgrade).

Behind the same ``find_pii_spans`` / ``apply`` contract and the same ``[TYPE]``
placeholder vocabulary as the spine ``L1Redact``, so nothing downstream changes.
Presidio's AnalyzerEngine plus the first-party GLiNERRecognizer
(urchade/gliner_multi_pii-v1, a zero-shot generalist NER) catch the title-less
names / orgs / locations the spine's single title heuristic misses.

A harm-relevant PRESERVE-LIST (protected-characteristic group nouns, a general
policy list built independently of the eval corpus) keeps identity-group /
nationality targets from being redacted, so HSD F1 is not depressed by removing
the harm target. Reframed narrowly: this is FAIRER entity redaction (avoids
name-shape bias), NOT the AAE harm-lexicon bias (that lives in L2 / harmcheck).

Optional deps: presidio-analyzer (+ spaCy model) and gliner (+ its model). Both
best-effort: GLiNER is added only if it loads; if presidio itself is absent,
``available()`` is False and the spine ``L1Redact`` is the fallback.
"""

from __future__ import annotations

from .base import Privatizer
from .l1_redact import L1Redact
from .patterns import find_pii_spans as _regex_spans

# Presidio entity_type -> spine [TYPE] placeholder vocabulary (patterns.py).
_TYPE_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "URL": "URL",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "IP_ADDRESS": "NUMBER",
    "CREDIT_CARD": "NUMBER",
    "US_SSN": "NUMBER",
    "IBAN_CODE": "NUMBER",
    "ORGANIZATION": "ORG",
}

# GLiNER is a zero-shot, natural-language-prompted NER: it scores far better on
# descriptive labels ("person") than on spaCy/HF tag codes ("PER"/"GPE"/"NORP",
# which is what presidio's DEFAULT NerModelConfiguration would query it with).
# Map each GLiNER prompt label to a presidio entity_type that _TYPE_MAP knows.
_GLINER_ENTITY_MAPPING = {
    "person": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "email": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
}

# General protected-characteristic group nouns kept as harm TARGETS (policy list,
# independent of the eval corpus). Preserving these keeps the harm interpretable.
DEFAULT_PRESERVE = frozenset({
    "women", "men", "muslims", "christians", "jews", "atheists", "blacks",
    "whites", "asians", "latinos", "immigrants", "migrants", "refugees", "gays",
    "lesbians", "trans", "disabled", "elderly",
})


def available() -> bool:
    try:
        import presidio_analyzer  # noqa: F401
        return True
    except Exception:
        return False


class PresidioGlinerRedact(Privatizer):
    level = "L1"
    name = "redact_presidio_gliner"

    def __init__(self, preserve: frozenset[str] | None = None,
                 use_gliner: bool = True, language: str = "en",
                 gliner_model: str = "urchade/gliner_multi_pii-v1"):
        self.preserve = DEFAULT_PRESERVE if preserve is None else preserve
        self.use_gliner = use_gliner
        self.language = language
        self.gliner_model = gliner_model
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return
        from presidio_analyzer import AnalyzerEngine
        engine = AnalyzerEngine()
        if self.use_gliner:
            try:  # GLiNER recognizer is best-effort: add it only if it loads
                from presidio_analyzer.predefined_recognizers import GLiNERRecognizer
                recognizer = GLiNERRecognizer(
                    model_name=self.gliner_model,
                    entity_mapping=_GLINER_ENTITY_MAPPING,
                )
                recognizer.load()  # download/load the model now so a failure
                # surfaces here (and is caught) rather than at analyze() time.
                engine.registry.add_recognizer(recognizer)
            except Exception:
                pass  # fall back to presidio's default recognizers
        self._engine = engine

    def find_pii_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Combine the spine's high-precision regexes (handles, emails, phones,
        urls) with Presidio's NER (names, locations, orgs). Regex spans win on
        overlap: they are the doc's "six transferable regexes" Presidio's
        recognizers (URL grabbing part of an email, etc.) must not clobber."""
        self._load()
        spans: list[tuple[int, int, str]] = []
        taken = [False] * (len(text) + 1)

        # 1. spine regexes first (high precision, kept verbatim per the doc).
        for s, e, ptype in _regex_spans(text):
            if text[s:e].strip().lower() in self.preserve:
                continue
            for i in range(s, e):
                taken[i] = True
            spans.append((s, e, ptype))

        # 2. Presidio NER for what the regexes do not cover. On partial overlap,
        # CLIP the NER span to its taken-free sub-runs (do not drop it wholesale -
        # that would leak the non-overlapping remainder of a PERSON/LOCATION/ORG).
        results = self._engine.analyze(text=text, language=self.language)
        for r in sorted(results, key=lambda r: (r.start, -(r.end - r.start))):
            ptype = _TYPE_MAP.get(r.entity_type, r.entity_type)
            a = r.start
            while a < r.end:
                if taken[a]:
                    a += 1
                    continue
                b = a
                while b < r.end and not taken[b]:
                    b += 1
                sub = text[a:b]
                clipped = (a, b) != (r.start, r.end)
                if sub.strip().lower() in self.preserve or not any(c.isalnum() for c in sub):
                    a = b  # keep harm targets and skip whitespace-only runs
                    continue
                # A CLIPPED remnant is only worth a placeholder if it still
                # carries entity-grade signal. When a higher-precision regex
                # already claimed the structured payload (e.g. presidio's URL
                # recognizer fires on "cell 555-019-2245" but the PHONE regex
                # took the number), the leftover is a connective word like
                # "cell " - redacting it would drop a real word and emit a junk
                # [URL]. Only NAME-shaped remnants (PERSON/LOCATION/ORG) survive
                # clipping; structured-type remnants (URL/EMAIL/PHONE/NUMBER) do
                # not - the regexes own those in full.
                if clipped and ptype not in ("PERSON", "LOCATION", "ORG"):
                    a = b
                    continue
                for i in range(a, b):
                    taken[i] = True
                spans.append((a, b, ptype))
                a = b
        spans.sort()
        return spans

    def apply(self, text: str) -> str:
        try:
            spans = self.find_pii_spans(text)
        except Exception:
            return L1Redact().apply(text)  # graceful fallback to the spine
        if not spans:
            return text
        out, cursor = [], 0
        for s, e, ptype in spans:
            out.append(text[cursor:s])
            out.append(f"[{ptype}]")
            cursor = e
        out.append(text[cursor:])
        return "".join(out)
