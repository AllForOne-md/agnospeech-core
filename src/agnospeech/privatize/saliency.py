"""Detector-grounded token saliency: the lexicon-free harm-span primitive.

The de-fixing core. Instead of a hand-built slur/target list (`HARM_LEXICON`,
`TARGET_CUES`), the harm-bearing tokens are whatever the *trained* HSD head
relies on. Two attribution methods, both behind one interface, no external data,
no hand list, so the kept spans transfer wherever the detector transfers:

- ``linear``    : signed contribution of each word to the head's hate logit
                  (logreg coefficient x that word's tf-idf weight). One matrix
                  read per post, ~2s / 1154 posts; the spine default.
- ``occlusion`` : leave-one-token-out drop in P(hate) (the ERASER
                  *comprehensiveness* operation, per token). Faithful but slow
                  (~180s / 1154 posts); offline / board-facing use.

Validated against the hand lexicon on reddit_25 (the corpus the lexicon was
partly fit to): both learned variants match utility and beat the lexicon on
worst-case privacy, internal TO, and the board proxy, with no hand list. This is
IN-DOMAIN evidence on the lexicon's home corpus; cross-corpus transfer (where
corpus-fitting becomes a liability) is untested. The kept spans follow the
detector's own coverage, so they transfer only as far as the detector does. See
the spike note `2026-06-17-lexicon-free-l2-ab-findings.md` and
`wiki/research/lexicon-free-privatization.md`.

Honesty: this is a *faithfulness* signal, not a privacy guarantee, and it
inherits the detector's bias (measure it with the faithfulness + subgroup
metrics; never assert a fairness upgrade). The linear single-matrix readout is
NOT an exact occlusion (sublinear_tf, boundary-spanning char_wb n-grams, and L2
row-normalization break the per-feature decomposition); occlusion is the faithful
cousin.
"""

from __future__ import annotations

import re

import numpy as np

# One tokenizer shared by every detector-grounded path so attribution, keep-mask,
# and re-join all agree on token boundaries. Keeps L1's [TYPE] placeholders whole.
TOKEN_RE = re.compile(r"\[[A-Z]+\]|\w+|[^\w\s]")
PLACEHOLDER_RE = re.compile(r"^\[[A-Z]+\]$")


def token_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, token) for every token, placeholders kept whole."""
    return [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(text)]


class LinearWordAttributor:
    """Reusable linear attributor built once from a fitted ``HsdHead``.

    Reads the logreg's learned word-channel weights so attribution costs one
    sparse vector op per post (no per-token re-inference). Built once per
    privatizer instance and reused across the corpus.
    """

    def __init__(self, hsd):
        # Linear attribution reads the sklearn word-channel coefficients, so it
        # requires the TF-IDF HsdHead. A transformer head (cardiff) has no such
        # channel: use method="occlusion" (proba-only) with those heads instead.
        model = getattr(hsd, "model", None)
        if model is None or not hasattr(model, "named_steps"):
            raise TypeError(
                "LinearWordAttributor needs the sklearn TF-IDF HsdHead (word "
                "channel). For a transformer head, use method='occlusion'.")
        feats = model.named_steps["feats"]
        self.word_vec = dict(feats.transformer_list)["word"]
        names = self.word_vec.get_feature_names_out()
        n_word = len(names)
        self.coef_word = hsd.model.named_steps["clf"].coef_[0][:n_word]
        self.feat_names = names

    def salience(self, text: str, spans: list[tuple[int, int, str]]) -> np.ndarray:
        """Per-token signed contribution to the hate logit, aligned to ``spans``.

        A bigram feature splits its contribution evenly across its words, so a
        token's salience sums every (uni/bi)-gram word-feature it appears in.
        L1 placeholders score 0 (never harm content).
        """
        x = self.word_vec.transform([text]).tocoo()
        contrib: dict[str, float] = {}
        for j, v in zip(x.col, x.data):
            name = self.feat_names[j]
            words = name.split()
            c = float(self.coef_word[j]) * float(v) / len(words)
            for w in words:
                contrib[w] = contrib.get(w, 0.0) + c
        sal = np.zeros(len(spans), dtype=float)
        for i, (_s, _e, tok) in enumerate(spans):
            if PLACEHOLDER_RE.match(tok):
                continue
            # Single-char and punctuation TOKEN_RE spans are absent from the
            # sklearn word vocabulary (default token_pattern \b\w\w+\b), and harm
            # words out-of-vocabulary under min_df are too; both fall through to
            # 0.0 here (harmless for 2+char harm content, but not every token is
            # attributable). The detector cannot vouch for what it never learned.
            sal[i] = contrib.get(tok.lower(), 0.0)
        return sal


def occlusion_salience(hsd, text: str, spans: list[tuple[int, int, str]]) -> np.ndarray:
    """Leave-one-token-out P(hate) drop per token (ERASER comprehensiveness).

    For each non-placeholder token, delete it and re-score; the drop in P(hate)
    is its salience. One ``proba`` call over all candidates (batched).
    """
    cands: list[str] = []
    idx: list[int] = []
    for i, (s, e, tok) in enumerate(spans):
        if PLACEHOLDER_RE.match(tok):
            continue
        cand = re.sub(r"\s{2,}", " ", (text[:s] + text[e:])).strip()
        cands.append(cand)
        idx.append(i)
    sal = np.zeros(len(spans), dtype=float)
    if not cands:
        return sal
    p_full = float(hsd.proba([text])[0])
    p_cand = hsd.proba(cands)
    for k, i in enumerate(idx):
        sal[i] = p_full - float(p_cand[k])  # positive => token pushed P(hate) up
    return sal


def keep_mask(sal: np.ndarray, keep_frac: float, window: int) -> list[bool]:
    """Keep tokens whose salience clears ``keep_frac`` x the post's max, plus a
    +-``window`` of neighbors so the kept harm stays interpretable."""
    n = len(sal)
    keep = [False] * n
    pos = sal[sal > 0]
    if not pos.size:
        return keep
    tau = max(keep_frac * float(sal.max()), 1e-9)
    for i, s in enumerate(sal):
        if s >= tau:
            for j in range(max(0, i - window), min(n, i + window + 1)):
                keep[j] = True
    return keep


def join_kept(tokens: list[str], keep: list[bool]) -> str:
    """Re-join kept tokens in original order, ellipsis-marking dropped gaps and
    tidying spacing before punctuation (the shape the lexicon L2 also emits)."""
    out: list[str] = []
    gap = False
    for i, t in enumerate(tokens):
        if keep[i]:
            if gap and out:
                out.append("…")
            out.append(t)
            gap = False
        else:
            gap = True
    s = " ".join(out)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return s.strip()
