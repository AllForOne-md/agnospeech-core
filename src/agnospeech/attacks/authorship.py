"""Authorship-attribution attack (the privacy adversary that feeds TO).

Closed-world N-candidate attribution: given a privatized post, guess which of the
N corpus authors wrote it. Character n-gram TF-IDF + linear SVM is the classic,
strong, CPU-cheap stylometric attacker (Koppel et al.; the char-n-gram
SVM baseline). The DeBERTa-v3 neural attacker is the event drop-in behind this
same interface.

Two strengths, both reported (the static-vs-adaptive split is non-negotiable):
- static  : trained on RAW text, then shown privatized text. Optimistic for us:
            our privatization scrambles the style it learned, so its accuracy
            falls fast. This is the INFLATED privacy number.
- adaptive : trained on privatized text at the SAME level. It adapts to the
            transform. Harder, more honest, and the default HEADLINE.

Privacy leakage = attack accuracy. Lower after protection is better.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .base import Attack


class AuthorshipAttacker(Attack):
    name = "authorship_char_word_svm"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = self._build(min_df=2)
        self._fitted = False

    def _build(self, min_df: int) -> Pipeline:
        # A deliberately strong stylometric adversary: character n-grams (the
        # classic authorship signal) plus word n-grams, so our privacy claims
        # are made against a credible attacker, not a weak one.
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True
        )
        word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True
        )
        return Pipeline(
            [
                ("feats", FeatureUnion([("char", char), ("word", word)])),
                ("clf", LinearSVC(C=1.0, random_state=self.seed)),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> "AuthorshipAttacker":
        # Corpus-size-aware min_df mirrors the HSD head: the attacker is re-fit on
        # shortened privatized text, where the fixed min_df=2 can empty the vocab.
        if len(texts) < 50:
            self.model = self._build(min_df=1)
        self.model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(texts)

    def accuracy(self, texts: list[str], labels: list[str]) -> float:
        pred = self.predict(texts)
        return float(np.mean(pred == np.asarray(labels)))

    def correct_mask(self, texts: list[str], labels: list[str]) -> np.ndarray:
        """Per-instance 1/0 correct attribution, cached once so the paired
        bootstrap can resample indices without re-running the attacker."""
        return (self.predict(texts) == np.asarray(labels)).astype(int)

    def fingerprint_strength(self, texts: list[str]) -> np.ndarray:
        """Per-text confidence of the attacker's best guess, normalized to
        roughly [0, 1] (top margin minus runner-up, squashed). Used only as a
        per-post visual of a style fingerprint collapsing across levels; it is
        never a per-author score that leaves the harness."""
        df = self.model.decision_function(texts)
        df = np.atleast_2d(df)
        top = np.sort(df, axis=1)
        margin = top[:, -1] - top[:, -2]
        return 1.0 / (1.0 + np.exp(-margin))
