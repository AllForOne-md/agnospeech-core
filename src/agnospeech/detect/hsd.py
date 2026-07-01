"""Hate-speech detection head.

ONE shared calibrated head, trained once on raw text and then applied unchanged
to raw and to each privatized version. This is the honest protocol for comparing
Utility_original against Utility_protected: the detector is frozen, privatization
only changes its inputs. (Master build prompt section 7: "One shared calibrated
HSD head across levels for honest Uo vs Up".)

Lead model at the event is cardiffnlp/twitter-roberta-base-hate-latest behind
this same ``HsdHead`` interface; the CPU-only spine ships a TF-IDF + logistic
head so every number reproduces with no model download.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


class HsdHead:
    """Calibrated binary hate-speech classifier. Outputs class labels and
    macro-F1. Macro-F1 (not accuracy) is the utility metric, because the corpus
    is class-imbalanced and macro-F1 penalizes a majority-class shortcut."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = self._build(min_df=2)
        self._fitted = False

    def _build(self, min_df: int) -> Pipeline:
        word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True
        )
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True
        )
        return Pipeline(
            [
                ("feats", FeatureUnion([("word", word), ("char", char)])),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=4.0,
                        class_weight="balanced",
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: list[int]) -> "HsdHead":
        # Corpus-size-aware min_df: on a tiny or heavily-privatized corpus, the
        # fixed min_df=2 throws "empty vocabulary"; drop to 1 there. On reddit_25
        # (N=808 train) this stays 2, so the headline numbers are unchanged.
        if len(texts) < 50:
            self.model = self._build(min_df=1)
        self.model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(texts)

    def proba(self, texts: list[str]) -> np.ndarray:
        """P(hate) per text. Used for the per-post demo readout (the detector's
        confidence that the post is hateful, which should stay stable across
        privacy levels)."""
        return self.model.predict_proba(texts)[:, 1]

    @staticmethod
    def macro_f1(y_true: list[int], y_pred) -> float:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


class DualHsdHead:
    """A char-only head that shares no features with the L2 WORD-CHANNEL keep
    decision, for a Goodhart-attenuated (not fully independent) utility read.

    The learned L2 keeps the tokens the PRIMARY head's WORD channel finds salient,
    so measuring utility with that same head inflates F1 and pushes ERASER
    sufficiency negative (rationale-only scores higher than the full text). This
    head is char_wb(3,5)-only with no word channel, so it shares no features with
    the L2 keep decision (which reads only the word channel). HONESTY: it is NOT a
    fully independent detector. Its char basis is a subset of the primary head's
    own char_wb(2,5) channel, so its proba correlates ~0.98 with the primary, and
    the negative ERASER sufficiency is ATTENUATED, not rescued (it stays negative).
    The meaningful Goodhart-resistant signal is the dual UTILITY ratio (which holds
    at 1.0), not the still-negative dual sufficiency. Different seed alone would be
    useless: the TF-IDF + lbfgs pipeline is deterministic, so two seeds give the
    identical model. The genuinely independent judge is a pretrained transformer
    head (cardiff, event P1); this is the CPU-only stand-in.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.model = self._build(min_df=2)
        self._fitted = False

    def _build(self, min_df: int) -> Pipeline:
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True
        )
        return Pipeline(
            [
                ("feats", char),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, C=4.0, class_weight="balanced",
                        random_state=self.seed,
                    ),
                ),
            ]
        )

    def fit(self, texts: list[str], labels: list[int]) -> "DualHsdHead":
        if len(texts) < 50:
            self.model = self._build(min_df=1)
        self.model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(texts)

    def proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)[:, 1]
