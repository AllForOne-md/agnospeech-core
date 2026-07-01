"""Transformer HSD head: cardiffnlp/twitter-roberta-base-hate-latest, behind the
same ``HsdHead`` interface (fit / predict / proba / macro_f1).

The event HSD lead. A pretrained subword transformer, so:
- ``fit`` is a NO-OP (the head is frozen; the honest Uo-vs-Up protocol is
  preserved exactly, the detector never sees privatized text in training).
- subword tokenization means NO out-of-vocabulary harm words (the min_df blindness
  the TF-IDF spine has on "parasites"/"vermin" disappears), which is why this head
  - not the spine head - is the one that could eventually ground harmcheck.

Evaluated OUT-OF-DISTRIBUTION on Reddit (the card is trained on Twitter); NEVER
quote the model card's own F1 as ours. Also fine-tuned on AAE-bias-implicated
datasets, so a fairness upgrade must be MEASURED (per-group F1), never asserted.

Optional deps: ``pip install transformers torch`` plus a one-time model fetch
(pin to a local HF snapshot + HF_HUB_OFFLINE=1 for offline use). ``available()``
guards it; the spine falls back to the TF-IDF ``HsdHead``.
"""

from __future__ import annotations

import re

import numpy as np

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-hate-latest"
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def resolve_hate_idx(id2label: dict) -> int:
    """Index of the HATE class from a model's id2label, excluding NOT-/NON-HATE.
    Shared by the head and the Captum attributor so they cannot diverge."""
    for i, lab in (id2label or {}).items():
        u = str(lab).upper()
        if "HATE" in u and "NOT" not in u and "NON" not in u:
            return int(i)
    return (len(id2label) - 1) if id2label else 1


class TransformerHsdHead:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 16,
                 max_length: int = 128, seed: int = 0):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self._tok = None
        self._model = None
        self._hate_idx = 1
        self._fitted = False

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self._device)   # GPU when available (Colab), else CPU
        self._hate_idx = resolve_hate_idx(getattr(self._model.config, "id2label", {}))

    @staticmethod
    def _preprocess(text: str) -> str:
        text = _MENTION.sub("@user", text)
        text = _URL.sub("http", text)
        return text

    def fit(self, texts: list[str], labels: list[int]) -> "TransformerHsdHead":
        # Frozen pretrained head: no training. Load lazily and keep the protocol
        # identical to the spine (train-time never sees privatized text).
        self._load()
        self._fitted = True
        return self

    def proba(self, texts: list[str]) -> np.ndarray:
        self._load()
        torch = self._torch
        out: list[float] = []
        proc = [self._preprocess(t if t else " ") for t in texts]
        with torch.no_grad():
            for i in range(0, len(proc), self.batch_size):
                batch = proc[i:i + self.batch_size]
                enc = self._tok(batch, return_tensors="pt", truncation=True,
                                padding=True, max_length=self.max_length)
                enc = {k: v.to(self._device) for k, v in enc.items()}
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self._hate_idx]
                out.extend(probs.cpu().numpy().tolist())
        return np.asarray(out, dtype=float)

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.proba(texts) >= 0.5).astype(int)

    @staticmethod
    def macro_f1(y_true, y_pred) -> float:
        from sklearn.metrics import f1_score
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
