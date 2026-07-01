"""Attack interface. Every attack lives strictly inside the eval harness. It
evaluates OUR OWN mechanism on a benchmark / pseudonymous corpus. Per-author
scores are never surfaced in the API or UI and never stored (hard rule guard).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Attack(ABC):
    name: str

    @abstractmethod
    def fit(self, texts: list[str], labels: list[str]) -> "Attack":  # pragma: no cover
        ...

    @abstractmethod
    def accuracy(self, texts: list[str], labels: list[str]) -> float:  # pragma: no cover
        ...
