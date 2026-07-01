"""Dataset interface. Every loader is swappable behind ``DatasetLoader`` so a
contributor can drop a new corpus (e.g. HateXplain) behind the same contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Post:
    """One unit of text. ``author`` is a pseudonymous id; we never store or
    surface a real identity (hard rule: never deanonymise, score text not people).
    ``label`` is the hate-speech gold label (1 = hate, 0 = not)."""

    id: str
    author: str
    text: str
    label: int


class DatasetLoader(ABC):
    """Loads a corpus as a list of :class:`Post`. Implementations must be pure
    reads with no network at eval time so a reviewer reproduces exact numbers."""

    name: str

    @abstractmethod
    def load(self) -> list[Post]:  # pragma: no cover - interface
        ...
