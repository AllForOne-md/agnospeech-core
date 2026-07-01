"""Privatizer interface. Every privacy level implements the same contract so the
harness iterates over levels uniformly and each level is independently shippable.

A privatizer is a pure text-to-text transform: raw text in, privatized text out.
It never sees author labels, never calls a cloud API, never stores raw text."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Privatizer(ABC):
    level: str  # "L0" | "L1" | "L2" | "L3"
    name: str

    @abstractmethod
    def apply(self, text: str) -> str:  # pragma: no cover - interface
        ...

    def apply_many(self, texts: list[str]) -> list[str]:
        return [self.apply(t) for t in texts]


class RawPassthrough(Privatizer):
    """L0: the untouched baseline. Utility_original and Privacy_original are
    measured here."""

    level = "L0"
    name = "raw"

    def apply(self, text: str) -> str:
        return text
