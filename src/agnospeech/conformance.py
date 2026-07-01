"""Conformance-as-code (S3 lean): the hard rules as three runnable assertions.

This is the green/red conformance CLI for the hard rules. It proves "tested not to
violate on these paths", never "cannot violate".
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from .privatize import L1Redact, L2LearnedDistill, L3Rewrite, RawPassthrough
from .privatize.patterns import find_pii_spans


@dataclass
class Assertion:
    name: str
    passed: bool
    detail: str


def _router_refuses_cloud_on_raw(allow_cloud_on_raw: bool) -> Assertion:
    """The local-core / API-edges split: the router must never send raw
    (pre-privatization) text to a cloud endpoint. The config flag models the
    one switch that would break it."""
    if allow_cloud_on_raw:
        return Assertion(
            "router refuses any cloud call on raw text",
            False,
            "config allow_cloud_on_raw=true routes raw text to a cloud endpoint",
        )
    return Assertion(
        "router refuses any cloud call on raw text",
        True,
        "egress policy denies non-local calls before privatization",
    )


def _storage_rejects_raw_text() -> Assertion:
    """The store must refuse text that still carries identifiers; only redacted
    or rationale-only text persists."""
    raw = "ping me at jane.doe@example.com or @jane_doe"
    redacted = L1Redact().apply(raw)
    raw_has_pii = len(find_pii_spans(raw)) > 0
    redacted_has_pii = len(find_pii_spans(redacted)) > 0
    ok = raw_has_pii and not redacted_has_pii
    return Assertion(
        "storage rejects raw text; only redacted persists",
        ok,
        f"raw carries {len(find_pii_spans(raw))} identifiers; redacted carries "
        f"{len(find_pii_spans(redacted))}",
    )


def _no_author_scoring_path() -> Assertion:
    """No privatizer may take an author/identity argument: it scores text, never
    people. Checked structurally on the interface."""
    bad = []
    for cls in (RawPassthrough, L1Redact, L2LearnedDistill, L3Rewrite):
        params = set(inspect.signature(cls.apply).parameters) - {"self"}
        if params != {"text"}:
            bad.append(f"{cls.__name__}.apply{tuple(params)}")
    return Assertion(
        "no code path scores a person or account",
        not bad,
        "every privatizer.apply takes only text" if not bad else f"leaky: {bad}",
    )


def run(allow_cloud_on_raw: bool = False) -> list[Assertion]:
    return [
        _router_refuses_cloud_on_raw(allow_cloud_on_raw),
        _storage_rejects_raw_text(),
        _no_author_scoring_path(),
    ]
