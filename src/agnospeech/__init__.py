"""AgnoSpeech: privacy-preserving hate speech detection.

A tiered text-to-text privatization mechanism whose every privacy claim is
attack-measured and conformance-tested.

Privacy levels: L1 Redact, L2 Distill, L3 DP-grounded Rewrite.
Objective: the PrivHSD trade-off score (TO)
    TO = (Utility_protected / Utility_original) - (Privacy_protected / Privacy_original)
with majority-class baseline correction on the utility term.

Honesty discipline (see docs/limitations-ledger.md):
- L3 output is DP-grounded, never "differentially private".
- The self-attack stop is a calibrated stopping rule, never a certificate.
- Privacy is closed-world attribution, an upper-bias (optimistic) estimate.
- The adaptive attacker is the default headline, not the static one.
"""

__version__ = "0.1.0"
