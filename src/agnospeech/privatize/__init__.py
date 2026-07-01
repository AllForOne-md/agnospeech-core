from .base import Privatizer, RawPassthrough
from .l1_redact import L1Redact
from .l2_rationale import L2LearnedDistill
from .l3_dp_rewrite import L3Rewrite


def learned_levels(hsd, l3_intensity: float = 0.6, seed: int = 0,
                   l2_method: str = "linear", keep_frac: float = 0.6,
                   ) -> dict[str, Privatizer]:
    """The lexicon-free L0->L3 dial: detector-grounded learned L2 + detector-anchored
    L3 (NLTK/WordNet style). Needs the fitted ``HsdHead``; no hand wordlist anywhere."""
    return {
        "L0": RawPassthrough(),
        "L1": L1Redact(),
        "L2": L2LearnedDistill(hsd, method=l2_method, keep_frac=keep_frac),
        "L3": L3Rewrite(intensity=l3_intensity, seed=seed, hsd=hsd, keep_frac=keep_frac),
    }


__all__ = [
    "Privatizer",
    "RawPassthrough",
    "L1Redact",
    "L2LearnedDistill",
    "L3Rewrite",
    "learned_levels",
]
