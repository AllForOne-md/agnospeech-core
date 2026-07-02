"""Regenerate the FULL privatized data files, 1:1 with the input.

For a (corpus, engine) pair this privatizes EVERY post of data/<corpus>.csv,
one output row per input row, in the SAME order as the input file, and writes
privatized/<corpus>_L{1,2,3}_<engine>.csv with schema id,author,label,text.

The HSD head is grounded exactly like scripts/compute_results.py (same seed and
the same author/index train split), so the rows that were previously shipped as
the held-out test split come out byte-identical; the file just now covers the
whole corpus in input order instead of a shuffled subset.

    python scripts/regen_full_privatized.py --engine fast     --corpora reddit_25 reddit_50 hatexplain_twitter
    python scripts/regen_full_privatized.py --engine zeroshot --corpora reddit_25 reddit_50 hatexplain_twitter
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
csv.field_size_limit(10_000_000)

from agnospeech.datasets import RedditCorpus  # noqa: E402
from agnospeech.harness import (  # noqa: E402
    _build_anchor_cache, _build_hsd, _build_l1, _index_split, split_by_author,
)
from agnospeech.privatize import L2LearnedDistill, L3Rewrite  # noqa: E402
from agnospeech.repro import GLOBAL_SEED, pin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENGINES = {
    "fast": dict(hsd="tfidf", l1="regex", l2_method="linear"),
    "zeroshot": dict(hsd="cardiff", l1="gliner", l2_method="occlusion"),
}
REAL_AUTHOR_MIN = 5   # matches compute_results: >= this many authors => author split


def regen(corpus: str, engine: str, seed: int) -> None:
    cfg = ENGINES[engine]
    path = ROOT / "data" / f"{corpus}.csv"
    if not path.exists():
        print(f"  [{corpus}/{engine}] missing {path}, skip"); return
    posts = RedditCorpus(str(path)).load()                       # input file order
    authors = RedditCorpus.authors(posts)
    has_priv = len(authors) >= REAL_AUTHOR_MIN

    # Ground the head exactly like compute_results: fit on the train split.
    train, _ = (split_by_author(posts, seed=seed) if has_priv
                else _index_split(posts, seed=seed))
    t0 = time.time()
    hsd = _build_hsd(cfg["hsd"], seed).fit([p.text for p in train],
                                           [p.label for p in train])
    all_texts = [p.text for p in posts]
    cache = _build_anchor_cache(hsd, all_texts)                  # {text -> harm words}
    levels = {
        "L1": _build_l1(cfg["l1"]),
        "L2": L2LearnedDistill(hsd, method=cfg["l2_method"], keep_frac=0.6),
        "L3": L3Rewrite(intensity=0.6, seed=seed, hsd=hsd, anchor_words=cache),
    }
    for code, lv in levels.items():
        out = lv.apply_many(all_texts)
        dst = ROOT / "privatized" / f"{corpus}_L{code[1]}_{engine}.csv"
        with open(dst, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "author", "label", "text"])
            for p, o in zip(posts, out):
                w.writerow([p.id, p.author, p.label, o])
        print(f"    wrote {dst.name}: {len(posts)} rows")
    print(f"  [{corpus}/{engine}] {len(posts)} rows, priv_axis={has_priv}, "
          f"{time.time() - t0:.0f}s", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["fast", "zeroshot"], required=True)
    ap.add_argument("--corpora", nargs="+",
                    default=["reddit_25", "reddit_50", "hatexplain_twitter"])
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    a = ap.parse_args(argv)
    pin(a.seed)
    (ROOT / "privatized").mkdir(exist_ok=True)
    for corpus in a.corpora:
        regen(corpus, a.engine, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
