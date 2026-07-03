"""Compute privatization RESULTS for later comparison.

Two engines, side by side, on each dataset:
  - fast     : TF-IDF head (refit per corpus) + learned L2 (linear) + L3 detector
               anchor + WordNet/NLTK style. No download, minutes. Adaptable per
               corpus, but the head is corpus-fit (dependency b remains).
  - performance : cardiff transformer head + learned L2 (occlusion) + L3 detector
               anchor + NLTK style. No corpus-fit vocab. Slow on
               CPU (transformer occlusion), GPU-appropriate.

For each (corpus, engine) writes results/<corpus>__<engine>.json (metrics) and
results/outputs/<corpus>__<engine>.csv (per-post L0->L3 text). Metrics: per-level
macro-F1, utility_ratio, mean length, un-privatized fraction (hate left == L1),
and -- when the corpus has real authors -- worst-case authorship attack + TO.

    python scripts/compute_results.py --engine fast              # all corpora, fast
    python scripts/compute_results.py --engine performance --cap 150
    python scripts/compute_results.py --engine both --corpora reddit_25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agnospeech.attacks import AuthorshipAttacker  # noqa: E402
from agnospeech.datasets import RedditCorpus  # noqa: E402
from agnospeech.harness import (  # noqa: E402
    _build_anchor_cache, _build_hsd, _build_l1, _index_split, _two_class_train,
    split_by_author,
)
from agnospeech.metrics import (  # noqa: E402
    macro_f1, majority_baseline_f1, privacy_ratio, to_score, utility_ratio,
)
from agnospeech.privatize import L2LearnedDistill, L3Rewrite, RawPassthrough  # noqa: E402
from agnospeech.repro import GLOBAL_SEED, pin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ENGINES = {
    "fast": dict(hsd="tfidf", l1="regex", l2_method="linear"),
    "performance": dict(hsd="cardiff", l1="gliner", l2_method="occlusion"),
}
CORPORA = ["reddit_25", "hatecheck", "toxigen", "civilcomments", "hatexplain",
           "synth_joint"]
REAL_AUTHOR_MIN = 5   # >= this many distinct authors => compute the privacy axis


def _levels(hsd, l1_kind, l2_method, anchor_cache):
    return {
        "L0": RawPassthrough(),
        "L1": _build_l1(l1_kind),
        "L2": L2LearnedDistill(hsd, method=l2_method, keep_frac=0.6),
        "L3": L3Rewrite(intensity=0.6, seed=GLOBAL_SEED, hsd=hsd,
                        anchor_words=anchor_cache),
    }


def run(corpus: str, engine: str, cap: int, seed: int) -> dict | None:
    cfg = ENGINES[engine]
    path = ROOT / "data" / f"{corpus}.csv"
    if not path.exists():
        print(f"  [{corpus}/{engine}] missing {path}, skip"); return None
    posts = RedditCorpus(str(path)).load()
    authors = RedditCorpus.authors(posts)
    has_priv = len(authors) >= REAL_AUTHOR_MIN
    train, test = (split_by_author(posts, seed=seed) if has_priv
                   else _index_split(posts, seed=seed))
    if not _two_class_train(train):
        print(f"  [{corpus}/{engine}] single-class train, skip"); return None
    if cap and len(test) > cap:
        rng = np.random.default_rng(seed)
        test = [test[i] for i in rng.permutation(len(test))[:cap]]
    tr_txt = [p.text for p in train]; tr_lab = [p.label for p in train]
    te_txt = [p.text for p in test]; te_lab = [p.label for p in test]
    te_au = [p.author for p in test]; tr_au = [p.author for p in train]

    t0 = time.time()
    hsd = _build_hsd(cfg["hsd"], seed).fit(tr_txt, tr_lab)
    f1_raw = macro_f1(te_lab, hsd.predict(te_txt)); f1_maj = majority_baseline_f1(te_lab)
    static = AuthorshipAttacker(seed=seed).fit(tr_txt, tr_au) if has_priv else None
    p_orig = static.accuracy(te_txt, te_au) if has_priv else None

    cache = _build_anchor_cache(hsd, te_txt)   # one attribution pass over the test set
    levels = _levels(hsd, cfg["l1"], cfg["l2_method"], cache)
    l1_out = levels["L1"].apply_many(te_txt)

    rows, per_post = {}, {p.id: {"author": p.author, "label": p.label} for p in test}
    for code, lv in levels.items():
        out = lv.apply_many(te_txt)
        for p, o in zip(test, out):
            per_post[p.id][code] = o
        f1 = macro_f1(te_lab, hsd.predict(out))
        ur = utility_ratio(f1, f1_raw, f1_maj)
        m = {"macro_f1": round(f1, 4), "utility_ratio": round(ur, 4),
             "mean_len_chars": round(float(np.mean([len(o) for o in out])), 1)}
        if code in ("L2", "L3"):
            hate = [i for i, p in enumerate(test) if p.label == 1]
            m["unprivatized_frac_hate"] = round(
                float(np.mean([out[i] == l1_out[i] for i in hate])) if hate else 0.0, 4)
        if has_priv:
            tr_priv = lv.apply_many(tr_txt)
            worst = max(static.accuracy(out, te_au),
                        AuthorshipAttacker(seed=seed).fit(tr_priv, tr_au).accuracy(out, te_au))
            m["attack_worst"] = round(worst, 4)
            m["to_honest"] = round(to_score(ur, privacy_ratio(worst, p_orig)), 4)
        rows[code] = m

    result = {
        "corpus": corpus, "engine": engine, "config": cfg,
        "n_train": len(train), "n_test": len(test), "n_authors": len(authors),
        "has_privacy_axis": has_priv, "test_capped_at": cap,
        "baselines": {"f1_raw": round(f1_raw, 4), "f1_majority": round(f1_maj, 4),
                      "privacy_original_acc": round(p_orig, 4) if has_priv else None},
        "levels": rows, "secs": round(time.time() - t0, 1),
    }
    RES.mkdir(exist_ok=True); (RES / "outputs").mkdir(exist_ok=True)
    (RES / f"{corpus}__{engine}.json").write_text(json.dumps(result, indent=2))
    with open(RES / "outputs" / f"{corpus}__{engine}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "author", "label", "L0", "L1", "L2", "L3"])
        for pid, d in per_post.items():
            w.writerow([pid, d["author"], d["label"], d.get("L0", ""), d.get("L1", ""),
                        d.get("L2", ""), d.get("L3", "")])
    print(f"  [{corpus}/{engine}] n_test={len(test)} priv={has_priv} "
          f"f1_raw={f1_raw:.3f} {result['secs']:.0f}s -> {corpus}__{engine}.json", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fast", "performance", "both"], default="both")
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--cap", type=int, default=250, help="cap test posts per corpus")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    a = ap.parse_args()
    pin(a.seed)
    engines = ["fast", "performance"] if a.engine == "both" else [a.engine]
    summary = []
    for engine in engines:
        print(f"== engine: {engine} ==", flush=True)
        for corpus in a.corpora:
            try:
                r = run(corpus, engine, a.cap, a.seed)
                if r:
                    summary.append({"corpus": corpus, "engine": engine,
                                    "levels": r["levels"], "has_priv": r["has_privacy_axis"]})
            except Exception as e:  # noqa: BLE001 - keep going across corpora
                print(f"  [{corpus}/{engine}] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / f"summary_{a.engine}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} result blocks under {RES}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
