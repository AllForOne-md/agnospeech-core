"""Runtime scaling benchmarks that complement benchmark_timing.py.

Three modes, same builders/hyperparameters as compute_results.py:

  fullcorpus   wall clock to privatize EVERY post of each corpus (fast engine's
               practitioner number: "how long for my whole dataset?")
  length       ms/post by post length (token buckets) — occlusion L2 is O(tokens)
               transformer calls, so zeroshot should scale ~linearly; fast ~flat
  batch        zeroshot only: L2 cost vs the HSD head's batch_size (GPU utilization)

Always records peak RSS and (if CUDA) peak GPU memory. Writes
results/scaling_<mode>__<engine>[__<label>].json.

    python scripts/benchmark_scaling.py --mode fullcorpus --engine fast
    python scripts/benchmark_scaling.py --mode length --engine zeroshot --label t4
    python scripts/benchmark_scaling.py --mode batch --engine zeroshot --label t4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_results import ENGINES, _levels  # noqa: E402
from agnospeech.datasets import RedditCorpus  # noqa: E402
from agnospeech.harness import (  # noqa: E402
    _build_anchor_cache, _build_hsd, _index_split, _two_class_train,
    split_by_author,
)
from agnospeech.repro import GLOBAL_SEED, pin  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REAL_AUTHOR_MIN = 5
BUCKETS = [(1, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 10_000)]
BUCKET_CAP = {"fast": 200, "zeroshot": 12}
BATCH_SIZES = [16, 64, 128]
BATCH_POSTS = 30


def _mem() -> dict:
    out = {}
    try:
        import psutil
        out["peak_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20)
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    except ImportError:
        pass
    return out


def _fitted(corpus: str, engine: str, seed: int):
    cfg = ENGINES[engine]
    posts = RedditCorpus(str(ROOT / "data" / f"{corpus}.csv")).load()
    has_priv = len(RedditCorpus.authors(posts)) >= REAL_AUTHOR_MIN
    train, test = (split_by_author(posts, seed=seed) if has_priv
                   else _index_split(posts, seed=seed))
    assert _two_class_train(train), f"{corpus}: single-class train"
    hsd = _build_hsd(cfg["hsd"], seed)
    hsd.fit([p.text for p in train], [p.label for p in train])
    return cfg, posts, test, hsd


def mode_fullcorpus(corpora: list[str], engine: str, seed: int) -> dict:
    rows = {}
    for corpus in corpora:
        cfg, posts, _, hsd = _fitted(corpus, engine, seed)
        texts = [p.text for p in posts]
        t0 = time.perf_counter()
        cache = _build_anchor_cache(hsd, texts)
        t_anchor = time.perf_counter() - t0
        levels = _levels(hsd, cfg["l1"], cfg["l2_method"], cache)
        r = {"n_posts": len(texts), "anchor_cache_s": round(t_anchor, 2)}
        for code in ("L1", "L2", "L3"):
            t0 = time.perf_counter()
            levels[code].apply_many(texts)
            r[f"{code}_s"] = round(time.perf_counter() - t0, 2)
        r["total_L2_pipeline_s"] = round(r["L2_s"], 2)
        r["total_L3_pipeline_s"] = round(r["anchor_cache_s"] + r["L3_s"], 2)
        rows[corpus] = r
        print(f"  [{corpus}] n={len(texts)} L1={r['L1_s']}s L2={r['L2_s']}s "
              f"L3={r['L3_s']}s anchor={r['anchor_cache_s']}s", flush=True)
    return rows


def mode_length(corpora: list[str], engine: str, seed: int) -> dict:
    cap = BUCKET_CAP[engine]
    rows = []
    for corpus in corpora:
        cfg, _, test, hsd = _fitted(corpus, engine, seed)
        levels = _levels(hsd, cfg["l1"], cfg["l2_method"], {})
        pool = [p.text for p in test]
        rng = np.random.default_rng(seed)
        for lo, hi in BUCKETS:
            sub = [t for t in pool if lo <= len(t.split()) < hi]
            if len(sub) < 3:
                continue
            sub = [sub[i] for i in rng.permutation(len(sub))[:cap]]
            levels["L2"].apply_many(sub[:2])  # warm
            row = {"corpus": corpus, "bucket": f"{lo}-{hi if hi < 10_000 else '+'}",
                   "n": len(sub),
                   "mean_tokens": round(float(np.mean([len(t.split()) for t in sub])), 1)}
            for code in ("L1", "L2"):
                t0 = time.perf_counter()
                levels[code].apply_many(sub)
                row[f"{code}_ms_per_post"] = round(
                    (time.perf_counter() - t0) / len(sub) * 1000, 2)
            rows.append(row)
            print(f"  [{corpus} {row['bucket']} tok] n={row['n']} "
                  f"L2={row['L2_ms_per_post']} ms/post", flush=True)
    # linear fit of L2 cost vs tokens (the O(tokens) claim, measured)
    xs = [r["mean_tokens"] for r in rows]; ys = [r["L2_ms_per_post"] for r in rows]
    fit = {}
    if len(rows) >= 3:
        b, a = np.polyfit(xs, ys, 1)
        pred = np.polyval([b, a], xs)
        ss = 1 - np.sum((np.array(ys) - pred) ** 2) / np.sum((ys - np.mean(ys)) ** 2)
        fit = {"L2_ms_intercept": round(float(a), 2),
               "L2_ms_per_token": round(float(b), 3), "r2": round(float(ss), 3)}
        print(f"  fit: L2 ms/post ~= {fit['L2_ms_intercept']} + "
              f"{fit['L2_ms_per_token']}*tokens (R2 {fit['r2']})", flush=True)
    return {"buckets": rows, "fit": fit}


def mode_batch(corpora: list[str], engine: str, seed: int) -> dict:
    assert engine == "zeroshot", "batch probe only makes sense for the transformer head"
    corpus = corpora[0]
    cfg, _, test, hsd = _fitted(corpus, engine, seed)
    texts = [p.text for p in test][:BATCH_POSTS]
    rows = []
    for bs in BATCH_SIZES:
        hsd.batch_size = bs
        levels = _levels(hsd, cfg["l1"], cfg["l2_method"], {})
        levels["L2"].apply_many(texts[:2])  # warm
        t0 = time.perf_counter()
        levels["L2"].apply_many(texts)
        ms = (time.perf_counter() - t0) / len(texts) * 1000
        rows.append({"batch_size": bs, "L2_ms_per_post": round(ms, 2),
                     "corpus": corpus, "n": len(texts)})
        print(f"  batch_size={bs}: L2 {ms:.1f} ms/post", flush=True)
    return {"rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fullcorpus", "length", "batch"], required=True)
    ap.add_argument("--engine", choices=["fast", "zeroshot"], default="fast")
    ap.add_argument("--corpora", nargs="+",
                    default=["reddit_25", "reddit_50", "twitter_10", "hatexplain_twitter"])
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--label", default="", help="machine label for the output filename")
    a = ap.parse_args()
    pin(a.seed)
    corpora = [c for c in a.corpora if (ROOT / "data" / f"{c}.csv").exists()]
    t0 = time.perf_counter()
    data = {"fullcorpus": mode_fullcorpus, "length": mode_length,
            "batch": mode_batch}[a.mode](corpora, a.engine, a.seed)
    result = {"mode": a.mode, "engine": a.engine, "seed": a.seed,
              "corpora": corpora, "data": data, "memory": _mem(),
              "wall_s": round(time.perf_counter() - t0, 1)}
    RES.mkdir(exist_ok=True)
    name = f"scaling_{a.mode}__{a.engine}" + (f"__{a.label}" if a.label else "")
    (RES / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(f"-> results/{name}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
