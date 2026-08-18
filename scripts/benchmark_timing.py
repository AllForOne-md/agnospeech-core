"""Runtime / speed benchmark for each privacy level on each engine.

Times the SAME pipeline stages that compute_results.py runs (same builders, same
hyperparameters), per (corpus, engine):

  one-time setup            hsd_fit (train the HSD head), build_levels (construct
                            the L0-L3 ladder; for zeroshot this loads GLiNER)
  per-post stages           L1/L2/L3 apply, l3_anchor_cache (the attribution pass
                            L3 needs on new text), hsd_predict (detection)

Per-post stages are timed over --repeats runs (median reported) after a small
warmup, and reported as ms/post and posts/sec so the fast and zeroshot engines
are comparable even at different post caps. The true end-to-end L3 cost on new
text is l3_anchor_cache + L3 apply; L2's cost is the same attribution work, so
l3_anchor_cache ~= L2 apply.

Writes results/timing_<corpus>__<engine>.json per run and, per invocation,
results/timing_summary.json + results/timing_report.md (a paste-ready table).

    python scripts/benchmark_timing.py --engine fast                # minutes, CPU
    python scripts/benchmark_timing.py --engine zeroshot --cap 40   # CPU-feasible slice
    python scripts/benchmark_timing.py --engine both --corpora reddit_25
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
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
# CPU-feasible defaults: fast matches the compute_results cap; zeroshot's
# occlusion attribution is O(tokens) transformer calls per post, so keep it small.
DEFAULT_CAP = {"fast": 250, "zeroshot": 40}
DEFAULT_REPEATS = {"fast": 3, "zeroshot": 1}
WARMUP_POSTS = 3


def _machine() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def _timed(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _stage(times: list[float], n_posts: int | None) -> dict:
    med = statistics.median(times)
    d = {"secs_median": round(med, 3), "secs_min": round(min(times), 3),
         "secs_max": round(max(times), 3), "repeats": len(times)}
    if n_posts:
        d["ms_per_post"] = round(med / n_posts * 1000, 2)
        d["posts_per_sec"] = round(n_posts / med, 1) if med > 0 else None
    return d


def bench(corpus: str, engine: str, cap: int, repeats: int, seed: int,
          label: str = "") -> dict | None:
    cfg = ENGINES[engine]
    path = ROOT / "data" / f"{corpus}.csv"
    if not path.exists():
        print(f"  [{corpus}/{engine}] missing {path}, skip"); return None
    posts = RedditCorpus(str(path)).load()
    has_priv = len(RedditCorpus.authors(posts)) >= REAL_AUTHOR_MIN
    train, test = (split_by_author(posts, seed=seed) if has_priv
                   else _index_split(posts, seed=seed))
    if not _two_class_train(train):
        print(f"  [{corpus}/{engine}] single-class train, skip"); return None
    if cap and len(test) > cap:
        rng = np.random.default_rng(seed)
        test = [test[i] for i in rng.permutation(len(test))[:cap]]
    tr_txt = [p.text for p in train]; tr_lab = [p.label for p in train]
    te_txt = [p.text for p in test]
    n = len(te_txt)
    print(f"  [{corpus}/{engine}] n_test={n} repeats={repeats}", flush=True)
    stages: dict[str, dict] = {}

    # -- one-time setup (timed once: model loads / corpus fit, not per-post) --
    hsd = _build_hsd(cfg["hsd"], seed)
    stages["setup_hsd_fit"] = _stage(_timed(lambda: hsd.fit(tr_txt, tr_lab), 1), None)

    t0 = time.perf_counter()
    cache = _build_anchor_cache(hsd, te_txt[:WARMUP_POSTS])  # warms attribution path
    levels = _levels(hsd, cfg["l1"], cfg["l2_method"], cache)
    stages["setup_build_levels"] = _stage([time.perf_counter() - t0], None)

    # -- per-post stages --
    stages["l3_anchor_cache"] = _stage(
        _timed(lambda: _build_anchor_cache(hsd, te_txt), repeats), n)
    cache.update(_build_anchor_cache(hsd, te_txt))  # L3 below reuses the real cache

    for code in ("L1", "L2", "L3"):
        lv = levels[code]
        lv.apply_many(te_txt[:WARMUP_POSTS])
        stages[f"{code}_apply"] = _stage(
            _timed(lambda lv=lv: lv.apply_many(te_txt), repeats), n)
        print(f"    {code}: {stages[f'{code}_apply']['ms_per_post']} ms/post", flush=True)

    hsd.predict(te_txt[:WARMUP_POSTS])
    stages["hsd_predict"] = _stage(_timed(lambda: hsd.predict(te_txt), repeats), n)

    # End-to-end per-post cost of the deployable configs (attribution included).
    derived = {
        "L1_total_ms_per_post": stages["L1_apply"]["ms_per_post"],
        "L2_total_ms_per_post": stages["L2_apply"]["ms_per_post"],
        "L3_total_ms_per_post": round(stages["l3_anchor_cache"]["ms_per_post"]
                                      + stages["L3_apply"]["ms_per_post"], 2),
    }

    result = {
        "corpus": corpus, "engine": engine, "config": cfg, "label": label,
        "l1_class": type(levels["L1"]).__name__,
        "n_test": n, "cap": cap, "repeats": repeats, "seed": seed,
        "machine": _machine(), "stages": stages, "per_post_totals": derived,
    }
    RES.mkdir(exist_ok=True)
    stem = f"timing_{corpus}__{engine}" + (f"__{label}" if label else "")
    (RES / f"{stem}.json").write_text(json.dumps(result, indent=2))
    print(f"  [{corpus}/{engine}] -> {stem}.json", flush=True)
    return result


def _report_md(results: list[dict]) -> str:
    lines = ["# Privatization runtime benchmark", ""]
    m = results[0]["machine"]
    dev = m.get("cuda_device") if m.get("cuda_available") else "CPU"
    lines += [f"Machine: {m['processor'] or m['platform']} ({m['cpu_count']} logical cores), "
              f"Python {m['python']}, device: {dev}.", "",
              "Median wall-clock over repeated runs after warmup; ms/post normalizes "
              "across post caps. L3 total includes the attribution (anchor-cache) pass "
              "it needs on new text; L2's own cost is that same attribution work.", ""]
    lines += ["| Corpus | Engine | Level | ms/post | posts/sec | timed posts |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        for code in ("L1", "L2", "L3"):
            ms = r["per_post_totals"][f"{code}_total_ms_per_post"]
            lines.append(f"| {r['corpus']} | {r['engine']} | {code} | {ms} | "
                         f"{round(1000 / ms, 1) if ms else 'n/a'} | {r['n_test']} |")
    lines += ["", "One-time setup and detection:", "",
              "| Corpus | Engine | HSD fit (s) | Level build (s) | Detect (ms/post) |",
              "| --- | --- | --- | --- | --- |"]
    for r in results:
        s = r["stages"]
        lines.append(f"| {r['corpus']} | {r['engine']} | "
                     f"{s['setup_hsd_fit']['secs_median']} | "
                     f"{s['setup_build_levels']['secs_median']} | "
                     f"{s['hsd_predict']['ms_per_post']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["fast", "zeroshot", "both"], default="both")
    ap.add_argument("--corpora", nargs="+", default=["reddit_25"])
    ap.add_argument("--cap", type=int, default=None,
                    help="posts to time (default: 250 fast / 40 zeroshot)")
    ap.add_argument("--repeats", type=int, default=None,
                    help="timed repeats per stage (default: 3 fast / 1 zeroshot)")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--label", default="",
                    help="machine label appended to output filenames (e.g. t4, laptop)")
    a = ap.parse_args()
    pin(a.seed)
    engines = ["fast", "zeroshot"] if a.engine == "both" else [a.engine]
    results = []
    for engine in engines:
        cap = a.cap if a.cap is not None else DEFAULT_CAP[engine]
        repeats = a.repeats if a.repeats is not None else DEFAULT_REPEATS[engine]
        print(f"== engine: {engine} ==", flush=True)
        for corpus in a.corpora:
            try:
                r = bench(corpus, engine, cap, repeats, a.seed, a.label)
                if r:
                    results.append(r)
            except Exception as e:  # noqa: BLE001 - keep going across corpora
                print(f"  [{corpus}/{engine}] FAILED: {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)
    if not results:
        print("no results"); return 1
    RES.mkdir(exist_ok=True)
    (RES / "timing_summary.json").write_text(json.dumps(results, indent=2))
    (RES / "timing_report.md").write_text(_report_md(results), encoding="utf-8")
    print(f"\nwrote {len(results)} timing blocks + timing_report.md under {RES}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
