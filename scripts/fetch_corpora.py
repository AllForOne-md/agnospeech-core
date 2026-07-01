"""Fetch the disjoint-from-cardiff hate corpora for the cross-dataset de-fixing test.

    python scripts/fetch_corpora.py            # all three
    python scripts/fetch_corpora.py --only hatecheck

Writes data/<name>.csv (ID, author[dummy], text, hs) for:
- hatecheck     : HateCheck functional suite, CC-BY-4.0, direct CSV from GitHub.
- toxigen       : ToxiGen annotated (human labels), MIT, via HF datasets.
- civilcomments : Civil Comments, CC0, streamed sample via HF datasets.

All three are OUTSIDE cardiff's 13-dataset training union (see the KB web-clip
2026-06-18-cross-dataset-corpus-options). Files are gitignored; regenerate here.
hs convention: HateCheck label_gold==hateful; ToxiGen toxicity_human>=3;
Civil Comments toxicity>=0.5.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import pathlib
import random
import ssl
import urllib.request

SEED = 20260617
HATECHECK_URL = ("https://raw.githubusercontent.com/paul-rottger/hatecheck-data/"
                 "main/test_suite_cases.csv")


def _write(name: str, rows: list[tuple]) -> None:
    pathlib.Path("data").mkdir(exist_ok=True)
    with open(f"data/{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "author", "text", "hs"])
        w.writerows(rows)
    c = collections.Counter(r[3] for r in rows)
    n = max(1, len(rows))
    print(f"wrote data/{name}.csv rows={len(rows)} hate={c[1]} nonhate={c[0]} "
          f"rate={c[1] / n:.3f}")


def fetch_hatecheck() -> None:
    with urllib.request.urlopen(HATECHECK_URL, timeout=120,
                                context=ssl.create_default_context()) as r:
        raw = r.read().decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(raw)):
        text = (row.get("test_case") or "").strip()
        gold = (row.get("label_gold") or "").strip().lower()
        if not text or gold not in ("hateful", "non-hateful"):
            continue
        rows.append((row.get("case_id") or f"hc{len(rows)}", "hc", text,
                     1 if gold == "hateful" else 0))
    _write("hatecheck", rows)


def fetch_toxigen() -> None:
    from datasets import load_dataset
    ds = load_dataset("toxigen/toxigen-data", name="annotated", split="train")
    rows = []
    for i, ex in enumerate(ds):
        text = (ex.get("text") or "").strip()
        th = ex.get("toxicity_human")
        if not text or th is None:
            continue
        rows.append((f"tg{i}", "tg", text, 1 if float(th) >= 3.0 else 0))
    _write("toxigen", rows)


def fetch_civilcomments(n_stream: int = 30000, cap: int = 6000) -> None:
    from datasets import load_dataset
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if i >= n_stream:
            break
        text = (ex.get("text") or "").strip()
        tox = ex.get("toxicity")
        if not text or tox is None:
            continue
        rows.append((f"cc{i}", "cc", text, 1 if float(tox) >= 0.5 else 0))
    rng = random.Random(SEED)
    tox = [r for r in rows if r[3] == 1]
    non = [r for r in rows if r[3] == 0]
    rng.shuffle(non)
    out = tox + non[: max(len(tox) * 2, 1500)]
    rng.shuffle(out)
    _write("civilcomments", out[:cap])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["hatecheck", "toxigen", "civilcomments"])
    a = ap.parse_args()
    targets = [a.only] if a.only else ["hatecheck", "toxigen", "civilcomments"]
    for t in targets:
        try:
            {"hatecheck": fetch_hatecheck, "toxigen": fetch_toxigen,
             "civilcomments": fetch_civilcomments}[t]()
        except Exception as e:  # noqa: BLE001 - surface per-corpus failure, continue
            print(f"{t} FAILED: {type(e).__name__}: {str(e)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
