"""Privatize an UNLABELED benchmark / submission CSV (ID + text) through the
AgnoSpeech pipeline, emitting ID + privatized-text CSV(s) ready to submit.

The privhsd board scores HS_accuracy - Author_accuracy on the server's HIDDEN
labels, so no local TO can be computed here; this only APPLIES the privatizer and
preserves the row IDs so the server can map back to its labels. The detector-grounded
L2 / L3 are grounded on a labeled hate corpus (``--ground``, default reddit_25) and
applied UNCHANGED to the target text - the de-fixing cross-dataset transfer setting,
so the kept harm spans follow the grounding detector's coverage (a Twitter-native
detector would align better; reddit_25 is the fast, no-download default).

    python scripts/privatize_submission.py --in data.csv --level L2
    python scripts/privatize_submission.py --in data.csv --all     # L1 + L2 + L3 files

Output mirrors the input format: columns ``ID,text`` with text = privatized text,
same IDs, same order, every row (empty text passes through as empty).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agnospeech.detect import HsdHead  # noqa: E402
from agnospeech.privatize import L1Redact, L2LearnedDistill, L3Rewrite  # noqa: E402
from agnospeech.repro import GLOBAL_SEED, pin  # noqa: E402

csv.field_size_limit(10_000_000)
_TEXT_KEYS = ("text", "tweet", "comment", "body", "content", "message", "post")
_LABEL_KEYS = ("hs", "label", "hate", "hateful", "toxic", "class", "y")
_ID_KEYS = ("id",)


def _find(cols, keys):
    low = {c.lower().strip(): c for c in cols if c}
    for k in keys:
        if k in low:
            return low[k]
    return None


def _load_target(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    cols = list(rows[0].keys())
    tc = _find(cols, _TEXT_KEYS)
    idc = _find(cols, _ID_KEYS)
    if not tc:
        raise SystemExit(f"{path}: no text column found in {cols}")
    out = []
    for i, r in enumerate(rows):
        rid = str(r.get(idc)) if idc and r.get(idc) is not None else f"row_{i}"
        out.append((rid, (r.get(tc) or "").strip()))
    return out


def _ground_head(path, seed):
    """Fit an HsdHead on a labeled hate corpus (needs text + a 0/1 label column;
    author column optional, unlike RedditCorpus.load)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    cols = list(rows[0].keys()) if rows else []
    tc, lc = _find(cols, _TEXT_KEYS), _find(cols, _LABEL_KEYS)
    if not tc or not lc:
        raise SystemExit(f"{path}: grounding corpus needs text + label columns "
                         f"(found text={tc}, label={lc} in {cols})")
    texts, labels = [], []
    for r in rows:
        t = (r.get(tc) or "").strip()
        try:
            y = int(float(r.get(lc)))
        except (TypeError, ValueError):
            continue
        if t:
            texts.append(t)
            labels.append(y)
    if len({*labels}) < 2:
        raise SystemExit(f"{path}: grounding corpus is single-class")
    return HsdHead(seed=seed).fit(texts, labels)


def _privatizer(level, head, intensity, keep_frac, seed):
    if level == "L1":
        return L1Redact()
    if level == "L2":
        return L2LearnedDistill(head, keep_frac=keep_frac)
    if level == "L3":
        # Lexicon-free: detector-grounded harm anchor + WordNet/NLTK style.
        return L3Rewrite(intensity=intensity, seed=seed, hsd=head, keep_frac=keep_frac)
    raise SystemExit(f"unknown level {level!r} (L1|L2|L3)")


def _write(items, level, priv, out_path):
    privatized = [priv.apply(t) if t else "" for _id, t in items]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "text"])
        for (rid, _t), pt in zip(items, privatized):
            w.writerow([rid, pt])
    raw_len = [len(t) for _i, t in items if t]
    pv_len = [len(p) for p in privatized if p]
    changed = sum(1 for (_i, t), p in zip(items, privatized) if t and t != p)
    nonempty = sum(1 for p in privatized if p.strip())
    print(f"  {level}: wrote {out_path}")
    print(f"      {len(items)} rows | changed {changed}/{len(items)} "
          f"({changed / len(items):.0%}) | non-empty out {nonempty}/{len(items)} | "
          f"mean len {statistics.fmean(pv_len) if pv_len else 0:.0f} "
          f"(raw {statistics.fmean(raw_len) if raw_len else 0:.0f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="unlabeled CSV (ID + text)")
    ap.add_argument("--ground", default="data/reddit_25.csv",
                    help="labeled hate corpus to ground the learned L2/L3 detector on")
    ap.add_argument("--level", choices=["L1", "L2", "L3"], default="L2")
    ap.add_argument("--all", action="store_true", help="produce L1, L2 and L3 files")
    ap.add_argument("--intensity", type=float, default=0.6)
    ap.add_argument("--keep-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--out", default=None, help="output path (single level only)")
    a = ap.parse_args(argv)
    pin(a.seed)

    items = _load_target(a.inp)
    levels = ["L1", "L2", "L3"] if a.all else [a.level]
    head = _ground_head(a.ground, a.seed) if any(lv in ("L2", "L3") for lv in levels) else None
    if head is not None:
        print(f"grounded HSD head on {a.ground}; privatizing {len(items)} rows from {a.inp}")
    stem = Path(a.inp).with_suffix("")
    for lv in levels:
        priv = _privatizer(lv, head, a.intensity, a.keep_frac, a.seed)
        out = a.out if (a.out and not a.all) else f"{stem}_priv_{lv}.csv"
        _write(items, lv, priv, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
