"""Fetch HateXplain and write the two artifacts the cross-dataset work needs.

    python scripts/fetch_hatexplain.py

- data/hatexplain.csv          : ID, author(dummy), text, hs  (utility-track corpus
                                 the twitter and gab evaluation corpus)
- data/hatexplain_rationales.jsonl : {"tokens":[...], "bio":[...]} per hate post
                                 (token rationales; unused by the minimal build)

hs = 1 if the majority annotator label is hatespeech/offensive, else 0. The BIO
mask is the majority-vote token rationale across annotators (hate/offensive posts
only). Source: github.com/hate-alert/HateXplain (Gab + Twitter; disjoint from the
Reddit eval corpus). These files are gitignored; regenerate with this script.
"""

from __future__ import annotations

import collections
import json
import pathlib
import ssl
import urllib.request

URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"


def main() -> int:
    print("fetching HateXplain ...")
    with urllib.request.urlopen(URL, timeout=180, context=ssl.create_default_context()) as r:
        data = json.loads(r.read().decode("utf-8"))
    print(f"posts: {len(data)}")

    pathlib.Path("data").mkdir(exist_ok=True)
    csv_rows, bio_rows = [], []
    for pid, rec in data.items():
        toks = rec.get("post_tokens") or []
        text = " ".join(toks).strip()
        labs = [a["label"] for a in rec.get("annotators", [])]
        if not text or not labs:
            continue
        maj = collections.Counter(labs).most_common(1)[0][0]
        hs = 1 if maj in ("hatespeech", "offensive") else 0
        csv_rows.append((pid, "hx", text, hs))

        rats = rec.get("rationales") or []
        if maj == "normal" or not rats:
            continue
        votes = [0] * len(toks)
        for m in rats:
            for i, v in enumerate(m[: len(toks)]):
                votes[i] += int(v)
        thr = len(rats) / 2.0
        mask = [1 if votes[i] > thr else 0 for i in range(len(toks))]
        if not any(mask):
            continue
        bio, prev = [], 0
        for m in mask:
            bio.append("B" if (m and not prev) else "I" if m else "O")
            prev = m
        bio_rows.append({"tokens": toks, "bio": bio})

    import csv as _csv
    with open("data/hatexplain.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["ID", "author", "text", "hs"])
        w.writerows(csv_rows)
    with open("data/hatexplain_rationales.jsonl", "w", encoding="utf-8") as f:
        for r in bio_rows:
            f.write(json.dumps(r) + "\n")

    rate = sum(r[3] for r in csv_rows) / len(csv_rows) if csv_rows else 0.0
    print(f"wrote data/hatexplain.csv rows={len(csv_rows)} hate_rate={rate:.3f}")
    print(f"wrote data/hatexplain_rationales.jsonl seqs={len(bio_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
