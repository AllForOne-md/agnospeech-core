# AgnoSpeech

Anonymous submission repository for the WOAH 2026 submission: *Introducing the Privacy-HSD Trade-off: Hate Speech Detection, but not at the Cost of Privacy*

This repository contains the `AgnoSpeech` privatization mechanism for privacy-preserving hate-speech detection.The mechanism is trained/grounded on a selected dataset but does not strictly rely on the exact data when run on a new dataset.

This repository contains the L1/L2/L3 mechanism, the two engines that run it, the datasets as used in the paper, and the best-configuration privatized data files.

## The mechanism

A four-position dial applied to each post:

- **L0** raw text (reference).
- **L1 Redact** replace direct identifiers with typed placeholders (PII regex; optional NER).
- **L2 Distill** keep only the detector-salient harm spans, drop the rest (`L2LearnedDistill`).
- **L3 Rewrite** detector-anchored local rewrite: harm spans pass through, the rest is canonicalized
  with WordNet/NLTK style transforms (`L3Rewrite`).

Two **engines** run the same dial:

- **`fast`** TF-IDF + logistic-regression HSD head (refit per corpus), regex L1, linear-attribution
  L2. All-sklearn, no downloads, minutes on CPU.
- **`performance`** `cardiffnlp/twitter-roberta-base-hate-latest` HSD head, GLiNER NER L1, occlusion L2.
  No corpus-fit vocabulary; GPU-suited (multi-hour on CPU).

## Install

```bash
pip install -e .                 # fast engine (all-sklearn), plus nltk for L3
python -m nltk.downloader stopwords wordnet omw-1.4

pip install -e ".[results]"      # add the zero-shot transformer engine
python -m spacy download en_core_web_sm
```

## Run

```bash
python scripts/compute_results.py --engine fast              # all corpora, minutes, no downloads
python scripts/compute_results.py --engine performance          # GPU-suited
python scripts/compute_results.py --engine both --corpora reddit_25 hatexplain
```

Each run writes per-level metrics to `results/<corpus>__<engine>.json` and per-post L0→L3 text to
`results/outputs/<corpus>__<engine>.csv`. To privatize an arbitrary unlabeled `ID,text` CSV with the
same mechanism, use `scripts/privatize_submission.py`.

Integrity check (the hard-rule assertions):

```bash
python -m agnospeech.cli conformance
```

## Privatized data files (`privatized/`)

The privatized text for every corpus, engine, and level, named `<corpus>_L<level>_<engine>.csv`
(schema `id,author,label,text`, where `text` is the privatized post at that level). **L0** is the raw
text (the `data/*.csv` originals for the reddit corpora); **L1** = redact, **L2** = distill
(recommended), **L3** = rewrite. Engines: `fast`, `performance`. All three levels are provided for each
corpus and engine.

**Row correspondence.** Each file is **one row per input post, in the same order as the input
`data/*.csv`** — a 1:1 line-for-line correspondence with the input, and every row also carries its
original `id` so it can be joined back regardless of order. (Earlier revisions shipped only the
shuffled held-out test split, which broke that correspondence; this is fixed.) The raw inputs (L0)
ship alongside for row-by-row diffing: `data/reddit_25.csv`, `data/reddit_50.csv`,
`data/hatexplain_twitter.csv`, and `data/twitter_10.csv`.

| corpus | posts (fast) | posts (performance) | author axis |
| --- | --- | --- | --- |
| `reddit_25` | 1154 (full) | 1154 (full) | yes — 25 pseudonymous authors |
| `reddit_50` | 1795 (full) | 1795 (full) | yes — 50 pseudonymous authors |
| `twitter_10` | 6792 (full) | 6792 (full) | yes — 10 pseudonymous authors |
| `hatexplain_twitter` | 9027 (full) | 500 (first 500) | no — utility only |

`twitter_10` is a 10-author Twitter corpus that **does** carry author identities, so both engines
cover it in full and it supports the authorship-attribution evaluation. The utility-only
`hatexplain_twitter` corpus (no real authors) is covered in full by `fast`; its `performance` output is
the **first 500 posts** in input order — a representative slice, since a utility-only track needs no
full occlusion sweep. All rows remain 1:1 with the corresponding input rows.

Example: `reddit_50_L2_performance.csv` is the L2 (distill) output of the performance engine on the
50-author Reddit corpus. The reddit corpora carry real (pseudonymous) author ids, so they support an
authorship-attribution evaluation; the Twitter files carry hate labels only (Gab rows dropped). The
configuration (TO) numbers above were computed on a held-out test partition of these rows. 

## Metric and honesty bounds

- **TO** = (utility_protected / utility_original) − (privacy_protected / privacy_original), in
  [−1, 1]. Structurally the relative-gain metric of Meisenbacher et al. (Findings of NAACL 2025).
- **Privacy** is closed-world N-candidate authorship attribution — an upper-bias (optimistic)
  estimate; reported worst-case over a static and an adaptive attacker.
- **L3 is "DP-grounded", not differentially private.** Its intensity knob is an epsilon-flavored
  proxy, not a calibrated budget.
- Utility is binary HSD macro-F1 with majority-class correction.

## Layout

```
src/agnospeech/     the package (privatize, detect, attacks, metrics, datasets, harness, conformance)
scripts/            compute_results.py, fetch_corpora.py, fetch_hatexplain.py, privatize_submission.py
data/               reddit_25.csv, reddit_50.csv, hatexplain_twitter.csv (the L0 inputs; other corpora fetched)
privatized/         the best-config (L2) privatized data files
results.html        browser overview: results + 100 sample privatizations per corpus
```

## License

MIT. See `LICENSE`.
