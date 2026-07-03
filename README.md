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
(schema `id,author,label,text`, where `text` is the privatized post at that level). **L1** = redact, **L2** = distill, **L3** = rewrite. Engines: `fast`, `performance`. All three levels are provided for each corpus and engine.


## Layout

```
src/agnospeech/     the package (privatize, detect, attacks, metrics, datasets, harness, conformance)
data/               reddit_25.csv, reddit_50.csv, hatexplain_twitter.csv (the L0 inputs; other corpora fetched)
privatized/         the best-config (L2) privatized data files
```

## License

MIT. See `LICENSE`.
