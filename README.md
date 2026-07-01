# AgnoSpeech (core)

A minimal, lexicon-free implementation of the AgnoSpeech privatization mechanism for
privacy-preserving hate-speech detection (the PrivHSD task). Raw text goes in, privatized text
comes out, such that a hate-speech detector still fires on the output while an authorship
attacker's accuracy drops. There is **no hand-curated wordlist in any code path**: the mechanism is
trained/grounded on data but does not rely on the exact data when run on a new dataset.

This repository is the trimmed research build: the L1/L2/L3 mechanism, the two engines that run it,
the reddit board corpus, and the best-configuration privatized data files. It omits the demo/web
layer, the interactive harm-preservation check, notebooks, and experimental drop-ins.

## The mechanism

A four-position dial applied to each post:

- **L0** raw text (reference).
- **L1 Redact** replace direct identifiers with typed placeholders (PII regex; optional Presidio +
  zero-shot GLiNER NER).
- **L2 Distill** keep only the detector-salient harm spans, drop the rest (`L2LearnedDistill`).
- **L3 Rewrite** detector-anchored local rewrite: harm spans pass through, the rest is canonicalized
  with WordNet/NLTK style transforms (`L3Rewrite`, non-DP, "DP-grounded").

Two **engines** run the same dial:

- **`fast`** TF-IDF + logistic-regression HSD head (refit per corpus), regex L1, linear-attribution
  L2. All-sklearn, no downloads, minutes on CPU.
- **`zeroshot`** `cardiffnlp/twitter-roberta-base-hate-latest` HSD head, GLiNER NER L1, occlusion L2.
  No corpus-fit vocabulary; GPU-suited (multi-hour on CPU).

## Configurations, and which worked best

The two axes are the **privacy level** (L1 / L2 / L3) and the **engine** (fast / zeroshot). Numbers
below are on `reddit_25`, the only corpus with real author identities, so the only one with an
authorship-attack and a trade-off (TO) number. Baseline L0: worst-case attack 0.456, utility 1.00,
TO 0. Utility is HSD macro-F1 (majority-corrected) relative to raw; privacy is the worst-case
authorship-attribution accuracy over a static and an adaptive attacker (lower is better); TO =
utility_ratio − privacy_ratio.

| Configuration | Engine | Utility ratio | Worst-case attack | TO | What it does |
| --- | --- | --- | --- | --- | --- |
| L1 (redact) | fast | 1.00 | 0.452 | 0.009 | identifier redaction only; almost no privacy gain |
| **L2 (distill)** | **fast** | **0.935** | **0.228** | **0.435** | keep only harm-salient spans; best privacy and TO |
| L3 (rewrite) | fast | 0.938 | 0.344 | 0.184 | reworded, readable; weaker privacy than L2 |
| L1 (redact) | zeroshot | 0.929 | 0.380 | 0.096 | GLiNER NER redaction |
| **L2 (distill)** | **zeroshot** | **1.00** | **0.264** | **0.421** | occlusion-distill; best privacy at zero utility cost |
| L3 (rewrite) | zeroshot | 1.00 | 0.336 | 0.263 | reworded, readable |

**Best configuration: L2 (distill), on either engine.**

- **`fast` @ L2** gives the strongest privacy (worst-case attack 0.228, TO 0.435) for a ~6.5% utility
  cost.
- **`zeroshot` @ L2** gives near-equal privacy (0.264, TO 0.421) at **zero** utility loss, plus the
  lowest cross-corpus hate leakage.

L1 alone is not a viable privacy configuration (it barely moves the attacker). L3 is the
readability-preserving fallback, not the privacy optimum. Across all six corpora both engines leave
only **0–7%** of hate un-privatized at L2, versus **37–54%** for the earlier hand-lexicon baseline
that this build replaced — the generalization result behind the "lexicon-free" claim.

## Install

```bash
pip install -e .                 # fast engine (all-sklearn), plus nltk for L3
python -m nltk.downloader stopwords wordnet omw-1.4

pip install -e ".[results]"      # add the zero-shot transformer engine
python -m spacy download en_core_web_sm
```

## Run

```bash
# reddit board corpus is included; other corpora are fetched on demand:
python scripts/fetch_corpora.py                 # hatecheck, toxigen, civilcomments
python scripts/fetch_hatexplain.py              # hatexplain (Twitter + Gab)

python scripts/compute_results.py --engine fast              # all corpora, minutes, no downloads
python scripts/compute_results.py --engine zeroshot          # GPU-suited
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

The recommended **L2** configuration, applied to `reddit_25` and to `hatexplain` (the Twitter/Gab
corpus), for both engines. Schema: `id,author,label,text`, where `text` is the L2-privatized post.

| file | corpus | engine | author axis |
| --- | --- | --- | --- |
| `reddit_25_L2_fast.csv` | Reddit board corpus | fast | yes (25 pseudonymous authors) |
| `reddit_25_L2_zeroshot.csv` | Reddit board corpus | zeroshot | yes |
| `hatexplain_L2_fast.csv` | HateXplain (Twitter + Gab) | fast | no (utility only) |
| `hatexplain_L2_zeroshot.csv` | HateXplain (Twitter + Gab) | zeroshot | no (utility only) |

`reddit_25` carries real (pseudonymous) author ids, so it supports an authorship-attribution privacy
evaluation. HateXplain carries only hate labels (a single dummy author), so its files are for the
utility (hate-detectability) side; a Twitter-only subset is obtainable by filtering ids ending
`_twitter`.

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
data/               reddit_25.csv (the rest are fetched)
privatized/         the best-config (L2) privatized data files
```

## License

MIT. See `LICENSE`.
