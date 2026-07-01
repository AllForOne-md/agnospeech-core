# AgnoSpeech

The AgnoSpeech privatization mechanism for privacy-preserving hate-speech detection (the PrivHSD
task). Raw text goes in, privatized text comes out, such that a hate-speech detector still fires on
the output while an authorship attacker's accuracy drops. The mechanism is trained/grounded on data
but does not rely on the exact data when run on a new dataset.

This repository contains the L1/L2/L3 mechanism, the two engines that run it, the reddit board
corpus, and the best-configuration privatized data files.

An HTML overview of the mechanism, the results, and 100 sample privatizations per corpus (Reddit
and Twitter) is in [`results.html`](results.html) — open it in a browser.

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
authorship-attack and a trade-off (TO) number. Baseline L0: worst-case attack 0.462, utility 1.00,
TO 0. All numbers are on the full held-out test split (n=346). Utility is HSD macro-F1 (majority-corrected) relative to raw; privacy is the worst-case
authorship-attribution accuracy over a static and an adaptive attacker (lower is better); TO =
utility_ratio − privacy_ratio.

| Configuration | Engine | Utility ratio | Worst-case attack | TO | What it does |
| --- | --- | --- | --- | --- | --- |
| L1 (redact) | fast | 1.00 | 0.462 | 0.000 | identifier redaction only; no privacy gain |
| **L2 (distill)** | **fast** | **1.00** | **0.222** | **0.519** | keep only harm-salient spans; best privacy and TO, zero utility loss |
| L3 (rewrite) | fast | 1.00 | 0.364 | 0.213 | reworded, readable; weaker privacy than L2 |
| L1 (redact) | zeroshot | 0.859 | 0.399 | -0.003 | GLiNER NER redaction; costs utility, no privacy gain |
| **L2 (distill)** | **zeroshot** | **1.00** | **0.269** | **0.419** | occlusion-distill; best privacy at zero utility cost |
| L3 (rewrite) | zeroshot | 1.00 | 0.356 | 0.231 | reworded, readable |

**Best configuration: L2 (distill), on either engine.**

- **`fast` @ L2** is the standout: it halves the worst-case attacker (0.462 → 0.222) at **zero**
  utility loss (utility ratio 1.00), for the best trade-off (TO 0.519).
- **`zeroshot` @ L2** gives near-equal privacy (attack 0.269, TO 0.419), also at zero utility loss,
  plus the lowest cross-corpus hate leakage.

L1 alone is not a viable privacy configuration (it barely moves the attacker). L3 is the
readability-preserving fallback, not the privacy optimum. Across all six corpora both engines leave
only **0–7%** of hate un-privatized at L2, so the mechanism generalizes across datasets rather than
fitting a single one.

### Expanded corpus: reddit_50 (50 authors, n=539)

The same run on the larger 50-author corpus. L2 stays the best configuration and the effect is
stronger (more candidate authors, chance 0.02): `fast @ L2` cuts the attacker from 0.379 to 0.148
(TO 0.571); `zeroshot @ L2` to 0.187 (TO 0.504), both at essentially no utility loss.

| Configuration | Engine | Utility | Attack | TO |
| --- | --- | --- | --- | --- |
| L1 redact | fast | 1.00 | 0.379 | 0.000 |
| **L2 distill** | fast | 0.963 | **0.148** | **0.571** |
| L3 rewrite | fast | 1.00 | 0.269 | 0.289 |
| L1 redact | zeroshot | 0.797 | 0.317 | -0.041 |
| **L2 distill** | zeroshot | 0.999 | **0.187** | **0.504** |
| L3 rewrite | zeroshot | 1.00 | 0.282 | 0.255 |

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

The privatized text for every corpus, engine, and level, named `<corpus>_L<level>_<engine>.csv`
(schema `id,author,label,text`, where `text` is the privatized post at that level). **L0** is the raw
text (the `data/*.csv` originals for the reddit corpora); **L1** = redact, **L2** = distill
(recommended), **L3** = rewrite. Engines: `fast`, `zeroshot`. All three levels are provided for each
corpus and engine.

| corpus | posts | author axis |
| --- | --- | --- |
| `reddit_25` | 346 (full test split) | yes — 25 pseudonymous authors |
| `reddit_50` | 539 (full test split) | yes — 50 pseudonymous authors |
| `hatexplain_twitter` | 500 (Twitter only) | no — utility only |

Example: `reddit_50_L2_zeroshot.csv` is the L2 (distill) output of the zeroshot engine on the
50-author Reddit corpus. The reddit corpora carry real (pseudonymous) author ids, so they support an
authorship-attribution evaluation; the Twitter files carry hate labels only (Gab rows dropped). These
are the full held-out test splits the configuration numbers above were computed on, generated on a
Colab T4 GPU.

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
results.html        browser overview: results + 100 sample privatizations per corpus
```

## License

MIT. See `LICENSE`.
