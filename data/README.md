# Data

`reddit_25.csv` (the PrivHSD board corpus: `ID,author,text,hs`; 1154 posts, 25 pseudonymous author
ids `0`-`24`, hate label `1`/`0`) is included here. The other corpora are fetched on demand:

| corpus | how to get it |
| --- | --- |
| `reddit_25.csv` | included in this directory |
| `hatecheck.csv`, `toxigen.csv`, `civilcomments.csv` | `python scripts/fetch_corpora.py` |
| `hatexplain.csv` | `python scripts/fetch_hatexplain.py` (Twitter + Gab) |

Every corpus uses the same `ID,author,text,hs` schema (`hs`: 1 = hate, 0 = not). Only `reddit_25`
carries real (pseudonymous) author ids and therefore an authorship/privacy axis; the others carry a
single dummy author and are utility-only.
