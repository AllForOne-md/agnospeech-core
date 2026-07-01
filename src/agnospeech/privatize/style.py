"""De-fixed L3 stylistic resources: prebuilt NLTK stopwords + WordNet canonicalization.

L3Rewrite's hand lists (``SYNONYMS``, ``_FUNCTION``) are stylistic, not the harm
anchor, but they are still hand-written. This swaps them for prebuilt,
corpus-independent resources:

- function words -> NLTK ``stopwords`` (English), a fixed published list.
- lexical canonicalization -> WordNet dominant-synset lemma: replace a content
  word with the canonical lemma of its most common sense, erasing an author's
  idiosyncratic word choice with no hand map.

This de-fixes ONLY the stylistic layer; it does NOT touch the harm anchor, so it
must pair with the detector-grounded (learned) anchor to be fully lexicon-free.

Optional dep: ``pip install nltk`` plus the ``stopwords`` / ``wordnet`` / ``omw-1.4``
corpora (``python -m nltk.downloader stopwords wordnet omw-1.4``). ``available()``
guards it; L3 falls back to the hand lists when NLTK or its corpora are absent.
Version is pinned for reproducibility: NLTK 3.9.x, WordNet 3.0 (via omw-1.4).
"""

from __future__ import annotations

import functools

NLTK_PINNED = "nltk>=3.9,<3.10 ; wordnet via omw-1.4"


def available() -> bool:
    try:
        from nltk.corpus import stopwords, wordnet
        stopwords.words("english")
        wordnet.synsets("test")
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def function_words() -> frozenset[str]:
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("english"))


@functools.lru_cache(maxsize=20000)
def wordnet_canonical(word: str) -> str:
    """Canonicalize a stylistic MODIFIER to its dominant synonym, meaning-preserving.

    Without sentence-level POS, WordNet synonym substitution distorts nouns/verbs
    (e.g. the verb "said" -> "state"/"aforesaid", number/tense breaks), so this is
    deliberately restricted to ADJECTIVES and ADVERBS, where a synonym swap is
    grammatically safe ("really" -> "truly", "huge" stays "huge"):

    - skip unless the word's DOMINANT sense (synsets[0]) is an adjective/adverb, so
      a word whose primary use is a noun/verb ("said", "buy", "kids") is left alone;
    - require the surface word to be a member of the chosen synset;
    - return the word unchanged on a multi-word lemma or when canonical == input.

    A stylistic normalizer, not a paraphraser. The harm anchor is never touched
    (this runs only on non-anchor tokens in L3).
    """
    from nltk.corpus import wordnet
    low = word.lower()
    if len(low) < 3 or not low.isalpha():
        return word
    syns = wordnet.synsets(low)
    _MOD = {"a", "s", "r"}  # adjective, adjective-satellite, adverb
    if not syns or syns[0].pos() not in _MOD:
        return word
    for syn in syns:
        if syn.pos() not in _MOD:
            continue
        names = {lemma.name().lower() for lemma in syn.lemmas()}
        if low not in names:
            continue
        cand = syn.lemmas()[0].name()
        if "_" in cand or cand == low or not cand.isalpha():
            return word
        return cand
    return word
