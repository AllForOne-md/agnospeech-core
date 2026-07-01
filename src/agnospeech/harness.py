"""The evaluation harness: one pass that produces every TO number honestly.

Protocol:
- Per-author 70/30 split so all N authors appear in train and test (the attacker
  needs every candidate; the detector sees a mixed split).
- ONE shared HSD head trained on raw train text, applied unchanged to every
  privatized version of the test set (honest Uo vs Up).
- Authorship attacker in two strengths: static (trained on raw) and adaptive
  (trained on the privatized version it is tested on). Adaptive is the headline.
- The L3 operating intensity is chosen by the curve's non-degenerate selector,
  then used as the L3 level everywhere else, so the curve stays consistent.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from .attacks import AuthorshipAttacker, pii_removed_fraction
from .datasets import Post, RedditCorpus
from .detect import DualHsdHead, HsdHead
from .metrics import (
    CurvePoint,
    Floors,
    board_score,
    build_curve,
    chance_corrected,
    dominance,
    dual_detector_utility,
    eraser_scores,
    length_tertile_slices,
    macro_f1,
    majority_baseline_f1,
    mean_readability,
    mean_semantic_similarity,
    paired_bootstrap,
    privacy_ratio,
    subgroup_f1,
    to_score,
    utility_ratio,
)
from .privatize import L1Redact, L2LearnedDistill, L3Rewrite, RawPassthrough

L3_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97]


def split_by_author(posts: list[Post], test_frac: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    by_author: dict[str, list[Post]] = {}
    for p in posts:
        by_author.setdefault(p.author, []).append(p)
    train, test = [], []
    for author in sorted(by_author):
        items = sorted(by_author[author], key=lambda p: p.id)
        idx = rng.permutation(len(items))
        n_test = max(1, int(round(len(items) * test_frac)))
        test_idx = set(idx[:n_test].tolist())
        for i, p in enumerate(items):
            (test if i in test_idx else train).append(p)
    return train, test


def _index_split(posts: list[Post], test_frac: float = 0.3, seed: int = 0):
    """Plain 70/30 split by index (when there are no usable author labels)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(posts))
    n_test = max(1, int(round(len(posts) * test_frac)))
    test_i = set(idx[:n_test].tolist())
    train = [p for i, p in enumerate(posts) if i not in test_i]
    test = [p for i, p in enumerate(posts) if i in test_i]
    return train, test


def _fittable(posts: list[Post]) -> bool:
    """A corpus an HsdHead can train on: enough posts and both label classes."""
    return len(posts) >= 4 and len({p.label for p in posts}) >= 2


def _two_class_train(train: list[Post]) -> bool:
    """The post-split TRAIN partition carries BOTH label classes (an unstratified
    split on a skewed corpus can strand the minority class in test, raising on fit)."""
    return len({p.label for p in train}) >= 2


def _build_hsd(kind: str, seed: int):
    """HSD head factory: 'tfidf' (spine, default) or 'cardiff' (transformer)."""
    if kind == "cardiff":
        from .detect.transformer_hsd import TransformerHsdHead
        return TransformerHsdHead(seed=seed)
    if kind != "tfidf":
        raise ValueError(f"unknown hsd kind {kind!r} (tfidf|cardiff)")
    return HsdHead(seed=seed)


def _supports_linear_attr(hsd) -> bool:
    """True if the head exposes the sklearn word channel linear attribution uses
    (the TF-IDF head); a transformer head does not (use occlusion / Captum there)."""
    return hasattr(getattr(hsd, "model", None), "named_steps")


def _learned_method(hsd) -> str:
    return "linear" if _supports_linear_attr(hsd) else "occlusion"


def _build_l1(kind: str):
    """L1 redactor factory: 'regex' (spine, no download) or 'gliner' (Presidio +
    zero-shot GLiNER NER, catches title-less names/orgs). Falls back to the regex
    spine if the optional deps/snapshot are missing, so a profile never hard-fails."""
    if kind == "gliner":
        from .privatize.l1_presidio import PresidioGlinerRedact, available
        if available():
            return PresidioGlinerRedact()
        return L1Redact()
    if kind != "regex":
        raise ValueError(f"unknown l1 kind {kind!r} (regex|gliner)")
    return L1Redact()


def _build_anchor_cache(hsd, texts: list[str]) -> dict[str, set[str]]:
    """Precompute {L1-applied text -> learned harm-anchor words} ONCE, so the L3
    detector anchor (which may need transformer occlusion) is computed a single
    time per text and reused across the whole intensity sweep + the ladder. Uses
    the SAME L2LearnedDistill spans, so L2 and L3 anchor on identical learned
    rationale."""
    l1 = L1Redact()
    l2 = L2LearnedDistill(hsd, method=_learned_method(hsd))
    cache: dict[str, set[str]] = {}
    for t in texts:
        lt = l1.apply(t)
        if lt in cache:
            continue
        cache[lt] = {lt[s:e].lower() for (s, e) in l2.harm_spans(t)}
    return cache


def _semantic_fn(kind: str):
    """Semantic-similarity proxy: 'tfidf' (char-ngram, default) or 'model2vec'
    (frozen pretrained embedding)."""
    if kind == "model2vec":
        from .metrics.embeddings import model2vec_semantic_similarity
        return model2vec_semantic_similarity
    if kind != "tfidf":
        raise ValueError(f"unknown semantic kind {kind!r} (tfidf|model2vec)")
    return mean_semantic_similarity


def _build_levels(hsd, op_intensity, seed, l1_kind: str = "regex",
                  anchor_cache: dict[str, set[str]] | None = None):
    """The L0->L3 ladder: detector-grounded learned L2, L1 'regex' or
    'gliner', detector-anchored L3 with NLTK/WordNet style. ``anchor_cache`` supplies
    precomputed harm-anchor words so L3 does no per-call attribution."""
    return {
        "L0": RawPassthrough(),
        "L1": _build_l1(l1_kind),
        "L2": L2LearnedDistill(hsd, method=_learned_method(hsd)),
        "L3": L3Rewrite(intensity=op_intensity, seed=seed, hsd=hsd,
                        anchor_words=anchor_cache),
    }


def _curve(train_raw, test_raw, train_authors, test_authors, test_labels, hsd,
           static_atk, f1_raw, f1_maj, p_orig, floors, seed,
           semantic_fn=mean_semantic_similarity,
           anchor_cache: dict[str, set[str]] | None = None) -> dict:
    points: list[CurvePoint] = []
    for intensity in L3_SWEEP:
        lv = L3Rewrite(intensity=intensity, seed=seed, hsd=hsd,
                       anchor_words=anchor_cache)
        tr = lv.apply_many(train_raw)
        te = lv.apply_many(test_raw)
        f1 = macro_f1(test_labels, hsd.predict(te))
        ur = utility_ratio(f1, f1_raw, f1_maj)
        # worst-case privacy: strongest of the static and adaptive attackers
        adaptive = AuthorshipAttacker(seed=seed).fit(tr, train_authors)
        p_acc = max(adaptive.accuracy(te, test_authors),
                    static_atk.accuracy(te, test_authors))
        pr = privacy_ratio(p_acc, p_orig)
        points.append(
            CurvePoint(
                intensity=intensity,
                utility_ratio=round(ur, 4),
                privacy_ratio=round(pr, 4),
                macro_f1=round(f1, 4),
                readability=round(mean_readability(te), 4),
                semantic_sim=round(semantic_fn(test_raw, te), 4),
                to=round(to_score(ur, pr), 4),
                feasible=True,
            )
        )
    return build_curve(points, floors)


def _ab_row(name, lv, train_raw, test_raw, train_authors, test_authors,
            test_labels, hsd, static_atk, f1_raw, f1_maj, p_orig, seed) -> dict:
    """One L2 A/B row: utility, worst-case privacy, board, TO."""
    tr = lv.apply_many(train_raw)
    te = lv.apply_many(test_raw)
    pred = hsd.predict(te)
    f1 = macro_f1(test_labels, pred)
    hs_acc = float(accuracy_score(test_labels, pred))
    acc_static = static_atk.accuracy(te, test_authors)
    acc_adaptive = AuthorshipAttacker(seed=seed).fit(tr, train_authors).accuracy(
        te, test_authors)
    acc_worst = max(acc_static, acc_adaptive)
    ur = utility_ratio(f1, f1_raw, f1_maj)
    pr = privacy_ratio(acc_worst, p_orig)
    return {
        "variant": name,
        "macro_f1": round(f1, 4),
        "utility_ratio": round(ur, 4),
        "attack_worst": round(acc_worst, 4),
        "to_honest": round(to_score(ur, pr), 4),
        "board_score": round(hs_acc - acc_worst, 4),
        "mean_len_chars": round(float(np.mean([len(t) for t in te])), 1),
    }


def _accuracy_strategies(levels, hsd, train_raw, test_raw, train_authors,
                         test_authors, test_labels, train_labels, f1_raw, f1_maj,
                         p_orig, static_atk, per_level, seed) -> dict:
    """Accuracy-measuring reads beyond the headline TO (all CPU-cheap, linear).

    board form, dual-detector utility, ERASER faithfulness of the learned L2
    rationale, an L2 A/B, and a subgroup
    F1 slice. Additive: does not touch the existing TO numbers.
    """
    y = np.asarray(test_labels)
    level_texts = {code: lv.apply_many(test_raw) for code, lv in levels.items()}

    # 1. Board form per level (HS_accuracy - worst-case author accuracy).
    board = {
        code: board_score(y, hsd.predict(level_texts[code]),
                          per_level[code]["attack_acc_honest"])
        for code in levels
    }

    # 2. Dual-detector utility: a char-only head (no word channel) reads each
    # level, so the L2's word-channel keep decision cannot inflate it. It is
    # Goodhart-attenuated, not a fully independent detector (proba corr ~0.98).
    dual = DualHsdHead(seed=seed).fit(train_raw, train_labels)
    f1_raw_dual = macro_f1(test_labels, dual.predict(test_raw))
    dual_util = dual_detector_utility(dual, test_labels, level_texts, f1_raw_dual,
                                      f1_maj)

    # 3-4. ERASER faithfulness + L2 A/B.
    # These need linear attribution (the TF-IDF word channel); for a transformer
    # head, skip them with a note (use Captum IG for transformer-grounded rationale).
    if _supports_linear_attr(hsd):
        l2_learned = L2LearnedDistill(hsd, method="linear", keep_frac=0.6)
        l1_test = L1Redact().apply_many(test_raw)
        spans = [l2_learned.harm_spans(t) for t in test_raw]
        faith = {
            "scored_by_primary_head": eraser_scores(hsd, l1_test, spans),
            "scored_by_dual_head": eraser_scores(dual, l1_test, spans),
            "note": "the primary head both attributes and scores, so its "
                    "sufficiency is optimistic; the char-only dual head attenuates "
                    "(does not rescue) it - sufficiency stays negative because the "
                    "dual reads the same retained tokens (proba corr ~0.98). The "
                    "load-bearing Goodhart-resistant signal is dual utility, not "
                    "the dual sufficiency.",
        }
        # Learned L2 row.
        ab = [
            _ab_row("L2_learned_linear", l2_learned, train_raw, test_raw,
                    train_authors, test_authors, test_labels, hsd, static_atk,
                    f1_raw, f1_maj, p_orig, seed),
        ]
    else:
        faith = {"note": "ERASER of the learned L2 needs a linear-attribution "
                         "(TF-IDF) head; a transformer-grounded rationale is not "
                         "available in this minimal build."}
        ab = []

    # 5. Subgroup F1 by raw text-length tertile, per level.
    slices = length_tertile_slices(test_raw)
    names = ["short", "medium", "long"][: len(slices)]
    subgroup = {
        code: subgroup_f1(test_labels, hsd.predict(level_texts[code]), slices, names)
        for code in levels
    }

    return {
        "board_form": board,
        "dual_detector_utility": dual_util,
        "faithfulness_learned_l2": faith,
        "l2_learned_ab": ab,
        "subgroup_f1_by_length": subgroup,
    }


def run_eval(csv_path: str, seed: int = 0, bootstrap_b: int = 2000,
             floors: Floors | None = None, with_accuracy: bool = True,
             hsd_kind: str = "tfidf", semantic: str = "tfidf",
             l1_kind: str = "regex") -> dict:
    floors = floors or Floors()
    semantic_fn = _semantic_fn(semantic)
    posts = RedditCorpus(csv_path).load()
    authors = RedditCorpus.authors(posts)
    n_candidates = len(authors)
    train, test = split_by_author(posts, seed=seed)
    train_authors = [p.author for p in train]
    test_authors = [p.author for p in test]
    test_labels = [p.label for p in test]
    train_labels = [p.label for p in train]
    train_raw = [p.text for p in train]
    test_raw = [p.text for p in test]

    # Shared HSD head, trained once on raw (tfidf spine by default; cardiff opt-in).
    hsd = _build_hsd(hsd_kind, seed).fit(train_raw, train_labels)
    f1_raw = macro_f1(test_labels, hsd.predict(test_raw))
    f1_maj = majority_baseline_f1(test_labels)

    # Privacy_original: adaptive == static on raw.
    static_atk = AuthorshipAttacker(seed=seed).fit(train_raw, train_authors)
    p_orig = static_atk.accuracy(test_raw, test_authors)

    # The detector-anchored L3 needs harm-anchor words for every text the curve and
    # ladder rewrite (train + test). Compute them ONCE here (a single attribution
    # pass per text, transformer-occlusion included) and reuse everywhere.
    anchor_cache = _build_anchor_cache(hsd, train_raw + test_raw)

    # Choose the L3 operating intensity from the non-degenerate curve selector.
    curve = _curve(train_raw, test_raw, train_authors, test_authors, test_labels,
                   hsd, static_atk, f1_raw, f1_maj, p_orig, floors, seed, semantic_fn,
                   anchor_cache=anchor_cache)
    op_intensity = curve["recommended"]["intensity"]

    # Lexicon-free L0->L3 ladder (detector-grounded learned L2 + detector-anchored L3).
    levels = _build_levels(hsd, op_intensity, seed, l1_kind=l1_kind,
                           anchor_cache=anchor_cache)

    per_level = {}
    hsd_pred = {}
    headline_correct = {}  # worst-case attacker's per-instance correctness
    for code, lv in levels.items():
        tr_txt = lv.apply_many(train_raw)
        te_txt = lv.apply_many(test_raw)
        pred = hsd.predict(te_txt)
        hsd_pred[code] = np.asarray(pred)
        f1 = macro_f1(test_labels, pred)
        ur = utility_ratio(f1, f1_raw, f1_maj)

        # Both attackers, always (the static/adaptive split is non-negotiable).
        acc_static = static_atk.accuracy(te_txt, test_authors)
        static_mask = static_atk.correct_mask(te_txt, test_authors)
        if code == "L0":
            adaptive = static_atk
        else:
            adaptive = AuthorshipAttacker(seed=seed).fit(tr_txt, train_authors)
        acc_adaptive = adaptive.accuracy(te_txt, test_authors)
        adaptive_mask = adaptive.correct_mask(te_txt, test_authors)

        # Honest headline = worst-case (strongest attacker). Optimistic = the
        # most favorable attacker an over-eager team would cherry-pick. The S1
        # toggle flips optimistic -> honest, which always lowers TO.
        if acc_static >= acc_adaptive:
            acc_honest, acc_opt, headline = acc_static, acc_adaptive, "static"
            headline_correct[code] = static_mask
        else:
            acc_honest, acc_opt, headline = acc_adaptive, acc_static, "adaptive"
            headline_correct[code] = adaptive_mask

        pr_opt = privacy_ratio(acc_opt, p_orig)
        pr_honest = privacy_ratio(acc_honest, p_orig)
        per_level[code] = {
            "level": code,
            "name": lv.name,
            "macro_f1": round(f1, 4),
            "utility_ratio": round(ur, 4),
            "attack_acc_static": round(acc_static, 4),
            "attack_acc_adaptive": round(acc_adaptive, 4),
            "attack_acc_optimistic": round(acc_opt, 4),
            "attack_acc_honest": round(acc_honest, 4),
            "headline_attacker": headline,
            "privacy_ratio_optimistic": round(pr_opt, 4),
            "privacy_ratio_honest": round(pr_honest, 4),
            "to_optimistic": round(to_score(ur, pr_opt), 4),
            "to_honest": round(to_score(ur, pr_honest), 4),
            "to_static": round(to_score(ur, privacy_ratio(acc_static, p_orig)), 4),
            "to_adaptive": round(to_score(ur, privacy_ratio(acc_adaptive, p_orig)), 4),
            "privacy_chance_corrected": round(chance_corrected(acc_honest, n_candidates), 4),
            "pii_removed_fraction": round(
                pii_removed_fraction(test_raw, te_txt), 4
            ),
        }

    # S16 paired-bootstrap dominance on the HONEST (worst-case) TO.
    boot = paired_bootstrap(
        np.asarray(test_labels), hsd_pred, headline_correct,
        levels=list(levels), b=bootstrap_b, seed=seed,
    )
    recommended = max(
        ["L1", "L2", "L3"], key=lambda c: per_level[c]["to_honest"]
    )
    dom = dominance(boot, recommended, [c for c in ["L1", "L2", "L3"] if c != recommended])
    boot_summary = {
        c: {k: round(v, 4) for k, v in boot[c].items() if k != "samples"}
        for c in levels
    }

    accuracy = (
        _accuracy_strategies(
            levels, hsd, train_raw, test_raw, train_authors, test_authors,
            test_labels, train_labels, f1_raw, f1_maj, p_orig, static_atk,
            per_level, seed,
        )
        if with_accuracy
        else None
    )

    return {
        "corpus": {
            "path_basename": csv_path.replace("\\", "/").split("/")[-1],
            "n_posts": len(posts),
            "n_authors": n_candidates,
            "n_train": len(train),
            "n_test": len(test),
            "hate_rate": round(np.mean([p.label for p in posts]), 4),
            "attack_setting": "closed-world N-candidate attribution, an upper-bias "
            "(optimistic) estimate of real-world protection",
        },
        "baselines": {
            "f1_raw": round(f1_raw, 4),
            "f1_majority": round(f1_maj, 4),
            "privacy_original_acc": round(p_orig, 4),
            "chance_acc": round(1.0 / n_candidates, 4),
        },
        "levels": per_level,
        "curve": curve,
        "operating_point": {
            "level": "L3",
            "intensity": op_intensity,
            "selected_by": "Kneedle on the feasible frontier; refuses the "
            "degenerate (unreadable) high-TO zone",
        },
        "dominance": {
            "recommended_level": recommended,
            "bootstrap": boot_summary,
            **dom,
        },
        "accuracy": accuracy,
        "_internal": {  # internal fields, stripped from the public JSON
            "hsd": hsd,
            "static_atk": static_atk,
            "op_intensity": op_intensity,
            "seed": seed,
        },
    }
