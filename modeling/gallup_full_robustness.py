"""Robustness checks for the full-survey models (see full_model_results.md).

These answer questions one run of gallup_full_models.py cannot:

  1. Factor ranking with a second method: drop each work-value factor and
     refit logistic regression (the factor's unique contribution), with a
     paired bootstrap CI. Tiers E and C.
  2. The pre-hire tier split into what the person wants vs what the job
     gives (both are worker-reported in Gallup).
  3. Five different random 80/20 splits: is the main split lucky?
  4. A short post-hire check-in: how far do 4-5 appraisal items get alone?

Logistic regression uses C = 0.01 (the value 5-fold CV picked at tiers B-E)
except in part 2 and 4, which tune C on the training set. Every test-set
number uses the main split (random_state=42) unless stated.

Usage (from the repository root): python modeling/gallup_full_robustness.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import gallup_full_models as g

SPLIT_SEEDS = (1, 7, 42, 123, 2026)
DRAWS = 1000
WANT = ["Q19", "Q33H", "Q32A", "Q32B", "Q32C", "Q28"]  # what the person wants
HAVE = ["Q18", "Q34", "Q31A", "Q31B", "Q31C"]  # what the job gives (worker-reported)
CHECK_IN = ["Q33G", "Q36", "Q33I", "Q41", "Q33F"]  # belong, respected, paid fairly, secure, supported


def main() -> None:
    data = g.load_data()
    y = data["satisfied"].to_numpy()
    tr, te = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=g.SEED)
    draws = np.random.default_rng(g.SEED).integers(0, len(te), size=(DRAWS, len(te)))

    def fixed_lr(numeric, categorical, train=tr, test=te):
        X = g.prepare(data, numeric, categorical)
        model = Pipeline([("prep", g.one_hot_preprocessor(numeric, categorical)), ("clf", LogisticRegression(C=0.01, max_iter=5000))])
        return model.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])[:, 1]

    def tuned_lr(numeric, categorical):
        X = g.prepare(data, numeric, categorical)
        return g.logistic(numeric, categorical).tune_and_fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])

    def paired(base, new):
        yt = y[te]
        deltas = [roc_auc_score(yt[s], new[s]) - roc_auc_score(yt[s], base[s]) for s in draws]
        return roc_auc_score(yt, new) - roc_auc_score(yt, base), *np.percentile(deltas, [2.5, 97.5])

    def ci(prob):
        aucs = [roc_auc_score(y[te][s], prob[s]) for s in draws]
        return roc_auc_score(y[te], prob), *np.percentile(aucs, [2.5, 97.5])

    print("1. Unique contribution of each factor: AUC lost when dropped and LR refit")
    for tier in ("E", "C"):
        numeric, categorical = g.tier_columns(tier)
        full = fixed_lr(numeric, categorical)
        rows = []
        for name, cols in g.factor_groups(numeric + categorical).items():
            keep_num = [c for c in numeric if c not in cols]
            keep_cat = [c for c in categorical if c not in cols]
            rows.append((name, len(cols), *paired(fixed_lr(keep_num, keep_cat), full)))
        print(f"  tier {tier} (full model AUC {roc_auc_score(y[te], full):.4f})")
        for name, n, loss, lo, hi in sorted(rows, key=lambda r: -r[2]):
            print(f"    {name:55s} {n:3d} items  {loss:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print("\n2. Pre-hire tier: want side vs have side (LR, C tuned by 5-fold CV)")
    b_num, b_cat = g.tier_columns("B")
    c_num, c_cat = g.tier_columns("C")
    variants = {
        "B (profile and job facts)": (b_num, b_cat),
        "B + want side": (b_num, b_cat + WANT),
        "B + have side": (b_num, b_cat + HAVE),
        "B + want + have (raw answers)": (b_num, b_cat + WANT + HAVE),
        "C (raw answers + 7 fit scores)": (c_num, c_cat),
    }
    preds = {name: tuned_lr(*cols) for name, cols in variants.items()}
    for name, prob in preds.items():
        print(f"  {name:32s} test AUC {roc_auc_score(y[te], prob):.4f}")
    base = preds["B (profile and job facts)"]
    for name in list(variants)[1:]:
        d, lo, hi = paired(base, preds[name])
        print(f"  {name:32s} gain over B {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    gain = paired(preds["B + want + have (raw answers)"], preds["C (raw answers + 7 fit scores)"])
    print(f"  fit scores beyond the raw answers {gain[0]:+.4f} [{gain[1]:+.4f}, {gain[2]:+.4f}]")

    print("\n3. Five random 80/20 splits (LR, C = 0.01)")
    for tier in ("A", "C", "E"):
        numeric, categorical = g.tier_columns(tier)
        aucs = []
        for seed in SPLIT_SEEDS:
            a, b = train_test_split(np.arange(len(data)), test_size=0.2, stratify=y, random_state=seed)
            aucs.append(roc_auc_score(y[b], fixed_lr(numeric, categorical, a, b)))
        print(f"  tier {tier}: " + ", ".join(f"{v:.4f}" for v in aucs) + f"  mean {np.mean(aucs):.4f}, sd {np.std(aucs, ddof=1):.4f}")

    print("\n4. Short post-hire check-in (LR, C tuned by 5-fold CV)")
    a_num, a_cat = g.tier_columns("A")
    check_ins = {
        "4 items: belong, respected, paid fairly, job security": ([], CHECK_IN[:4]),
        "5 items: + feels supported": ([], CHECK_IN),
        "all 11 tier-E appraisal items": ([], g.TIERS["E"][2]),
        "tier A background + 5 items": (a_num, a_cat + CHECK_IN),
    }
    for name, cols in check_ins.items():
        auc, lo, hi = ci(tuned_lr(*cols))
        print(f"  {name:55s} test AUC {auc:.4f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
