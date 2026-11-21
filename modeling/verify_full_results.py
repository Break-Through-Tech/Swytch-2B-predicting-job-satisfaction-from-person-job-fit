"""Independent check of every number in modeling/full_model_results.md.

Uses its own AUC (Mann-Whitney ranks), not the project code, and re-derives
the data facts from the raw Gallup file. Usage (from the repository root):
    python modeling/verify_full_results.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "modeling" / "output" / "full"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def auc(y, p, w=None):
    """ROC-AUC = P(score of a random positive > random negative), ties count 1/2."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    if w is None:
        ranks = pd.Series(p).rank(method="average").to_numpy()
        n1, n0 = y.sum(), (~y).sum()
        return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    w = np.asarray(w, dtype=float)
    # weighted: sum over pos/neg pairs of w_i w_j [p_i > p_j] + 0.5 [tie]
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], y[order], w[order]
    total, neg_below = 0.0, 0.0
    i = 0
    while i < len(p):
        j = i
        while j < len(p) and p[j] == p[i]:
            j += 1
        pos_w = w[i:j][y[i:j]].sum()
        neg_w = w[i:j][~y[i:j]].sum()
        total += pos_w * (neg_below + 0.5 * neg_w)
        neg_below += neg_w
        i = j
    return total / (w[y].sum() * w[~y].sum())


def num(s):
    return pd.to_numeric(s.astype(str).str.strip(), errors="coerce")


raw = pd.read_csv(REPO / "data/raw/JFF_formatted_public_release_values.csv", encoding="cp1252", low_memory=False)
fit = pd.read_csv(REPO / "notebooks/gallup_eda/output/gallup_fsu_fit_model.csv")
pred = pd.read_csv(OUT / "full_test_predictions.csv")
metrics = pd.read_csv(OUT / "full_model_metrics.csv")

print("=== 1. Sample, target and split ===")
q4 = num(raw["Q4"])
print("raw rows", len(raw), "| Q4 = -98:", (q4 == -98).sum(), "| raw rows with valid Q4:", q4.between(0, 10).sum())
print("modelling rows", len(fit), "| unique IDs", fit.ENTITY_ID.nunique())
d = fit[["ENTITY_ID", "satisfied"]].merge(raw, on="ENTITY_ID", how="left")
target = (num(d["Q4"]) >= 7).astype(int)
print("satisfied == (Q4 >= 7) for every row:", bool((target == d["satisfied"]).all()), "| share satisfied", round(d.satisfied.mean(), 4))
y = fit["satisfied"].to_numpy()
tr, te = train_test_split(np.arange(len(fit)), test_size=0.2, stratify=y, random_state=42)
print("train", len(tr), "test", len(te), "| test share satisfied", round(y[te].mean(), 4))
print("saved test IDs == re-derived test split IDs (same order):", bool((pred.ENTITY_ID.to_numpy() == fit.ENTITY_ID.to_numpy()[te]).all()))
print("saved test labels == re-derived:", bool((pred.satisfied.to_numpy() == y[te]).all()))
print("train/test ID overlap:", len(set(fit.ENTITY_ID.iloc[tr]) & set(fit.ENTITY_ID.iloc[te])))
p0 = y[te].mean()
print(f"no-skill: PR-AUC {p0:.4f}, always-satisfied F1 {2 * p0 / (1 + p0):.4f}, constant-prevalence Brier {p0 * (1 - p0):.4f}")

print("\n=== 2. Recompute every model's test metrics from saved predictions ===")
yt = pred.satisfied.to_numpy()
w = num(d["WEIGHT"]).to_numpy()[te]
print("weights: any missing?", np.isnan(w).any(), "| min", w.min().round(4), "max", w.max().round(3))
rows = []
for _, m in metrics.iterrows():
    p = pred[f"{m.tier} | {m.model}"].to_numpy()
    yhat = (p >= m.threshold).astype(int)
    tp = ((yhat == 1) & (yt == 1)).sum(); fp = ((yhat == 1) & (yt == 0)).sum()
    fn = ((yhat == 0) & (yt == 1)).sum(); tn = ((yhat == 0) & (yt == 0)).sum()
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    ba = (rec + tn / (tn + fp)) / 2
    yh5 = (p >= 0.5).astype(int)
    tp5 = ((yh5 == 1) & (yt == 1)).sum(); fp5 = ((yh5 == 1) & (yt == 0)).sum(); fn5 = ((yh5 == 0) & (yt == 1)).sum()
    f1_5 = 2 * tp5 / (2 * tp5 + fp5 + fn5)
    # average precision, own implementation (step-wise, like sklearn)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = yt[order], p[order]
    distinct = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tps = np.cumsum(ys)[distinct]; fps = (distinct + 1) - tps
    precision_c = tps / (tps + fps); recall_c = tps / ys.sum()
    ap = np.sum(np.diff(np.r_[0, recall_c]) * precision_c)
    mine = {"roc_auc": auc(yt, p), "roc_auc_survey_weighted": auc(yt, p, w), "pr_auc": ap,
            "brier": np.mean((p - yt) ** 2), "f1": f1, "precision": prec, "recall": rec,
            "balanced_accuracy": ba, "f1_at_0.5": f1_5}
    rows.append({"tier": m.tier, "model": m.model[:22], **{k: abs(v - m[k]) for k, v in mine.items()}})
diff = pd.DataFrame(rows)
print("largest absolute difference, saved vs recomputed, per metric:")
print(diff.drop(columns=["tier", "model"]).max().map(lambda v: f"{v:.2e}").to_string())

print("\n=== 3. Bootstrap CIs: same draws (seed 42) and a fresh seed (2026) ===")
n = len(yt)
draws42 = np.random.default_rng(42).integers(0, n, size=(2000, n))
draws_new = np.random.default_rng(2026).integers(0, n, size=(2000, n))
key_models = [("A", "Logistic regression"), ("C", "Ensemble (mean of LR, boosting, neural net)"),
              ("E", "Ensemble (mean of LR, boosting, neural net)"), ("E", "Logistic regression"), ("F", "Gradient boosting")]
boot_cache = {}
for t, mname in key_models:
    p = pred[f"{t} | {mname}"].to_numpy()
    a42 = np.array([auc(yt[s], p[s]) for s in draws42]); anew = np.array([auc(yt[s], p[s]) for s in draws_new])
    boot_cache[(t, mname)] = a42
    saved = metrics[(metrics.tier == t) & (metrics.model == mname)].iloc[0]
    print(f"{t} {mname[:20]:20s} AUC {auc(yt, p):.4f} | saved CI [{saved.roc_auc_ci_low:.4f}, {saved.roc_auc_ci_high:.4f}]"
          f" | recomputed seed42 [{np.percentile(a42, 2.5):.4f}, {np.percentile(a42, 97.5):.4f}]"
          f" | seed2026 [{np.percentile(anew, 2.5):.4f}, {np.percentile(anew, 97.5):.4f}]")

print("\n=== 4. Paired gains (recomputed) ===")
gains = pd.read_csv(OUT / "full_tier_gains.csv")


def paired(base, new, draws):
    b, n_ = pred[base].to_numpy(), pred[new].to_numpy()
    deltas = np.array([auc(yt[s], n_[s]) - auc(yt[s], b[s]) for s in draws])
    return auc(yt, n_) - auc(yt, b), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5), (deltas > 0).mean()


LR, GB, ENS = "Logistic regression", "Gradient boosting", "Ensemble (mean of LR, boosting, neural net)"
checks = [("LR", "B vs A", f"A | {LR}", f"B | {LR}"), ("LR", "C vs B", f"B | {LR}", f"C | {LR}"),
          ("LR", "D vs C", f"C | {LR}", f"D | {LR}"), ("LR", "E vs D", f"D | {LR}", f"E | {LR}"),
          ("LR", "F vs E", f"E | {LR}", f"F | {LR}"), ("LR", "C vs A", f"A | {LR}", f"C | {LR}"),
          ("LR", "E vs A", f"A | {LR}", f"E | {LR}"),
          ("LR", "C + job-level averages vs C", f"C | {LR}", f"C+job_avg | {LR}"),
          ("LR", "E vs C + job-level averages", f"C+job_avg | {LR}", f"E | {LR}"),
          ("GB", "C + job-level averages vs C", f"C | {GB}", f"C+job_avg | {GB}"),
          ("ENS", "E: ensemble vs Gradient boosting", f"E | {GB}", f"E | {ENS}"),
          ("ENS", "E: ensemble vs Logistic regression", f"E | {LR}", f"E | {ENS}"),
          ("NEW", "E ensemble vs A LR (team baseline)", f"A | {LR}", f"E | {ENS}"),
          ("NEW", "C ensemble vs A LR (team baseline)", f"A | {LR}", f"C | {ENS}")]
name = {"LR": LR, "GB": GB, "ENS": ENS}
for m_, comp, base, new in checks:
    g, lo, hi, share = paired(base, new, draws42)
    if m_ != "NEW":
        s = gains[(gains.model == name[m_]) & (gains.comparison == comp)].iloc[0]
        ok = max(abs(g - s.auc_improvement), abs(lo - s.ci_low), abs(hi - s.ci_high))
        print(f"{m_:3s} {comp:36s} {g:+.4f} [{lo:+.4f}, {hi:+.4f}] >0 in {share:.1%} | max diff vs saved {ok:.1e}")
    else:
        print(f"{m_:3s} {comp:36s} {g:+.4f} [{lo:+.4f}, {hi:+.4f}] >0 in {share:.1%} | (not in saved file)")

print("\n=== 5. Ensemble membership sensitivity (test AUC) ===")
for t in ("C", "E"):
    cols = {k: pred[f"{t} | {k}"] for k in (LR, GB, "Random forest", "Keras neural network")}
    variants = {"LR+GB+Keras (used)": [LR, GB, "Keras neural network"], "all 4 (adds RF)": list(cols),
                "LR+GB": [LR, GB], "LR+GB+RF": [LR, GB, "Random forest"]}
    out = {k: round(auc(yt, np.mean([cols[c] for c in v], axis=0)), 4) for k, v in variants.items()}
    print(t, out, "| best single:", max((round(auc(yt, cols[c]), 4), c) for c in cols))

print("\n=== 6. Tier C at other thresholds (balanced accuracy) ===")
for t in ("C", "E"):
    p = pred[f"{t} | {ENS}"].to_numpy()
    for thr in (0.5, 0.6, 0.65, p0):
        yh = p >= thr
        tpr = (yh & (yt == 1)).sum() / (yt == 1).sum(); tnr = (~yh & (yt == 0)).sum() / (yt == 0).sum()
        print(f"  {t} threshold {thr:.3f}: balanced accuracy {(tpr + tnr) / 2:.3f}  (TPR {tpr:.3f}, TNR {tnr:.3f})")

print("\n=== 7. Subgroup AUCs and gap CIs (tier E and C ensembles) ===")
dt = d.iloc[te].reset_index(drop=True)
female = (num(dt.Q65) == 1).to_numpy(); male = (num(dt.Q65) == 2).to_numpy()
black = (num(dt.Q61_3) == 1).to_numpy(); notblack = (num(dt.Q61_3) == -99).to_numpy()
for t in ("E", "C"):
    p = pred[f"{t} | {ENS}"].to_numpy()
    for label, a, b in (("women - men", female, male), ("Black - not Black", black, notblack)):
        gap = auc(yt[a], p[a]) - auc(yt[b], p[b])
        gs = []
        for s in draws42[:1000]:
            sa, sb = s[a[s]], s[b[s]]
            gs.append(auc(yt[sa], p[sa]) - auc(yt[sb], p[sb]))
        print(f"  {t} {label:18s} n={a.sum()}/{b.sum()}  AUC {auc(yt[a], p[a]):.3f} vs {auc(yt[b], p[b]):.3f}"
              f"  gap {gap:+.3f} [{np.percentile(gs, 2.5):+.3f}, {np.percentile(gs, 97.5):+.3f}]")

print("\n=== 8. Data facts quoted in the write-up (modelling sample, unweighted) ===")
q37 = pd.concat([num(d[f"Q37{c}"]) == 1 for c in "ABCDEFGHI"], axis=1).any(axis=1)
q38 = pd.concat([num(d[f"Q38{c}"]) == 1 for c in "ABC"], axis=1).any(axis=1)
print(f"any discrimination (Q37A-I == 1): {q37.mean():.3f}; satisfied {d.satisfied[q37].mean():.3f} vs {d.satisfied[~q37].mean():.3f}")
print(f"any harassment (Q38A-C == 1): {q38.mean():.3f}; satisfied {d.satisfied[q38].mean():.3f} vs {d.satisfied[~q38].mean():.3f}")
q36 = num(d.Q36)
print(f"disagree respected (Q36 in 4,5): {q36.isin([4, 5])[q37].mean():.3f} if discrimination vs {q36.isin([4, 5])[~q37].mean():.3f} otherwise")
f_, g_ = num(d.Q33F), num(d.Q33G)
both = f_.between(1, 5) & g_.between(1, 5)
print(f"Spearman Q33F vs Q33G (both 1-5, n={both.sum()}): {f_[both].corr(g_[both], method='spearman'):.3f}")
q6, q7 = num(d.Q6), num(d.Q7)
print("Q6 codes:", sorted(q6.dropna().unique().astype(int)), "\nQ7 codes:", sorted(q7.dropna().unique().astype(int)))
for c, ref in (("Q52", "MARITAL_STATUS"), ("Q57", "EDUCATION")):
    a, b = num(d[c]), num(d[ref])
    answered = a > 0
    print(f"{c}: share -98/missing {(~answered).mean():.3f}; {ref} valid (not 9) {(~b.isin([9])).mean():.3f}; "
          f"cross-tab of answered rows:\n{pd.crosstab(a[answered], b[answered])}")
qcols = [c for c in raw.columns if c.startswith("Q")]
print("raw columns", raw.shape[1], "| Q* columns", len(qcols), "| numbered questions", len({c.split('_')[0].rstrip('ABCDEFGHIJ') for c in qcols if c[1:2].isdigit()}))
