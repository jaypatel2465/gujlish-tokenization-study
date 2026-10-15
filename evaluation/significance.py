"""
significance.py — Phase 9: paired statistical tests on shared predictions
=========================================================================
Run after evaluation/evaluate.py (reads results/classification/
merged_predictions_wide.csv, the 50,827 sentences present in every run).

Paper outputs reproduced here (written to results/classification/):
  * Table 12  significance_tests.json
              - McNemar's test with continuity correction (n10 / n01 / chi2 / p)
              - paired bootstrap (N=1,000) 95% CI of the macro-F1 difference
              - paired t-test on per-fold macro-F1 (n=5) + Cohen's d
  * Table 10  per_class_metrics.csv     (precision / recall / F1 per class)
  * Table 13  truncation_by_quartile.csv (approx. sentences with > 96 tokens,
              token count ~= mBERT fragmentation ratio x word count)
  * Table 14  audit_subset_check.csv    (150-sentence stratified label audit)

Usage:  python evaluation/significance.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, ttest_rel

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "classification"
WIDE = OUT / "merged_predictions_wide.csv"
AUDIT = ROOT / "data" / "processed" / "human_audit_sample.csv"

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
NAMES = {
    "mbert_baseline": "mBERT baseline",
    "muril_baseline": "MuRIL baseline",
    "mbert_obpe498_nomlm": "mBERT OBPE +498 (no MLM)",
    "mbert_798_mlm": "mBERT 798-tok + MLM",
}
BASELINE, ADAPTED = "mbert_baseline", "mbert_798_mlm"
N_BOOTSTRAP, SEED, MAX_SEQ_LEN = 1000, 42, 96


def prf(y_true, y_pred):
    """Per-class precision / recall / F1 from integer labels (fast, no sklearn)."""
    cm = np.bincount(y_true * 3 + y_pred, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.nan_to_num(tp / cm.sum(0))
        r = np.nan_to_num(tp / cm.sum(1))
        f = np.nan_to_num(2 * p * r / (p + r))
    return p, r, f


def mcnemar(correct_a, correct_b):
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    return {"n10": n10, "n01": n01, "chi2": round(stat, 4), "p_value": float(chi2.sf(stat, 1))}


def bootstrap_f1_diff(y, pred_a, pred_b):
    """Paired bootstrap over sentences: macro-F1(a) - macro-F1(b)."""
    np.random.seed(SEED)
    n = len(y)
    diffs = np.empty(N_BOOTSTRAP)
    for k in range(N_BOOTSTRAP):
        i = np.random.randint(0, n, n)
        diffs[k] = prf(y[i], pred_a[i])[2].mean() - prf(y[i], pred_b[i])[2].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_resamples": N_BOOTSTRAP, "mean_diff": round(float(diffs.mean()), 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4)}


def fold_ttest(model_a, model_b):
    fa = pd.read_csv(PRED / model_a / "fold_metrics.csv").sort_values("fold")["macro_f1"].values
    fb = pd.read_csv(PRED / model_b / "fold_metrics.csv").sort_values("fold")["macro_f1"].values
    t = ttest_rel(fa, fb)
    d = fa - fb
    return {"n_folds": len(d), "t": round(float(t.statistic), 4), "p_value": round(float(t.pvalue), 4),
            "mean_diff": round(float(d.mean()), 4), "cohens_d": round(float(d.mean() / d.std(ddof=1)), 4),
            "diff_ci95": [round(float(x), 4) for x in
                          d.mean() + np.array([-1, 1]) * 2.776 * d.std(ddof=1) / np.sqrt(len(d))]}


def norm_text(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def main():
    if not WIDE.exists():
        sys.exit("Run evaluation/evaluate.py first (creates merged_predictions_wide.csv).")
    wide = pd.read_csv(WIDE)
    models = [m for m in NAMES if f"{m}_pred_label" in wide.columns]
    y = wide[f"{BASELINE}_true_label"].map(L2I).to_numpy(np.int8)
    pred = {m: wide[f"{m}_pred_label"].map(L2I).to_numpy(np.int8) for m in models}
    correct = {m: pred[m] == y for m in models}
    print(f"Shared predictions: {len(y):,}")

    # ── Table 12 ──────────────────────────────────────────────────────────────
    results = {"n_shared_predictions": int(len(y)), "mcnemar": {}, "fold_ttest": {}}
    for a, b in [(ADAPTED, BASELINE), (BASELINE, "muril_baseline"), (ADAPTED, "muril_baseline")]:
        results["mcnemar"][f"{a} vs {b}"] = mcnemar(correct[a], correct[b])
    results["bootstrap_macro_f1"] = {f"{ADAPTED} - {BASELINE}": bootstrap_f1_diff(y, pred[ADAPTED], pred[BASELINE])}
    for m in [ADAPTED, "muril_baseline", "mbert_obpe498_nomlm"]:
        if (PRED / m / "fold_metrics.csv").exists():
            results["fold_ttest"][f"{m} vs {BASELINE}"] = fold_ttest(m, BASELINE)
    (OUT / "significance_tests.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ── Table 10 ──────────────────────────────────────────────────────────────
    rows = []
    for m in models:
        p, r, f = prf(y, pred[m])
        for k, lbl in enumerate(LABELS + ["macro"]):
            if lbl == "macro":
                rows.append({"model": NAMES[m], "class": lbl, "n": len(y), "precision": p.mean(),
                             "recall": r.mean(), "f1": f.mean()})
            else:
                rows.append({"model": NAMES[m], "class": lbl, "n": int((y == k).sum()),
                             "precision": p[k], "recall": r[k], "f1": f[k]})
    per_class = pd.DataFrame(rows).round(4)
    per_class.to_csv(OUT / "per_class_metrics.csv", index=False)

    # ── Table 13 truncation column ────────────────────────────────────────────
    frag = wide[f"{BASELINE}_fragmentation_ratio"]
    approx_tokens = frag * wide["text"].str.split().str.len()
    quartile = pd.qcut(frag, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    trunc = (approx_tokens > MAX_SEQ_LEN).groupby(quartile, observed=True).mean()
    trunc["Overall"] = (approx_tokens > MAX_SEQ_LEN).mean()
    trunc.round(4).rename("approx_truncated_share").to_csv(OUT / "truncation_by_quartile.csv")

    # ── Table 14 ──────────────────────────────────────────────────────────────
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig")
    pos = pd.Series(range(len(wide)), index=wide["text"])
    pos = pos[~pos.index.duplicated()]
    idx = audit["text"].map(norm_text).map(pos)
    ok = idx.notna().to_numpy()
    ya = audit["sentiment_label"].map(L2I).to_numpy(np.int8)[ok]
    ii = idx[ok].astype(int).to_numpy()
    audit_rows = []
    for m in models:
        yp = pred[m][ii]
        audit_rows.append({"model": NAMES[m], "n_matched": int(ok.sum()),
                           "audit_accuracy": round(float((yp == ya).mean()), 4),
                           "audit_macro_f1": round(float(prf(ya, yp)[2].mean()), 4),
                           "shared_set_accuracy": round(float(correct[m].mean()), 4)})
    pd.DataFrame(audit_rows).to_csv(OUT / "audit_subset_check.csv", index=False)

    # ── Console summary ───────────────────────────────────────────────────────
    for k, v in results["mcnemar"].items():
        print(f"McNemar {k}: n10={v['n10']} n01={v['n01']} chi2={v['chi2']:.2f} p={v['p_value']:.2e}")
    for k, v in results["bootstrap_macro_f1"].items():
        print(f"Bootstrap {k}: mean={v['mean_diff']:+.4f} 95% CI [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}]")
    for k, v in results["fold_ttest"].items():
        print(f"Fold t-test {k}: t={v['t']:.2f} p={v['p_value']:.4f} d={v['cohens_d']:+.2f}")
    print(per_class[per_class["class"] != "macro"].to_string(index=False))
    print(trunc.round(3).to_string())
    print(pd.DataFrame(audit_rows).to_string(index=False))
    print(f"\nWrote {OUT.relative_to(ROOT).as_posix()}/")


if __name__ == "__main__":
    main()
