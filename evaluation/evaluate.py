"""
evaluate.py — Phase 9: classification summary and fragmentation analyses
========================================================================
Rebuilds the classification tables and figures directly from the 5-fold
prediction files in results/predictions/<model>/ (no GPU needed).

Paper outputs reproduced here:
  * Table 9   classification_summary.csv            (mean ± SD over folds)
  * Table 5   code_mixed_fragmentation_ttest.csv    (script-mixed vs single-script, Welch t)
  * Table 13  fragmentation_stratified_accuracy.csv (accuracy by mBERT fragmentation quartile)
  * Figures   figures/*.png

The mBERT 798-token + MLM run used the 51,155-sentence corpus version, so its
CV split differs from the 51,161-sentence baselines. All paired analyses
use an inner join on normalised text (50,827 sentences present in every
run); merged_predictions_wide.csv holds that paired set and is the input of
evaluation/significance.py.

Usage:  python evaluation/evaluate.py
"""
import json, re, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results" / "predictions"
OUT = ROOT / "results" / "classification"
IMG = OUT / "figures"
IMG.mkdir(parents=True, exist_ok=True)

ALL_MODELS = {  # prediction dir -> (display name, colour); fixed order for every figure
    "mbert_baseline":      ("mBERT (baseline)", "#e94560"),
    "muril_baseline":      ("MuRIL (baseline)", "#f5a623"),
    "mbert_obpe498_nomlm": ("mBERT OBPE +498 (no MLM)", "#9b59b6"),
    "mbert_798_mlm":       ("mBERT 798-tok + MLM", "#53d8fb"),
}
BASELINE, ADAPTED = "mbert_baseline", "mbert_798_mlm"
MODELS = {m: v for m, v in ALL_MODELS.items() if (CKPT / m / "fold_metrics.csv").exists()}
for required in [BASELINE, "muril_baseline", ADAPTED]:
    if required not in MODELS:
        sys.exit(f"Missing predictions: {CKPT / required}")
BG_FIG, BG_AX, SPINE = "#1a1a2e", "#16213e", "#444466"


def style(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor(SPINE)
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def fmt_p(p):
    return "p<1e-300" if p == 0 else (f"p={p:.1e}" if p < 1e-3 else f"p={p:.4f}")


def norm_text(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


# ── Load ──────────────────────────────────────────────────────────────────────
preds, folds = {}, {}
for m in MODELS:
    d = CKPT / m
    df = pd.concat(
        [pd.read_csv(d / f"fold{k}_predictions.csv", encoding="utf-8-sig") for k in range(1, 6)],
        ignore_index=True,
    )
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    df["is_code_mixed"] = df["is_code_mixed"].astype(str).str.lower().eq("true")
    df["key"] = df["text"].map(norm_text)
    preds[m] = df
    folds[m] = pd.read_csv(d / "fold_metrics.csv", encoding="utf-8-sig")
    print(f"{m}: {len(df):,} predictions, acc={df.correct.mean():.4f}")

# ── 1. Classification summary + model comparison ──────────────────────────────
rows = []
for m, (name, _) in MODELS.items():
    f = folds[m]
    info = json.loads((CKPT / m / "label_map.json").read_text(encoding="utf-8"))
    r = {"model_dir": m, "model": name, "dataset": info["dataset_name"], "n_folds": len(f)}
    for c in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        r[f"mean_{c}"] = f[c].mean()
        r[f"std_{c}"] = f[c].std(ddof=1)
    r["mean_train_time_min"] = f["train_time_min"].mean()
    rows.append(r)
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "classification_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor(BG_FIG)
for ax, metric, title in [(axes[0], "macro_f1", "Macro-F1"), (axes[1], "accuracy", "Accuracy")]:
    style(ax)
    names = summary["model"]; means = summary[f"mean_{metric}"]; sds = summary[f"std_{metric}"]
    bars = ax.bar(names, means, yerr=sds, capsize=8, color=[c for _, c in MODELS.values()],
                  edgecolor="white", linewidth=0.6, error_kw={"ecolor": "white"})
    for b, v, s in zip(bars, means, sds):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.003, f"{v:.4f}",
                ha="center", va="bottom", color="white", fontweight="bold")
    lo = max(0.0, (means - sds).min() - 0.03)
    ax.set_ylim(lo, (means + sds).max() + 0.015)  # zoomed so ~1-pt differences are visible
    ax.set_title(f"{title} (mean ± SD, 5 folds)", fontsize=13, fontweight="bold")
    ax.set_ylabel(title)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
fig.suptitle("Model Comparison — Macro-F1 and Accuracy (y-axis truncated)",
             color="white", fontweight="bold")
plt.tight_layout()
fig.savefig(IMG / "model_comparison.png", dpi=200, bbox_inches="tight", facecolor=BG_FIG)
plt.close(fig)

# ── 2. Paired merge (inner join on normalised text) ───────────────────────────
wide = None
for m, df in preds.items():
    d = df.drop_duplicates("key").drop(columns="text").set_index("key").add_prefix(f"{m}_")
    wide = d if wide is None else wide.join(d, how="inner")
wide = wide.reset_index().rename(columns={"key": "text"})
n_paired = len(wide)
print(f"paired rows (present in all {len(MODELS)} runs): {n_paired:,}")
wide.to_csv(OUT / "merged_predictions_wide.csv", index=False)

# ── 3. Fragmentation vs confidence (each model on its own predictions) ────────
corr_rows = []
fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5.5))
fig.patch.set_facecolor(BG_FIG)
rng = np.random.default_rng(42)
for ax, (m, (name, color)) in zip(axes, MODELS.items()):
    df = preds[m]
    r, p = stats.pearsonr(df["fragmentation_ratio"], df["confidence"])
    rho, p_s = stats.spearmanr(df["fragmentation_ratio"], df["confidence"])
    corr_rows.append({"model_dir": m, "model": name, "pearson_r": round(r, 4), "p_value": p,
                      "spearman_rho": round(rho, 4), "spearman_p": p_s, "n": len(df),
                      "significant": bool(p < 0.05)})
    style(ax)
    idx = rng.choice(len(df), size=min(8000, len(df)), replace=False)
    ax.scatter(df["fragmentation_ratio"].values[idx], df["confidence"].values[idx],
               s=5, alpha=0.35, color=color, edgecolors="none")
    ax.set_xlim(0.8, df["fragmentation_ratio"].quantile(0.999))
    ax.set_title(f"{name}\nr={r:.4f} ({fmt_p(p)}), n={len(df):,}", fontsize=11)
    ax.set_xlabel("Fragmentation ratio (subwords/word, own tokenizer)")
    ax.set_ylabel("Confidence")
fig.suptitle("Fragmentation vs. Prediction Confidence (8k-point sample, x clipped at 99.9th pct)",
             color="white", fontweight="bold")
plt.tight_layout()
fig.savefig(IMG / "fragmentation_confidence_scatter.png", dpi=200, bbox_inches="tight", facecolor=BG_FIG)
plt.close(fig)
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT / "fragmentation_confidence_correlation.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG_FIG); style(ax)
bars = ax.bar(corr["model"], corr["pearson_r"].abs(), color=[c for _, c in MODELS.values()],
              edgecolor="white", linewidth=0.6)
for b, v in zip(bars, corr["pearson_r"]):
    ax.text(b.get_x() + b.get_width() / 2, abs(v) + 0.001, f"r={v:.4f}",
            ha="center", va="bottom", color="white", fontweight="bold")
ax.set_ylabel("|Pearson r|  (fragmentation vs. confidence)")
ax.set_title("Fragmentation-Confidence Correlation Strength\n(lower = fragmentation predicts confidence less)",
             fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
plt.tight_layout()
fig.savefig(IMG / "correlation_shrinkage.png", dpi=200, bbox_inches="tight", facecolor=BG_FIG)
plt.close(fig)

# ── 4. Accuracy by fragmentation quartile (paired rows, common stratifier) ────
wide["quartile"] = pd.qcut(wide[f"{BASELINE}_fragmentation_ratio"], 4,
                           labels=["Q1_low", "Q2", "Q3", "Q4_high"])
strat = wide.groupby("quartile", observed=True).agg(
    n=("text", "size"),
    **{name: (f"{m}_correct", "mean") for m, (name, _) in MODELS.items()},
)
strat["delta_adapted_minus_baseline"] = strat[MODELS[ADAPTED][0]] - strat[MODELS[BASELINE][0]]
strat.to_csv(OUT / "fragmentation_stratified_accuracy.csv")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(BG_FIG); style(ax)
x = np.arange(len(strat)); w = 0.8 / len(MODELS)
for i, (m, (name, color)) in enumerate(MODELS.items()):
    ax.bar(x + (i - (len(MODELS) - 1) / 2) * w, strat[name], w, label=name, color=color,
           edgecolor="white", linewidth=0.5)
ax.set_xticks(x, [f"{q}\n(n={n:,})" for q, n in zip(strat.index, strat["n"])])
ax.set_ylim(0.65, strat[[n for n, _ in MODELS.values()]].values.max() + 0.03)
ax.set_ylabel("Accuracy")
ax.set_title(f"Accuracy by Fragmentation Quartile\n(quartiles of mBERT-baseline fragmentation; "
             f"{n_paired:,} texts present in all runs; y-axis truncated)", fontweight="bold")
ax.legend(facecolor=BG_AX, labelcolor="white", edgecolor=SPINE)
plt.tight_layout()
fig.savefig(IMG / "fragmentation_stratified_accuracy.png", dpi=200, bbox_inches="tight", facecolor=BG_FIG)
plt.close(fig)

# ── 5. Code-mixed vs monolingual fragmentation (Welch t) ──────────────────────
tt = []
for m, (name, _) in MODELS.items():
    df = preds[m]
    cm = df.loc[df.is_code_mixed, "fragmentation_ratio"]; mono = df.loc[~df.is_code_mixed, "fragmentation_ratio"]
    t, p = stats.ttest_ind(cm, mono, equal_var=False)
    tt.append({"model_dir": m, "model": name, "n_code_mixed": len(cm), "n_monolingual": len(mono),
               "mean_frag_code_mixed": round(cm.mean(), 4), "mean_frag_monolingual": round(mono.mean(), 4),
               "welch_t": round(t, 4), "p_value": p, "significant": bool(p < 0.05)})
pd.DataFrame(tt).to_csv(OUT / "code_mixed_fragmentation_ttest.csv", index=False)

print(f"\nWrote {OUT.relative_to(ROOT).as_posix()}/*.csv and {IMG.relative_to(ROOT).as_posix()}/*.png")
print(summary[["model", "mean_accuracy", "mean_macro_f1"]].to_string(index=False))
print(strat.round(4).to_string())
