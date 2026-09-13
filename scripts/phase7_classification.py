import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Phase 7 -- Sentiment Classification Experiment
================================================
Fine-tunes and compares three models on Gujarati-English code-mixed sentiment:
  1. mBERT (bert-base-multilingual-cased) -- baseline
  2. MuRIL (google/muril-base-cased)      -- baseline
  3. Vocabulary-adapted mBERT             -- proposed

Protocol:
  - 5-fold stratified cross-validation on the 3,000-row dev subset
  - CPU training; 2 epochs per fold; batch size 16
  - Metrics per fold: accuracy, macro-precision, macro-recall, macro-F1
  - Fold checkpointing: saves progress after each fold so runs can be inspected
    even if interrupted
  - Significance test: paired t-test on macro-F1 across 5 folds
    (adapted-mBERT vs baseline-mBERT)

Outputs:
  results/tables/table6_classification_results.csv
  results/figures/fig5_model_performance_comparison.png
  logs/phase7_fold_results.csv   (intermediate checkpoint)
"""

import csv
import json
import datetime
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
from scipy.stats import ttest_rel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset

import os
ROOT           = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")
DATA_PROCESSED = ROOT / "data" / "processed"
TABLES         = ROOT / "results" / "tables"
FIGURES        = ROOT / "results" / "figures"
MODELS         = ROOT / "models"
LOGS           = ROOT / "logs"

RANDOM_SEED   = 42
N_FOLDS       = 5
N_EPOCHS      = 3          # 3 epochs — beneficial with GPU; same wall-time as 2 on CPU
LEARNING_RATE = 2e-5
MAX_SEQ_LEN   = 128

# ── Device detection & tuning ──────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE          = "cuda"
    DEVICE_NAME     = torch.cuda.get_device_name(0)
    TOTAL_VRAM_GB   = torch.cuda.get_device_properties(0).total_memory / 1e9
    # Scale batch size based on VRAM: 8GB→32, 16GB→64, 24GB→96
    if TOTAL_VRAM_GB >= 20:
        BATCH_SIZE = 64
    elif TOTAL_VRAM_GB >= 12:
        BATCH_SIZE = 32
    elif TOTAL_VRAM_GB >= 6:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 8
    USE_FP16        = True    # Half-precision on CUDA → ~2x speed
    USE_CPU         = False
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE          = "mps"
    DEVICE_NAME     = "Apple Silicon MPS"
    TOTAL_VRAM_GB   = 0
    BATCH_SIZE      = 16
    USE_FP16        = False   # MPS doesn't support fp16 yet
    USE_CPU         = False
else:
    DEVICE          = "cpu"
    DEVICE_NAME     = "CPU (no GPU found)"
    TOTAL_VRAM_GB   = 0
    BATCH_SIZE      = 16
    USE_FP16        = False
    USE_CPU         = True

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def append_log(row: dict):
    log_file = LOGS / "experiment_log.csv"
    fieldnames = ["date","phase","model","tokenizer","dataset_version",
                  "sample_size","vocab_size","learning_rate","batch_size",
                  "epochs","seed","result"]
    write_header = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})

print("=" * 60)
print("PHASE 7 -- Sentiment Classification Experiment")
print("=" * 60)
print(f"  Device      : {DEVICE.upper()} — {DEVICE_NAME}")
if TOTAL_VRAM_GB > 0:
    print(f"  VRAM        : {TOTAL_VRAM_GB:.1f} GB")
print(f"  Batch size  : {BATCH_SIZE}  (auto-set for device)")
print(f"  FP16        : {'Enabled' if USE_FP16 else 'Disabled'}")
print(f"  Config      : {N_FOLDS}-fold CV, {N_EPOCHS} epochs, lr={LEARNING_RATE}")
if DEVICE == "cpu":
    print("  [WARNING] Running on CPU — Phase 7 will take 4-8 hours.")
    print("            For fast results, run on a machine with CUDA GPU (3-15 min total).")

# ── Load dev subset ────────────────────────────────────────────────────────────
print("\n[1/5] Loading dev subset...")
dev_path = DATA_PROCESSED / "dev_subset.csv"
if not dev_path.exists():
    raise FileNotFoundError("Run phase2_clean.py first.")

df = pd.read_csv(dev_path, encoding="utf-8", low_memory=False)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].notna() & (df["text"] != "") & df["sentiment_label"].notna()].copy()
df = df.reset_index(drop=True)

# Encode labels
label2id = {lbl: i for i, lbl in enumerate(sorted(df["sentiment_label"].unique()))}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["sentiment_label"].map(label2id)

print(f"  Loaded {len(df):,} rows")
print(f"  Labels: {label2id}")
print(f"  Class distribution:")
for lbl, idx in label2id.items():
    cnt = (df["label"] == idx).sum()
    print(f"    {lbl:12}: {cnt:5,}  ({cnt/len(df)*100:.1f}%)")

NUM_LABELS = len(label2id)

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS_TO_EVAL = [
    {
        "name":       "mBERT_baseline",
        "model_path": "bert-base-multilingual-cased",
        "tok_path":   "bert-base-multilingual-cased",
        "label":      "mBERT (baseline)",
        "color":      "#e94560",
    },
    {
        "name":       "MuRIL_baseline",
        "model_path": "google/muril-base-cased",
        "tok_path":   "google/muril-base-cased",
        "label":      "MuRIL (baseline)",
        "color":      "#0f3460",
    },
    {
        "name":       "mBERT_adapted",
        "model_path": str(MODELS / "mbert_adapted"),
        "tok_path":   str(MODELS / "mbert_adapted"),
        "label":      "mBERT (adapted)",
        "color":      "#53d8fb",
    },
]

# ── Fold checkpoint file ───────────────────────────────────────────────────────
fold_checkpoint_path = LOGS / "phase7_fold_results.csv"
fold_fieldnames = ["model_name", "fold", "accuracy", "macro_precision",
                   "macro_recall", "macro_f1", "timestamp"]

def load_completed_folds():
    """Return set of (model_name, fold) already computed."""
    done = set()
    if fold_checkpoint_path.exists():
        with open(fold_checkpoint_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["model_name"], int(row["fold"])))
    return done

def save_fold_result(model_name, fold, acc, prec, rec, f1):
    write_header = not fold_checkpoint_path.exists()
    with open(fold_checkpoint_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model_name": model_name, "fold": fold,
            "accuracy": round(acc, 6), "macro_precision": round(prec, 6),
            "macro_recall": round(rec, 6), "macro_f1": round(f1, 6),
            "timestamp": datetime.datetime.now().isoformat()
        })

# ── Training / eval loop ───────────────────────────────────────────────────────
def tokenize_dataset(texts, labels, tokenizer, max_len=MAX_SEQ_LEN):
    encodings = tokenizer(
        list(texts), truncation=True, padding=True,
        max_length=max_len, return_tensors=None
    )
    encodings["labels"] = list(labels)
    return Dataset.from_dict(encodings)

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":         accuracy_score(labels, preds),
        "macro_f1":         f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision":  precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall":     recall_score(labels, preds, average="macro", zero_division=0),
    }

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
texts_all  = df["text"].values
labels_all = df["label"].values

completed_folds = load_completed_folds()
if completed_folds:
    print(f"\n  [RESUME] Found {len(completed_folds)} previously completed fold results.")

all_fold_results = {m["name"]: [] for m in MODELS_TO_EVAL}

# Load any existing fold results from checkpoint
if fold_checkpoint_path.exists():
    with open(fold_checkpoint_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mn = row["model_name"]
            if mn in all_fold_results:
                all_fold_results[mn].append({
                    "fold":             int(row["fold"]),
                    "accuracy":         float(row["accuracy"]),
                    "macro_precision":  float(row["macro_precision"]),
                    "macro_recall":     float(row["macro_recall"]),
                    "macro_f1":         float(row["macro_f1"]),
                })

print("\n[2/5] Running 5-fold cross-validation...")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts_all, labels_all)):
    fold_num = fold_idx + 1
    train_texts, val_texts   = texts_all[train_idx],  texts_all[val_idx]
    train_labels, val_labels = labels_all[train_idx], labels_all[val_idx]

    print(f"\n  --- Fold {fold_num}/{N_FOLDS} ---")
    print(f"    Train: {len(train_texts):,}  |  Val: {len(val_texts):,}")

    for model_cfg in MODELS_TO_EVAL:
        mname = model_cfg["name"]

        if (mname, fold_num) in completed_folds:
            print(f"    [{mname}] fold {fold_num} already done -- skipping.")
            continue

        print(f"\n    [{mname}] Loading tokenizer + model...")
        t_start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_cfg["tok_path"])
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["model_path"],
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )

        train_ds = tokenize_dataset(train_texts, train_labels, tokenizer)
        val_ds   = tokenize_dataset(val_texts,   val_labels,   tokenizer)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        output_dir = str(MODELS / f"tmp_{mname}_fold{fold_num}")
        training_args = TrainingArguments(
            output_dir                  = output_dir,
            num_train_epochs            = N_EPOCHS,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = BATCH_SIZE,
            learning_rate               = LEARNING_RATE,
            weight_decay                = 0.01,
            eval_strategy               = "epoch",
            save_strategy               = "no",
            load_best_model_at_end      = False,
            seed                        = RANDOM_SEED,
            logging_steps               = 20,
            report_to                   = "none",
            use_cpu                     = USE_CPU,
            no_cuda                     = USE_CPU,
            fp16                        = USE_FP16,         # half-precision on CUDA
            dataloader_num_workers      = 0,
            gradient_accumulation_steps = max(1, 32 // BATCH_SIZE),  # effective batch ≈32
            warmup_ratio                = 0.1,
        )

        trainer = Trainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_ds,
            eval_dataset    = val_ds,
            tokenizer       = tokenizer,
            data_collator   = collator,
            compute_metrics = compute_metrics_fn,
        )

        print(f"    [{mname}] Training fold {fold_num}...")
        trainer.train()

        print(f"    [{mname}] Evaluating...")
        eval_results = trainer.evaluate()

        acc  = eval_results.get("eval_accuracy",        0.0)
        prec = eval_results.get("eval_macro_precision", 0.0)
        rec  = eval_results.get("eval_macro_recall",    0.0)
        f1   = eval_results.get("eval_macro_f1",        0.0)
        elapsed = time.time() - t_start

        print(f"    [{mname}] fold {fold_num}: "
              f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  "
              f"({elapsed/60:.1f}min)")

        save_fold_result(mname, fold_num, acc, prec, rec, f1)
        all_fold_results[mname].append({
            "fold": fold_num, "accuracy": acc,
            "macro_precision": prec, "macro_recall": rec, "macro_f1": f1
        })

        # Free memory between models
        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Clean up tmp output dir
        import shutil
        tmp_dir = Path(output_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

print("\n[3/5] Aggregating results...")

# ── Compute mean/std across folds ─────────────────────────────────────────────
summary_rows = []
for model_cfg in MODELS_TO_EVAL:
    mname = model_cfg["name"]
    folds = all_fold_results[mname]
    if not folds:
        print(f"  [!] No fold results for {mname} -- skipping.")
        continue

    fold_df = pd.DataFrame(folds)
    row = {
        "model":             model_cfg["label"],
        "model_id":          mname,
        "n_folds":           len(fold_df),
        "mean_accuracy":     fold_df["accuracy"].mean(),
        "std_accuracy":      fold_df["accuracy"].std(),
        "mean_macro_f1":     fold_df["macro_f1"].mean(),
        "std_macro_f1":      fold_df["macro_f1"].std(),
        "mean_macro_precision": fold_df["macro_precision"].mean(),
        "std_macro_precision":  fold_df["macro_precision"].std(),
        "mean_macro_recall":    fold_df["macro_recall"].mean(),
        "std_macro_recall":     fold_df["macro_recall"].std(),
    }
    summary_rows.append(row)

    print(f"\n  [{model_cfg['label']}]")
    print(f"    Accuracy     : {row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}")
    print(f"    Macro-F1     : {row['mean_macro_f1']:.4f} +/- {row['std_macro_f1']:.4f}")
    print(f"    Macro-Prec   : {row['mean_macro_precision']:.4f} +/- {row['std_macro_precision']:.4f}")
    print(f"    Macro-Recall : {row['mean_macro_recall']:.4f} +/- {row['std_macro_recall']:.4f}")

# ── Significance test: adapted-mBERT vs baseline-mBERT ────────────────────────
print("\n[4/5] Paired significance test (adapted-mBERT vs baseline-mBERT)...")

mbert_folds   = pd.DataFrame(all_fold_results.get("mBERT_baseline", []))
adapted_folds = pd.DataFrame(all_fold_results.get("mBERT_adapted",  []))

sig_result = {}
if len(mbert_folds) == N_FOLDS and len(adapted_folds) == N_FOLDS:
    # Sort by fold to ensure alignment
    mbert_f1   = mbert_folds.sort_values("fold")["macro_f1"].values
    adapted_f1 = adapted_folds.sort_values("fold")["macro_f1"].values

    t_stat, p_value = ttest_rel(adapted_f1, mbert_f1)
    sig_result = {
        "test":        "paired t-test",
        "statistic":   round(float(t_stat), 6),
        "p_value":     round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "alpha":       0.05,
        "n_folds":     N_FOLDS,
        "interpretation": (
            "Adapted mBERT significantly outperforms baseline mBERT on macro-F1."
            if p_value < 0.05 and t_stat > 0
            else "Adapted mBERT significantly underperforms baseline mBERT."
            if p_value < 0.05 and t_stat < 0
            else "No significant difference between adapted and baseline mBERT."
        )
    }
    print(f"    Test          : paired t-test on macro-F1 across {N_FOLDS} folds")
    print(f"    Adapted mean  : {adapted_f1.mean():.4f}")
    print(f"    Baseline mean : {mbert_f1.mean():.4f}")
    print(f"    t-statistic   : {t_stat:.4f}")
    print(f"    p-value       : {p_value:.4f}")
    print(f"    Significant   : {'YES' if p_value < 0.05 else 'NO'}  (alpha=0.05)")
    print(f"    Interpretation: {sig_result['interpretation']}")
else:
    print(f"  [!] Cannot run significance test: mBERT folds={len(mbert_folds)}, adapted folds={len(adapted_folds)}")

# ── Save Table 6 ───────────────────────────────────────────────────────────────
df_summary = pd.DataFrame(summary_rows)
for col in df_summary.select_dtypes(include='float').columns:
    df_summary[col] = df_summary[col].round(6)
df_summary.to_csv(TABLES / "table6_classification_results.csv", index=False, encoding="utf-8")
print(f"\n    Saved -> results/tables/table6_classification_results.csv")

# Save significance test results
if sig_result:
    with open(TABLES / "table6_significance_test.json", "w", encoding="utf-8") as f:
        json.dump(sig_result, f, indent=2)
    print(f"    Saved -> results/tables/table6_significance_test.json")

# Save per-fold results
if fold_checkpoint_path.exists():
    import shutil
    shutil.copy(fold_checkpoint_path, TABLES / "table6_per_fold_results.csv")
    print(f"    Saved -> results/tables/table6_per_fold_results.csv")

# ── Figure 5 ───────────────────────────────────────────────────────────────────
print("\n[5/5] Generating fig5_model_performance_comparison.png...")

model_labels = [r["model"] for r in summary_rows]
metrics = {
    "Accuracy":       ("mean_accuracy",     "std_accuracy"),
    "Macro-F1":       ("mean_macro_f1",     "std_macro_f1"),
    "Macro-Precision":("mean_macro_precision","std_macro_precision"),
    "Macro-Recall":   ("mean_macro_recall",  "std_macro_recall"),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')
axes = axes.flatten()
colors = [m["color"] for m in MODELS_TO_EVAL[:len(model_labels)]]

for ax_idx, (metric_name, (mean_col, std_col)) in enumerate(metrics.items()):
    ax = axes[ax_idx]
    ax.set_facecolor('#16213e')

    means = [r[mean_col] for r in summary_rows]
    stds  = [r[std_col]  for r in summary_rows]
    x     = np.arange(len(model_labels))

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors[:len(model_labels)], edgecolor='white',
                  linewidth=0.5, alpha=0.85,
                  error_kw=dict(ecolor='white', elinewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9, color='white', rotation=10, ha='right')
    ax.set_ylabel(metric_name, fontsize=11, color='white')
    ax.set_title(metric_name, fontsize=12, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_ylim(0, min(1.0, max(means) * 1.2) if means else 1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    # Annotate bar values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + std + 0.005,
                f'{mean:.3f}', ha='center', va='bottom',
                fontsize=9, color='white', fontweight='bold')

# Significance annotation on F1 plot
if sig_result and len(model_labels) >= 2:
    ax = axes[1]
    sig_text = (f"paired t-test: p={sig_result['p_value']:.4f} "
                f"({'*sig*' if sig_result['significant'] else 'n.s.'})")
    ax.text(0.5, 0.02, sig_text, transform=ax.transAxes, ha='center',
            fontsize=9, color='#aaaacc', style='italic')

fig.suptitle(
    f"Model Performance Comparison: {N_FOLDS}-Fold Stratified CV\n"
    f"(3,000-row dev subset | {N_EPOCHS} epochs | CPU)",
    fontsize=13, color='white', fontweight='bold'
)
plt.tight_layout()
fig.savefig(FIGURES / "fig5_model_performance_comparison.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"    Saved -> results/figures/fig5_model_performance_comparison.png")

# ── Log ────────────────────────────────────────────────────────────────────────
for row in summary_rows:
    append_log({
        "date": datetime.datetime.now().isoformat(),
        "phase": "Phase7",
        "model": row["model_id"],
        "dataset_version": "dev_subset_3000",
        "sample_size": len(df),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": N_EPOCHS,
        "seed": RANDOM_SEED,
        "result": (f"macro_f1={row['mean_macro_f1']:.4f}+/-{row['std_macro_f1']:.4f}, "
                   f"acc={row['mean_accuracy']:.4f}+/-{row['std_accuracy']:.4f}")
    })

print("\n" + "=" * 60)
print("PHASE 7 COMPLETE")
print("=" * 60)
for row in summary_rows:
    print(f"  [{row['model']:30}]  "
          f"F1={row['mean_macro_f1']:.4f}+/-{row['std_macro_f1']:.4f}  "
          f"Acc={row['mean_accuracy']:.4f}+/-{row['std_accuracy']:.4f}")
if sig_result:
    print(f"\n  Significance test: p={sig_result['p_value']:.4f}  "
          f"({'significant' if sig_result['significant'] else 'not significant'})")
print(f"\n  Output: table6_classification_results.csv, fig5_model_performance_comparison.png")
print(f"\n  [!] IMPORTANT: These results use model-predicted labels.")
print(f"      Rerun evaluation on human_audit_sample.csv when labels are returned.")
