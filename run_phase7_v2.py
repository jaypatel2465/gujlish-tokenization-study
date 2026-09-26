"""
run_phase7_v2.py  —  Full Dataset Run (GPU Required)
=====================================================
WHAT'S DIFFERENT FROM THE ORIGINAL run_phase7.py:
  1. Uses FULL 21,346-sentence dataset (not the 3,000 dev_subset)
  2. Class-weighted CrossEntropyLoss — fixes negative class imbalance
  3. 5 epochs (was 3)
  4. Saves to Output/v2_* so it doesn't overwrite previous results
  5. Runs on CUDA GPU (auto-detected)

Same models as before:
  1. mBERT baseline   (bert-base-multilingual-cased)
  2. MuRIL baseline   (google/muril-base-cased)
  3. mBERT adapted    (our 75-token vocabulary-adapted version)

Research question preserved:
  "Does vocabulary adaptation of mBERT reduce fragmentation AND
   improve sentiment classification over baseline mBERT?"

Setup:
  git clone https://github.com/jaypatel2465/gujlish-tokenization-study
  cd gujlish-tokenization-study
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  python run_phase7_v2.py
"""

import sys, io, os, json, csv, time, datetime, shutil, warnings, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).parent

os.environ["HF_HOME"]             = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"]  = str(ROOT / "hf_cache")

OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 65)
print("  PHASE 7 v2 — 50k Dataset + 552-Token Adapted mBERT")
print("=" * 65)

# ── GPU Check ────────────────────────────────────────────────────────────────
import torch
print(f"\n[1/6] Checking hardware...")
if torch.cuda.is_available():
    DEVICE     = "cuda"
    GPU_NAME   = torch.cuda.get_device_name(0)
    VRAM_GB    = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH_SIZE = 32 if VRAM_GB >= 10 else 16
    USE_FP16   = True
    USE_CPU    = False
    print(f"  GPU  : {GPU_NAME}  ({VRAM_GB:.1f} GB VRAM)")
    print(f"  Batch: {BATCH_SIZE} (auto-scaled)")
else:
    print("  WARNING: No CUDA GPU found — running on CPU (will be very slow)")
    DEVICE     = "cpu"
    GPU_NAME   = "CPU"
    VRAM_GB    = 0
    BATCH_SIZE = 16
    USE_FP16   = False
    USE_CPU    = True

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_FOLDS       = 5
N_EPOCHS      = 5          # ↑ was 3, now 5
LEARNING_RATE = 2e-5
MAX_SEQ_LEN   = 128

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Load FULL dataset ─────────────────────────────────────────────────────────
print(f"\n[2/6] Loading dataset...")

DATA_PROCESSED = ROOT / "data" / "processed"

# Priority: merged 50k dataset > single datasets > huggingface
DATASET_1 = DATA_PROCESSED / "filtered_clean.csv"
DATASET_2 = DATA_PROCESSED / "expanded_dev_subset_full.csv"
DEV_SUBSET = DATA_PROCESSED / "dev_subset.csv"

frames = []
if DATASET_1.exists():
    print(f"  Found {DATASET_1.name}")
    df1 = pd.read_csv(DATASET_1, encoding='utf-8-sig', low_memory=False)
    frames.append(df1)
if DATASET_2.exists():
    print(f"  Found {DATASET_2.name}")
    df2 = pd.read_csv(DATASET_2, encoding='utf-8-sig', low_memory=False)
    frames.append(df2)

if len(frames) > 0:
    df_raw = pd.concat(frames, ignore_index=True)
    col = 'sentiment_label' if 'sentiment_label' in df_raw.columns else 'label'
    df = df_raw[df_raw[col].notna()].copy()
    dataset_name = "merged_50k" if len(frames) > 1 else frames[0].name
    print(f"  Using combined dataset: {len(df):,} labeled rows")
elif DEV_SUBSET.exists():
    df = pd.read_csv(DEV_SUBSET, encoding='utf-8', low_memory=False)
    dataset_name = "dev_subset_3000"
    print(f"  WARNING: Using 3k dev_subset — full dataset not found")
else:
    print("  No local dataset found — downloading from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("ShrutiPatel3011/gujarati-english-codemixed-sentiment")
    frames = [split.to_pandas() for split in ds.values()]
    df = pd.concat(frames, ignore_index=True)
    dataset_name = "hf_download"

# Standardize column names — filtered_clean.csv uses 'text' and 'sentiment_label'
if 'text' not in df.columns:
    text_col = next((c for c in df.columns if 'text' in c.lower()), df.columns[0])
    df = df.rename(columns={text_col: 'text'})

if 'sentiment_label' not in df.columns and 'label' in df.columns:
    df = df.rename(columns={'label': 'sentiment_label'})

df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].notna() & (df['text'] != '') & df['sentiment_label'].notna()].copy()
df = df.reset_index(drop=True)

# Encode labels
label2id = {lbl: i for i, lbl in enumerate(sorted(df['sentiment_label'].unique()))}
id2label = {v: k for k, v in label2id.items()}
df['label'] = df['sentiment_label'].map(label2id)
NUM_LABELS = len(label2id)

print(f"  Total rows        : {len(df):,}")
print(f"  Label encoding    : {label2id}")
print(f"  Class distribution:")
for lbl, idx in label2id.items():
    cnt = (df['label'] == idx).sum()
    print(f"    {lbl:15}: {cnt:5,}  ({cnt/len(df)*100:.1f}%)")

# ── Model registry ────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
MODELS_TO_EVAL = [
    {
        "name":        "mBERT_baseline",
        "model_path":  "bert-base-multilingual-cased",
        "label":       "mBERT (baseline)",
        "color":       "#e94560",
    },
    {
        "name":        "MuRIL_baseline",
        "model_path":  "google/muril-base-cased",
        "label":       "MuRIL (baseline)",
        "color":       "#0f3460",
    },
    {
        "name":        "mBERT_adapted",
        "model_path":  str(MODELS_DIR / "mbert_adapted_500"),
        "label":       "mBERT (adapted, 552 tokens)",
        "color":       "#53d8fb",
    },
]

# Skip adapted model if not present — check 500-token version first, fall back to 75-token
if (MODELS_DIR / "mbert_adapted_500").exists():
    pass  # already set above
elif (MODELS_DIR / "mbert_adapted").exists():
    print("\n  NOTE: mbert_adapted_500 not found — falling back to 75-token mbert_adapted")
    for m in MODELS_TO_EVAL:
        if m['name'] == 'mBERT_adapted':
            m['model_path'] = str(MODELS_DIR / "mbert_adapted")
            m['label'] = "mBERT (adapted, 75 tokens)"
else:
    print("\n  NOTE: No adapted model found — skipping")
    MODELS_TO_EVAL = [m for m in MODELS_TO_EVAL if m['name'] != 'mBERT_adapted']

print(f"\n[3/6] Models to evaluate:")
for m in MODELS_TO_EVAL:
    print(f"  ✓ {m['name']}")

# ── Imports for training ──────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import ttest_rel, wilcoxon
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset as HFDataset

# ── Custom Trainer with class-weighted loss ───────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropyLoss to fix negative class imbalance."""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def tokenize_fn(texts, labels, tokenizer):
    enc = tokenizer(list(texts), truncation=True, padding=True,
                    max_length=MAX_SEQ_LEN, return_tensors=None)
    enc["labels"] = list(labels)
    return HFDataset.from_dict(enc)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":        accuracy_score(labels, preds),
        "macro_f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall":    recall_score(labels, preds, average="macro", zero_division=0),
    }

# ── 5-Fold Cross-Validation ───────────────────────────────────────────────────
print(f"\n[4/6] Running {N_FOLDS}-fold cross-validation...")
print(f"  Epochs      : {N_EPOCHS}")
print(f"  Dataset size: {len(df):,}")
print(f"  Train/fold  : ~{int(len(df)*0.8):,}  |  Val/fold: ~{int(len(df)*0.2):,}")

skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
texts_all  = df["text"].values
labels_all = df["label"].values

# Checkpoint file
ckpt_path = OUTPUT_DIR / "v2_fold_checkpoint.csv"
ckpt_fields = ["model_name","fold","accuracy","macro_precision","macro_recall","macro_f1","timestamp"]

def load_done():
    done = {}
    if ckpt_path.exists():
        for row in csv.DictReader(open(ckpt_path, encoding='utf-8')):
            key = (row['model_name'], int(row['fold']))
            done[key] = {k: float(row[k]) for k in ['accuracy','macro_precision','macro_recall','macro_f1']}
    return done

def save_fold(model_name, fold, acc, prec, rec, f1):
    write_hdr = not ckpt_path.exists()
    with open(ckpt_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ckpt_fields)
        if write_hdr: w.writeheader()
        w.writerow({"model_name": model_name, "fold": fold,
                    "accuracy": round(acc,6), "macro_precision": round(prec,6),
                    "macro_recall": round(rec,6), "macro_f1": round(f1,6),
                    "timestamp": datetime.datetime.now().isoformat()})

done_folds = load_done()
if done_folds:
    print(f"\n  Resuming — {len(done_folds)} folds already done.")

all_results = {m["name"]: {} for m in MODELS_TO_EVAL}
# Pre-load checkpoint results
for (mname, fold_num), metrics in done_folds.items():
    if mname in all_results:
        all_results[mname][fold_num] = metrics

total_start = time.time()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts_all, labels_all)):
    fold_num = fold_idx + 1
    train_texts  = texts_all[train_idx]
    val_texts    = texts_all[val_idx]
    train_labels = labels_all[train_idx]
    val_labels   = labels_all[val_idx]

    print(f"\n  ════ Fold {fold_num}/{N_FOLDS}  (train={len(train_texts):,}  val={len(val_texts):,}) ════")

    # Compute class weights from training labels (handles imbalance)
    weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    weights_tensor = torch.FloatTensor(weights)

    for model_cfg in MODELS_TO_EVAL:
        mname = model_cfg["name"]

        if (mname, fold_num) in done_folds:
            res = done_folds[(mname, fold_num)]
            print(f"    [{mname}] fold {fold_num} — already done (F1={res['macro_f1']:.4f})")
            continue

        print(f"\n    [{mname}] Loading model...", flush=True)
        t0 = time.time()

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_path"], use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_cfg["model_path"],
                num_labels=NUM_LABELS,
                ignore_mismatched_sizes=True,
                id2label=id2label,
                label2id=label2id,
            )
        except Exception as e:
            print(f"    [{mname}] ERROR loading model: {e}")
            continue

        train_ds = tokenize_fn(train_texts, train_labels, tokenizer)
        val_ds   = tokenize_fn(val_texts,   val_labels,   tokenizer)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)

        tmp_dir = str(ROOT / f"tmp_{mname}_fold{fold_num}")
        args = TrainingArguments(
            output_dir                  = tmp_dir,
            num_train_epochs            = N_EPOCHS,
            per_device_train_batch_size = BATCH_SIZE,
            per_device_eval_batch_size  = BATCH_SIZE,
            learning_rate               = LEARNING_RATE,
            weight_decay                = 0.01,
            eval_strategy               = "epoch",
            save_strategy               = "no",
            load_best_model_at_end      = False,
            seed                        = RANDOM_SEED,
            logging_steps               = 50,
            report_to                   = "none",
            use_cpu                     = USE_CPU,
            no_cuda                     = USE_CPU,
            fp16                        = USE_FP16,
            dataloader_num_workers      = 0,
            gradient_accumulation_steps = max(1, 32 // BATCH_SIZE),
            warmup_ratio                = 0.1,
        )

        trainer = WeightedTrainer(
            class_weights   = weights_tensor,
            model           = model,
            args            = args,
            train_dataset   = train_ds,
            eval_dataset    = val_ds,
            tokenizer       = tokenizer,
            data_collator   = collator,
            compute_metrics = compute_metrics,
        )

        print(f"    [{mname}] Training...", flush=True)
        trainer.train()

        eval_res = trainer.evaluate()
        acc  = eval_res.get("eval_accuracy",        0.0)
        prec = eval_res.get("eval_macro_precision", 0.0)
        rec  = eval_res.get("eval_macro_recall",    0.0)
        f1   = eval_res.get("eval_macro_f1",        0.0)
        elapsed = time.time() - t0

        print(f"    [{mname}] fold {fold_num}: F1={f1:.4f}  Acc={acc:.4f}  ({elapsed/60:.1f}min)")

        save_fold(mname, fold_num, acc, prec, rec, f1)
        all_results[mname][fold_num] = {"accuracy": acc, "macro_precision": prec,
                                        "macro_recall": rec, "macro_f1": f1}
        done_folds[(mname, fold_num)] = all_results[mname][fold_num]

        del model, trainer
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree(tmp_dir, ignore_errors=True)

total_time = time.time() - total_start

# ── Aggregate results ─────────────────────────────────────────────────────────
print(f"\n[5/6] Aggregating results...")

summary_rows = []
for model_cfg in MODELS_TO_EVAL:
    mname = model_cfg["name"]
    folds = all_results[mname]
    if not folds:
        continue

    fold_df = pd.DataFrame(list(folds.values()))
    row = {
        "model":                 model_cfg["label"],
        "model_id":              mname,
        "dataset":               dataset_name,
        "n_folds":               len(fold_df),
        "mean_accuracy":         fold_df["accuracy"].mean(),
        "std_accuracy":          fold_df["accuracy"].std(),
        "mean_macro_f1":         fold_df["macro_f1"].mean(),
        "std_macro_f1":          fold_df["macro_f1"].std(),
        "mean_macro_precision":  fold_df["macro_precision"].mean(),
        "std_macro_precision":   fold_df["macro_precision"].std(),
        "mean_macro_recall":     fold_df["macro_recall"].mean(),
        "std_macro_recall":      fold_df["macro_recall"].std(),
    }
    summary_rows.append(row)
    print(f"\n  [{model_cfg['label']}]")
    print(f"    Macro-F1 : {row['mean_macro_f1']:.4f} ± {row['std_macro_f1']:.4f}")
    print(f"    Accuracy : {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")

# ── Significance tests ────────────────────────────────────────────────────────
print(f"\n[6/6] Statistical significance tests...")

def get_f1_array(model_name):
    folds = all_results.get(model_name, {})
    if len(folds) < N_FOLDS:
        return None
    return np.array([folds[f]['macro_f1'] for f in range(1, N_FOLDS+1)])

mbert_f1   = get_f1_array("mBERT_baseline")
adapted_f1 = get_f1_array("mBERT_adapted")
muril_f1   = get_f1_array("MuRIL_baseline")

sig_results = {}

def run_tests(name_a, f1_a, name_b, f1_b):
    if f1_a is None or f1_b is None:
        return None
    t_stat, p_ttest  = ttest_rel(f1_a, f1_b)
    try:
        _, p_wilcoxon = wilcoxon(f1_a, f1_b)
    except Exception:
        p_wilcoxon = 1.0
    diff = f1_a.mean() - f1_b.mean()
    from numpy import std
    cohens_d = diff / std(f1_a - f1_b) if std(f1_a - f1_b) > 0 else 0
    result = {
        "comparison":     f"{name_a} vs {name_b}",
        "mean_diff":      round(float(diff), 6),
        "cohens_d":       round(float(cohens_d), 4),
        "paired_ttest_p": round(float(p_ttest), 6),
        "wilcoxon_p":     round(float(p_wilcoxon), 6),
        "significant_ttest":    bool(p_ttest < 0.05),
        "significant_wilcoxon": bool(p_wilcoxon < 0.05),
        "n_folds": N_FOLDS,
    }
    print(f"\n  {name_a} vs {name_b}")
    print(f"    Mean diff (A-B) : {diff:+.4f}")
    print(f"    Cohen's d       : {cohens_d:.4f}")
    print(f"    t-test p-value  : {p_ttest:.4f}  {'✓ significant' if p_ttest < 0.05 else '✗ not significant'}")
    print(f"    Wilcoxon p      : {p_wilcoxon:.4f}  {'✓ significant' if p_wilcoxon < 0.05 else '✗ not significant'}")
    return result

if adapted_f1 is not None:
    sig_results["adapted_vs_baseline"] = run_tests("mBERT_adapted", adapted_f1, "mBERT_baseline", mbert_f1)
if muril_f1 is not None and mbert_f1 is not None:
    sig_results["muril_vs_baseline"] = run_tests("MuRIL", muril_f1, "mBERT_baseline", mbert_f1)

# ── Save outputs ──────────────────────────────────────────────────────────────
print(f"\n  Saving results...")

df_summary = pd.DataFrame(summary_rows)
for col in df_summary.select_dtypes('float').columns:
    df_summary[col] = df_summary[col].round(6)
df_summary.to_csv(OUTPUT_DIR / "v2_classification_results.csv", index=False)

with open(OUTPUT_DIR / "v2_significance_tests.json", 'w', encoding='utf-8') as f:
    json.dump({
        "metadata": {
            "timestamp":    datetime.datetime.now().isoformat(),
            "gpu":          GPU_NAME,
            "dataset":      dataset_name,
            "n_rows":       len(df),
            "n_folds":      N_FOLDS,
            "n_epochs":     N_EPOCHS,
            "batch_size":   BATCH_SIZE,
            "class_weighted_loss": True,
            "total_time_minutes": round(total_time / 60, 1)
        },
        "significance": sig_results
    }, f, indent=2, ensure_ascii=False)

shutil.copy(ckpt_path, OUTPUT_DIR / "v2_per_fold_results.csv")

# ── Figure ────────────────────────────────────────────────────────────────────
model_labels = [r['model'] for r in summary_rows]
colors = [m['color'] for m in MODELS_TO_EVAL if m['name'] in {r['model_id'] for r in summary_rows}]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('#1a1a2e')
for ax_idx, (metric_name, mean_col, std_col) in enumerate([
    ("Macro-F1",  "mean_macro_f1",  "std_macro_f1"),
    ("Accuracy",  "mean_accuracy",  "std_accuracy"),
]):
    ax = axes[ax_idx]
    ax.set_facecolor('#16213e')
    means = [r[mean_col] for r in summary_rows]
    stds  = [r[std_col]  for r in summary_rows]
    x = np.arange(len(model_labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors[:len(model_labels)], edgecolor='white',
                  linewidth=0.5, alpha=0.85,
                  error_kw=dict(ecolor='white', elinewidth=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9, color='white', rotation=12, ha='right')
    ax.set_ylabel(metric_name, fontsize=11, color='white')
    ax.set_title(metric_name, fontsize=13, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ymin = max(0, min(means) - 0.05)
    ax.set_ylim(ymin, min(1.0, max(means) + 0.05))
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 0.003,
                f'{m:.4f}', ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

    # Add p-value annotation on F1 chart
    if ax_idx == 0 and sig_results.get("adapted_vs_baseline"):
        sr = sig_results["adapted_vs_baseline"]
        sig_txt = f"t-test: p={sr['paired_ttest_p']:.4f} ({'*sig*' if sr['significant_ttest'] else 'n.s.'})"
        ax.text(0.5, 0.02, sig_txt, transform=ax.transAxes, ha='center',
                fontsize=8, color='#aaffaa' if sr['significant_ttest'] else '#ffaaaa', style='italic')

fig.suptitle(
    f"mBERT Vocabulary Adaptation: filtered_clean ({len(df):,} sentences)\n"
    f"{N_FOLDS}-Fold CV | {N_EPOCHS} epochs | Class-Weighted Loss | {GPU_NAME}",
    fontsize=11, color='white', fontweight='bold'
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "v2_model_comparison.png", dpi=300,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 7 v2 — COMPLETE")
print("=" * 65)
print(f"  Dataset    : {dataset_name} ({len(df):,} rows)")
print(f"  GPU        : {GPU_NAME}")
print(f"  Total time : {total_time/60:.1f} minutes\n")

print(f"  {'Model':<30} {'Macro-F1':>12}  {'±':>7}  {'Accuracy':>10}")
print(f"  {'-'*63}")
for row in sorted(summary_rows, key=lambda r: r['mean_macro_f1'], reverse=True):
    print(f"  {row['model']:<30} {row['mean_macro_f1']:>12.4f}  {row['std_macro_f1']:>7.4f}  {row['mean_accuracy']:>10.4f}")

if sig_results.get("adapted_vs_baseline"):
    sr = sig_results["adapted_vs_baseline"]
    print(f"\n  Adapted vs Baseline: Δ={sr['mean_diff']:+.4f}  d={sr['cohens_d']:+.4f}  p={sr['paired_ttest_p']:.4f}  {'✓ SIGNIFICANT' if sr['significant_ttest'] else '✗ not significant'}")

print(f"\n  Files saved to Output/:")
print(f"    v2_classification_results.csv")
print(f"    v2_significance_tests.json")
print(f"    v2_per_fold_results.csv")
print(f"    v2_model_comparison.png")
print(f"\n  → Zip the Output/ folder and send back to Jay")
