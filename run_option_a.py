"""
run_option_a.py — GujaratiBERT Fine-Tuning (GPU Required)
==========================================================
This is the standalone script for the friend's GPU machine.

What it does:
  1. Downloads 4 models: mBERT (baseline), MuRIL, mBERT-adapted, GujaratiBERT
  2. Fine-tunes all 4 on the full Gujlish dataset using 5-fold CV
  3. Compares fragmentation rates across tokenizers
  4. Saves all results to Output/ folder

Requirements:
  - NVIDIA GPU with >= 6 GB VRAM (8 GB+ recommended)
  - Python 3.9+
  - Install: pip install torch transformers datasets scikit-learn pandas numpy tqdm matplotlib seaborn

Usage:
  python run_option_a.py

Expected time:
  RTX 3060 (12 GB):  ~3-4 hours
  RTX 4080/4090:     ~1-2 hours
  A100 (Colab):      ~45-60 min
"""

import sys, io, os, json, time, csv, warnings, gc
import sys; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).parent

# ── Cache to local folder (avoid C: drive filling up) ─────────────────────────
CACHE_DIR = ROOT / "hf_cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

OUTPUT_DIR = ROOT / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 65)
print("  GUJLISH RESEARCH — OPTION A: GujaratiBERT + Full Dataset")
print("=" * 65)

# ── Step 1: Check GPU ─────────────────────────────────────────────────────────
print("\n[1/7] Checking GPU...")
import torch
if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected!")
    print("This script requires an NVIDIA GPU with CUDA support.")
    print("If you have an NVIDIA GPU, install PyTorch with CUDA:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
DEVICE = torch.device("cuda")

# Auto-select batch size based on VRAM
BATCH_SIZE = 16 if gpu_vram >= 8 else 8
print(f"  Auto batch size: {BATCH_SIZE}")

# ── Step 2: Load dataset ──────────────────────────────────────────────────────
print("\n[2/7] Loading dataset...")

# Try expanded dataset first, fall back to original
EXPANDED = ROOT / "data" / "processed" / "combined_full_dataset.csv"
ORIGINAL = ROOT / "data" / "processed" / "gujlish_clean.csv"

if EXPANDED.exists():
    df_all = pd.read_csv(EXPANDED, encoding='utf-8-sig')
    df = df_all[df_all['label'].notna()].copy()
    print(f"  Using EXPANDED dataset: {len(df):,} labeled rows")
    print(f"  (+ {len(df_all) - len(df):,} unlabeled rows available for MLM)")
elif ORIGINAL.exists():
    df = pd.read_csv(ORIGINAL, encoding='utf-8-sig')
    print(f"  Using ORIGINAL dataset: {len(df):,} rows")
    print("  NOTE: Run phase1b_expand_dataset.py on Jay's laptop to get more data")
else:
    # Download fresh from HuggingFace
    print("  Local dataset not found — downloading from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("ShrutiPatel3011/gujarati-english-codemixed-sentiment")
    frames = []
    for split in ds.values():
        frames.append(split.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df = df[df['label'].notna()].copy()
    print(f"  Downloaded from HuggingFace: {len(df):,} rows")

# Normalize labels to 0/1/2
print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
label2id = {v: i for i, v in enumerate(sorted(df['label'].unique()))}
id2label = {v: k for k, v in label2id.items()}
df['label_id'] = df['label'].map(label2id)
NUM_LABELS = len(label2id)
print(f"  Classes: {label2id}")
print(f"  Total labeled: {len(df):,}")

# ── Step 3: Define models ─────────────────────────────────────────────────────
print("\n[3/7] Setting up models...")

MODELS = {
    "mBERT_baseline": {
        "hf_id":   "bert-base-multilingual-cased",
        "desc":    "mBERT baseline (no adaptation)",
        "adapted": False
    },
    "MuRIL_baseline": {
        "hf_id":   "google/muril-base-cased",
        "desc":    "MuRIL (multilingual Indian languages)",
        "adapted": False
    },
    "mBERT_adapted": {
        "hf_id":   str(ROOT / "models" / "mbert_adapted"),
        "desc":    "mBERT + 75 Gujarati tokens (our adaptation)",
        "adapted": True
    },
    "GujaratiBERT": {
        "hf_id":   "l3cube-pune/gujarati-bert",
        "desc":    "GujaratiBERT — pre-trained on Gujarati (L3Cube)",
        "adapted": False
    },
}

# Check if mBERT_adapted model exists locally; skip if not
adapted_path = ROOT / "models" / "mbert_adapted"
if not adapted_path.exists():
    print("  NOTE: mBERT_adapted not found locally — skipping it")
    MODELS.pop("mBERT_adapted")

for name, info in MODELS.items():
    print(f"  ✓ {name}: {info['desc']}")

# ── Step 4: Fragmentation analysis ───────────────────────────────────────────
print("\n[4/7] Tokenization fragmentation analysis...")
from transformers import AutoTokenizer

SAMPLE_SIZE = min(500, len(df))
sample_texts = df['text'].sample(SAMPLE_SIZE, random_state=42).tolist()

fragmentation_results = {}
for model_name, info in MODELS.items():
    try:
        tok = AutoTokenizer.from_pretrained(info['hf_id'], use_fast=True)
        total_words, fragmented_words = 0, 0
        for text in sample_texts:
            words = text.split()
            for word in words:
                tokens = tok.tokenize(word)
                total_words += 1
                if len(tokens) > 1:
                    fragmented_words += 1
        frag_rate = fragmented_words / max(total_words, 1)
        fragmentation_results[model_name] = {
            "fragmentation_rate": round(frag_rate, 4),
            "vocab_size": tok.vocab_size
        }
        print(f"  {model_name}: {frag_rate:.1%} fragmentation | vocab={tok.vocab_size:,}")
        del tok
    except Exception as e:
        print(f"  {model_name}: ERROR loading tokenizer — {e}")
        fragmentation_results[model_name] = {"fragmentation_rate": None, "vocab_size": None}

# Save fragmentation table
frag_df = pd.DataFrame(fragmentation_results).T.reset_index().rename(columns={'index': 'model'})
frag_df.to_csv(OUTPUT_DIR / "fragmentation_comparison.csv", index=False)

# ── Step 5: Fine-tuning with 5-fold CV ───────────────────────────────────────
print("\n[5/7] Fine-tuning all models (5-fold cross-validation)...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, accuracy_score

EPOCHS    = 4
MAX_LEN   = 128
LR        = 2e-5
N_FOLDS   = 5

class GujlishDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

def run_fold(model_hf_id, train_texts, train_labels, val_texts, val_labels):
    """Train one fold and return val metrics."""
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_hf_id,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    train_ds = GujlishDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds   = GujlishDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Class weights for imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    # Mixed precision training
    scaler = torch.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy  = accuracy_score(all_labels, all_preds)
    report    = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return {"macro_f1": macro_f1, "accuracy": accuracy, "report": report}

# Run all models
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
texts  = df['text'].tolist()
labels = df['label_id'].tolist()

all_results = {}
start_time  = time.time()

for model_name, info in MODELS.items():
    print(f"\n  ── {model_name} ({info['desc']}) ──")
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"    Fold {fold+1}/{N_FOLDS}...", end=' ', flush=True)
        t0 = time.time()
        try:
            result = run_fold(
                info['hf_id'],
                [texts[i] for i in train_idx], [labels[i] for i in train_idx],
                [texts[i] for i in val_idx],   [labels[i] for i in val_idx]
            )
            fold_scores.append(result['macro_f1'])
            print(f"F1={result['macro_f1']:.4f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            fold_scores.append(None)

    valid_scores = [s for s in fold_scores if s is not None]
    mean_f1 = np.mean(valid_scores) if valid_scores else 0
    std_f1  = np.std(valid_scores)  if valid_scores else 0
    print(f"  → {model_name}: mean F1 = {mean_f1:.4f} ± {std_f1:.4f}")

    all_results[model_name] = {
        "fold_scores":   fold_scores,
        "mean_macro_f1": round(mean_f1, 4),
        "std_macro_f1":  round(std_f1, 4),
        "desc":          info['desc']
    }

total_time = time.time() - start_time

# ── Step 6: Save all results ──────────────────────────────────────────────────
print("\n[6/7] Saving results...")

# Results JSON
with open(OUTPUT_DIR / "option_a_results.json", 'w', encoding='utf-8') as f:
    json.dump({
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu":             gpu_name,
        "total_time_min":  round(total_time / 60, 1),
        "dataset_size":    len(df),
        "epochs":          EPOCHS,
        "batch_size":      BATCH_SIZE,
        "n_folds":         N_FOLDS,
        "models":          all_results,
        "fragmentation":   fragmentation_results
    }, f, indent=2, ensure_ascii=False)

# CSV summary table
rows = []
for model_name, res in all_results.items():
    frag = fragmentation_results.get(model_name, {})
    rows.append({
        "Model":            model_name,
        "Description":      res['desc'],
        "Mean_Macro_F1":    res['mean_macro_f1'],
        "Std_Macro_F1":     res['std_macro_f1'],
        "Fragmentation_%":  round(frag.get('fragmentation_rate', 0) * 100, 2) if frag.get('fragmentation_rate') else "N/A",
        "Vocab_Size":       frag.get('vocab_size', "N/A")
    })
results_df = pd.DataFrame(rows).sort_values("Mean_Macro_F1", ascending=False)
results_df.to_csv(OUTPUT_DIR / "option_a_summary.csv", index=False)

# ── Step 7: Print final summary ───────────────────────────────────────────────
print("\n[7/7] FINAL RESULTS")
print("=" * 65)
print(f"{'Model':<20} {'Macro F1':>10} {'±':>5} {'Frag %':>8} {'Vocab':>9}")
print("-" * 65)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<20} {row['Mean_Macro_F1']:>10.4f} {row['Std_Macro_F1']:>5.4f} "
          f"{str(row['Fragmentation_%']):>8} {str(row['Vocab_Size']):>9}")
print("=" * 65)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"GPU used:   {gpu_name}")
print(f"Dataset:    {len(df):,} sentences")
print(f"\n✅ All results saved to: Output/")
print("Files to send back:")
print("  - Output/option_a_results.json")
print("  - Output/option_a_summary.csv")
print("  - Output/fragmentation_comparison.csv")
