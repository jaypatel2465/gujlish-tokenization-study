"""
run_phase7.py — Standalone Phase 7 Runner for GPU Machine
===========================================================
This script is the ONLY file you need to run on a GPU computer.

HOW TO USE (step by step):
1. Clone the repo:
       git clone https://github.com/YOUR_USERNAME/gujlish-tokenization-study.git
       cd gujlish-tokenization-study

2. Install PyTorch with CUDA (visit https://pytorch.org to get the right command):
       # Example for CUDA 12.1:
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. Install other dependencies:
       pip install -r requirements.txt

4. Run this script:
       python run_phase7.py

5. Wait for it to finish (usually 5-15 minutes on a modern GPU).
   Results are saved to: results/tables/table6_classification_results.csv
                         results/figures/fig5_model_performance_comparison.png

That's it! The script will handle everything automatically.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache")

# ─── Step 0: Quick environment check ──────────────────────────────────────────
print("=" * 65)
print("  Gujlish Tokenization Study — Phase 7 Setup & Runner")
print("=" * 65)

# Check Python version
import platform
print(f"\n  Python  : {platform.python_version()}")
print(f"  OS      : {platform.system()} {platform.release()}")

# Check PyTorch + GPU
try:
    import torch
    print(f"  PyTorch : {torch.__version__}")
    if torch.cuda.is_available():
        name  = torch.cuda.get_device_name(0)
        vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU     : ✅ {name}  ({vram:.1f} GB VRAM)")
        print(f"  CUDA    : {torch.version.cuda}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  GPU     : ✅ Apple Silicon MPS")
    else:
        print("  GPU     : ❌ Not found — will run on CPU (slow!)")
        ans = input("\n  Continue anyway on CPU? This will take 4-8 hours. [y/N] ").strip().lower()
        if ans != "y":
            print("  Aborted. Install CUDA PyTorch and re-run.")
            sys.exit(0)
except ImportError:
    print("\n  ❌ PyTorch is NOT installed!")
    print("     Install it first: https://pytorch.org/get-started/locally/")
    sys.exit(1)

# Check required packages
REQUIRED = ["transformers", "datasets", "sklearn", "scipy", "matplotlib", "pandas", "numpy"]
missing = []
for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"\n  ❌ Missing packages: {', '.join(missing)}")
    print("     Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n  ✅ All dependencies satisfied.")

# ─── Step 1: Check that pre-computed data files exist ─────────────────────────
print("\n" + "=" * 65)
print("  Checking required data files...")
print("=" * 65)

required_files = {
    "data/processed/dev_subset.csv"           : "3000-row dev subset for fine-tuning",
    "data/processed/gujlish_vocab_additions.txt": "Gujlish vocabulary additions list",
    "models/mbert_adapted/config.json"        : "Adapted mBERT tokenizer config",
    "models/mbert_adapted/tokenizer.json"     : "Adapted mBERT tokenizer",
}

all_ok = True
for rel_path, desc in required_files.items():
    fpath = ROOT / rel_path
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024
        print(f"  ✅  {rel_path:<50}  ({size_kb:.0f} KB)")
    else:
        print(f"  ❌  MISSING: {rel_path}")
        print(f"       ({desc})")
        all_ok = False

if not all_ok:
    print("\n  ❌ Some required files are missing!")
    print("     Make sure you cloned the full repo and that the repo owner")
    print("     has committed data/processed/ and models/mbert_adapted/ (tokenizer files).")
    print("     The large model.safetensors (~680 MB) is downloaded automatically below.")
    choice = input("\n  Attempt to continue anyway? [y/N] ").strip().lower()
    if choice != "y":
        sys.exit(1)

# ─── Step 2: Download mBERT model weights if not present ──────────────────────
# The tokenizer is in models/mbert_adapted/ (committed to git).
# The model weights (~680 MB) are too large for git; download from HuggingFace.
model_weights = ROOT / "models" / "mbert_adapted" / "model.safetensors"
if not model_weights.exists():
    print("\n" + "=" * 65)
    print("  Downloading mBERT model weights (~680 MB)...")
    print("  (This only happens once; subsequent runs use local cache)")
    print("=" * 65)
    try:
        from transformers import BertForSequenceClassification, AutoTokenizer
        # First load the tokenizer from adapted dir
        tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "mbert_adapted"))
        vocab_size = len(tok)
        print(f"  Tokenizer loaded. Vocab size: {vocab_size:,}")

        # Download and resize model
        print("  Downloading bert-base-multilingual-cased weights...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        model.resize_token_embeddings(vocab_size)
        print(f"  Embedding matrix resized to {vocab_size:,}.")

        # Save adapted model back
        model.save_pretrained(str(ROOT / "models" / "mbert_adapted"))
        print("  ✅ Adapted model weights saved to models/mbert_adapted/")
    except Exception as e:
        print(f"\n  ❌ Failed to download model weights: {e}")
        print("     Check your internet connection and try again.")
        sys.exit(1)
else:
    size_mb = model_weights.stat().st_size / 1e6
    print(f"\n  ✅  models/mbert_adapted/model.safetensors  ({size_mb:.0f} MB) — already present")

# ─── Step 3: Run Phase 7 ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Launching Phase 7 — Sentiment Classification Experiment")
print("=" * 65)
print("  Outputs will be saved to:")
print("    results/tables/table6_classification_results.csv")
print("    results/tables/table6_significance_test.json")
print("    results/figures/fig5_model_performance_comparison.png")
print()

phase7_script = ROOT / "scripts" / "phase7_classification.py"
result = subprocess.run(
    [sys.executable, str(phase7_script)],
    cwd=str(ROOT)
)

if result.returncode == 0:
    print("\n" + "=" * 65)
    print("  ✅ Phase 7 COMPLETE!")
    print("=" * 65)

    # Print the summary table
    table6 = ROOT / "results" / "tables" / "table6_classification_results.csv"
    if table6.exists():
        import csv
        print("\n  === RESULTS SUMMARY ===")
        with open(table6, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model  = row.get("model", "?")
                f1     = float(row.get("mean_macro_f1",  0))
                acc    = float(row.get("mean_accuracy",  0))
                f1_std = float(row.get("std_macro_f1",   0))
                print(f"  {model:<30}  F1={f1:.4f}±{f1_std:.4f}  Acc={acc:.4f}")
    print()
    print("  Please send the following files back to the project owner:")
    print("    results/tables/table6_classification_results.csv")
    print("    results/tables/table6_significance_test.json")
    print("    results/tables/table6_per_fold_results.csv")
    print("    results/figures/fig5_model_performance_comparison.png")
    print("    logs/phase7_fold_results.csv")
else:
    print(f"\n  ❌ Phase 7 exited with error code {result.returncode}.")
    print("     Check the output above for error details.")
    sys.exit(result.returncode)
