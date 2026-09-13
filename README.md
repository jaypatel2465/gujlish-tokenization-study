# Gujlish Tokenization Fragmentation Study

**Paper**: *"Investigating the Impact of Subword Tokenization Fragmentation on Semantic Representation of Gujarati-English Code-Mixed Text"*

---

## 🚀 Quick Start for GPU Collaborators (Phase 7 only)

If someone shared this repo with you to run the final experiment on your GPU:

```bash
# 1. Clone
git clone https://github.com/jaypatel2465/gujlish-tokenization-study.git
cd gujlish-tokenization-study

# 2. Install PyTorch with CUDA (visit https://pytorch.org for the exact command for your CUDA version)
#    Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run — that's it!
python run_phase7.py
```

The script will auto-detect your GPU, set optimal batch sizes, download the ~680 MB model weights once, run 5-fold CV across 3 models, and tell you exactly which files to send back.

**Expected time**: ~5–15 min on a modern GPU (RTX 3060+), ~4–8 hours on CPU.

---

## Project Structure

```
gujlish-tokenization-study/
├── run_phase7.py                   # ← GPU collaborator runs THIS
├── requirements.txt
├── data/
│   ├── raw/                        # Downloaded raw dataset (git-ignored, large)
│   └── processed/
│       ├── dev_subset.csv          # 3000-row subset for Phase 7 fine-tuning ✅
│       ├── gujlish_vocab_additions.txt  # 75 Gujlish tokens added to mBERT ✅
│       └── ...                     # Other processed files (git-ignored, large)
├── scripts/
│   ├── phase1_dataset.py           # Stage 1: Download + filter dataset
│   ├── phase2_clean.py             # Stage 1: Clean dataset
│   ├── phase3_fragmentation.py     # Stage 2: Fragmentation baseline
│   ├── phase4_vocab_selection.py   # Stage 2: Gujlish vocab selection
│   ├── phase5_vocab_adapt.py       # Stage 2: Adapt mBERT tokenizer
│   ├── phase6_semantic.py          # Stage 3: Semantic similarity analysis
│   ├── phase7_classification.py    # Stage 3: Sentiment classification (GPU-optimised)
│   └── error_analysis.py           # Stage 3: Error analysis
├── models/
│   └── mbert_adapted/              # Adapted tokenizer + model weights
│       ├── config.json             # ✅ committed
│       ├── tokenizer.json          # ✅ committed
│       └── model.safetensors       # ❌ git-ignored (680 MB); auto-downloaded
├── results/
│   ├── tables/                     # CSV tables (Phases 1-6 already computed)
│   └── figures/                    # PNG figures (300 dpi)
├── logs/
│   └── experiment_log.csv          # One row per experiment run
├── RESULTS_SUMMARY.md
└── LIMITATIONS.md
```

---

## Full Pipeline (run in order on your own machine)

### Stage 1 — Data Foundation
```bash
python scripts/phase1_dataset.py
python scripts/phase2_clean.py
```
**Checkpoint A**: Review `results/tables/table1_dataset_statistics.csv` and `results/tables/table2_cleaning_report.csv`.  
Fill in `data/processed/human_audit_sample.csv` at your convenience.

### Stage 2 — Tokenization & Vocabulary
```bash
python scripts/phase3_fragmentation.py
python scripts/phase4_vocab_selection.py
python scripts/phase5_vocab_adapt.py
```
**Checkpoint B**: Review `results/tables/table3_top_fragmented_words.csv` — do the words make linguistic sense?

### Stage 3 — Experiments
```bash
python scripts/phase6_semantic.py
python scripts/phase7_classification.py    # GPU recommended
python scripts/error_analysis.py
```
**Checkpoint C**: Final review of all results.

---

## Key Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Gujlish dominance filter | `(guj+eng) > 1.5 × hindi` | Ensures clear Gujlish signal |
| Frequency threshold (Phase 4) | ≥ 10 occurrences | Avoids rare-word noise |
| Vocabulary additions | Top 75 words | Balanced coverage vs. model size |
| CV folds (Phase 7) | 5-fold stratified | Standard for robust evaluation |
| Epochs (Phase 7) | 3 (GPU) | Auto-set; falls back to 2 on CPU |
| Batch size (Phase 7) | 8–64 (auto) | Scales with available VRAM |
| FP16 / Half-precision | Enabled on CUDA | ~2× speed, same accuracy |
| Random seed | 42 | Fixed for reproducibility |

---

## Results Produced So Far (Phases 1–6 ✅)

| Output | File |
|---|---|
| Table 1 — Dataset statistics | `results/tables/table1_dataset_statistics.csv` |
| Table 2 — Fragmentation baseline (mBERT vs MuRIL) | `results/tables/table2_fragmentation_baseline.csv` |
| Table 3 — Top 20 fragmented Gujlish words | `results/tables/table3_top_fragmented_words.csv` |
| Table 4 — Before/after vocabulary adaptation | `results/tables/table4_before_after_fragmentation.csv` |
| Table 5 — Semantic similarity experiment | `results/tables/table5_semantic_similarity.csv` |
| Fig 1 — Subwords-per-word distribution | `results/figures/fig1_subwords_per_word_distribution.png` |
| Fig 2 — Top 20 fragmented words | `results/figures/fig2_top20_fragmented_words.png` |
| Fig 3 — Fragmentation before/after | `results/figures/fig3_fragmentation_before_after.png` |
| Fig 4 — Semantic similarity before/after | `results/figures/fig4_semantic_similarity_before_after.png` |

**Phase 7 pending** (Table 6, Fig 5) — requires GPU or overnight CPU run.

---

## Notes
- ~95%+ of sentiment labels are `model_predicted`, not human-annotated. Results are preliminary until human audit CSV is returned.
- The `hf_cache/` directory (~1.6 GB) is git-ignored. HuggingFace downloads models there automatically.
