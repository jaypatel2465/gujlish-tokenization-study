# LIMITATIONS

This file documents every methodological compromise made during the experiment. The student should include this content in the paper's Limitations section.

---

## 1. Label Quality (Critical)

- **97.2% of sentiment labels are `model_predicted`** (automated, not human-verified).
- Only 2.8% are `llm_annotated` — still not human ground truth.
- A 150-row human-audit sample (`data/processed/human_audit_sample.csv`) was generated but requires manual annotation by the student.
- **Impact**: All classification results in Phase 7 should be treated as preliminary until the human-audit subset is scored and compared. Final headline numbers must note this caveat.
- **Mitigation**: The paper should explicitly state that results reflect model-predicted label quality, not certified human annotation quality.

---

## 2. Compute Constraints (CPU Only)

- No GPU was available during experimentation.
- Phase 7 fine-tuning runs on CPU with a **3,000-row stratified dev subset** (not the full 21,346-row cleaned dataset).
- Training: 2 epochs per fold (not the 3–5 epochs typical in published work).
- **Impact**: Classification results are likely lower than what would be achieved with GPU and full dataset. Results represent a lower bound on model capability.
- **Mitigation**: All compute constraints are documented in `logs/experiment_log.csv`. The paper should explicitly state the hardware and sample size used.

---

## 3. Vocabulary Size (66 vs 75 words)

- The target vocabulary addition was **75 words**, but the pool of non-ASCII, fragmented, high-frequency words was exhausted at **66 words** after filtering.
- **Why**: Only 276 Gujarati-script words appeared ≥10 times AND were fragmented by mBERT; after ranking, the pool produced 66 candidates above the combined score cutoff.
- **Impact**: Marginally smaller vocabulary adaptation than planned. The direction of results is unaffected.
- **Mitigation**: Documented explicitly. The frequency threshold (≥10) could be lowered to reach 75, but this risks adding noisy/rare words. The conservative threshold is methodologically defensible.

---

## 4. Gujarati-English Dominance Filter

- Filter: `(gujarati_tokens + english_tokens) > 1.5 × hindi_tokens`
- This retained **98.3%** of the original dataset — a very high retention rate, suggesting the filter is lenient.
- **Risk**: Some Hindi-heavy sentences may have been retained. A stricter threshold (e.g., 2×) would be more conservative but reduce the dataset further.
- **Impact**: The research question is about Gujarati-English code-mixing; Hindi sentences in the retained set may introduce noise.
- **Mitigation**: The filter threshold and retention rate are reported in Table 1. Future work could compare results under stricter filters.

---

## 5. Token Count Imbalance (English Dominance)

- The filtered corpus is **84% English tokens**, with only 12% Gujarati.
- **Impact**: The corpus is more English-with-Gujarati-words than truly bilingual code-mixed text. This limits the generalizability of fragmentation findings to heavily Gujarati-dominant text.
- **Mitigation**: This is an inherent property of Gujarati YouTube comment data. The paper should frame results accordingly.

---

## 6. Class Imbalance

- Negative sentiment: only **6.2%** of the corpus.
- **Impact**: Models may achieve high accuracy by ignoring negative examples. Macro-F1 is the appropriate metric and is reported, but the minority class performance may be unreliable with small training folds.
- **Mitigation**: Stratified sampling used for all train/val splits. Class distribution reported explicitly.

---

## 7. Semantic Similarity (Phase 6) — Pre-Fine-Tuning Embeddings

- Phase 6 extracts embeddings from **non-fine-tuned** models.
- New token embeddings (for the 66 added words) are **randomly initialized** — they have no semantic meaning until fine-tuned.
- **Impact**: Phase 6 similarity results for newly-added tokens reflect random initialization noise, not learned representations. This is expected and should be stated clearly.
- **Mitigation**: Phase 6 results should be interpreted as a baseline measurement. Post-fine-tuning contextual analysis (Phase 9, extended) would provide more meaningful representations.

---

## 8. Phases Not Completed (Extended)

- **Phase 8** (Ablation study): Not run due to compute time constraints.
- **Phase 9** (Contextual embedding analysis): Not run due to compute time constraints.
- **Human audit labels**: Not yet incorporated into final Phase 7 metrics.

---

## 9. Single Dataset

- All results are from a single dataset (ShrutiPatel3011/gujarati-english-codemixed-sentiment, YouTube comments).
- **Impact**: Findings may not generalize to other Gujlish text genres (WhatsApp messages, news comments, etc.).
- **Mitigation**: The 44,672-sentence dataset from academic literature was requested for comparison but not received in time.
