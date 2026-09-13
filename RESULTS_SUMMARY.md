# RESULTS SUMMARY
**Paper**: *Investigating the Impact of Subword Tokenization Fragmentation on Semantic Representation of Gujarati-English Code-Mixed Text*

> This file is the single source of truth for writing the paper's Methodology, Results, and Discussion sections.
> Results are populated progressively as each phase completes.
> Values marked `[PENDING]` require the corresponding phase to finish.

---

## Table 1: Dataset Statistics

| Metric | Value |
|---|---|
| Raw rows downloaded | 21,729 |
| After Gujlish dominance filter (1.5×) | 21,359 (98.3% retained) |
| After cleaning (dedup, spam removal) | 21,346 (99.94% of filtered) |
| Dominance filter threshold | `(guj + eng) > 1.5 × hindi tokens` |
| Dev subset for experiments | 3,000 (stratified) |
| **Class: Positive** | 12,538 (58.7%) |
| **Class: Neutral** | 7,491 (35.1%) |
| **Class: Negative** | 1,317 (6.2%) |
| Avg sentence length (words) | 17.88 (median: 12) |
| Avg CMI score | 15.85 (max: 66.67) |
| English token share | 84.0% |
| Gujarati token share | 12.0% |
| Hindi token share | 4.0% |
| `model_predicted` labels | 20,765 (97.2%) |
| `llm_annotated` labels | 594 (2.8%) |

**Interpretation**: The filtered corpus is heavily English-dominant (84% English tokens), with Gujarati words embedded at ~12%. This reflects the natural code-mixing pattern in YouTube comments from Gujarati-speaking creators, where English serves as the matrix language and Gujarati is the embedded language. The severe class imbalance (negative = 6.2%) makes macro-F1 the appropriate primary metric for Phase 7.

---

## Table 2: Cleaning Report

| Step | Removed | % of Input |
|---|---|---|
| Empty/null rows | 0 | 0.00% |
| Missing label | 0 | 0.00% |
| Spam/pure-URL | 13 | 0.06% |
| Exact duplicates | 0 | 0.00% |
| **Total removed** | **13** | **0.06%** |
| **Remaining** | **21,346** | **99.94%** |

**Interpretation**: The dataset is very clean — only 13 spam/URL rows were removed. No spelling normalization was performed; natural variation (e.g., "chhe" vs "che") is preserved as it is part of the research signal.

---

## Figure 1: Subword Fragmentation Distribution

![Figure 1](results/figures/fig1_subwords_per_word_distribution.png)

**Interpretation**: See Table 2b below.

---

## Table 2b: Tokenizer Fragmentation Baseline

| Metric | mBERT | MuRIL |
|---|---|---|
| Vocab size | 119,547 | 197,258 |
| Unique words analyzed | 51,182 | 51,182 |
| Avg subwords/word | **2.659** | **2.323** |
| Median subwords/word | 2.0 | 2.0 |
| % words fragmented (2+ pieces) | **79.3%** | **70.0%** |
| % words single token | 20.7% | 30.0% |
| % words 3+ pieces | 43.5% | 33.4% |
| Max pieces (any word) | 66 | 65 |
| Avg fragmentation ratio (frag. words) | 3.091 | 2.889 |

**Interpretation**: mBERT fragments 79.3% of all unique words in this corpus — a strikingly high rate driven by Gujarati-script words that are almost entirely absent from its vocabulary. MuRIL performs better (70% fragmented) due to its larger, Indian-language-aware vocabulary. This confirms the hypothesis that standard multilingual tokenizers are ill-suited for code-mixed Indic text.

---

## Figure 2: Top 20 Fragmented Words

![Figure 2](results/figures/fig2_top20_fragmented_words.png)

---

## Table 3: Top Vocabulary Candidates Selected

| Selection Parameter | Value |
|---|---|
| Frequency threshold | ≥ 10 occurrences |
| Fragmentation filter | mBERT pieces ≥ 2 |
| ASCII exclusion | All pure-ASCII words excluded |
| Candidate pool after filters | 276 non-ASCII fragmented words |
| **Final selected** | **66 Gujarati-script words** |
| Combined score formula | `log₂(freq) × mBERT_pieces` |
| Score range | 14.34 – 30.25 |

> See `results/tables/table3_top_fragmented_words.csv` for the full ranked word list with tokenizations.

**Interpretation**: 66 Gujarati-script words were selected (target was 75 — the pool of non-ASCII fragmented high-frequency words was exhausted at 66, documented in LIMITATIONS.md). The top words show 4–7 mBERT subword pieces each, confirming severe fragmentation. All selected words are genuine Gujarati-script tokens, not English contractions (which were explicitly filtered).

---

## Table 4: Fragmentation Before vs After Adaptation

`[PENDING — Phase 5 completion]`

---

## Figure 3: Fragmentation Before/After

`[PENDING — Phase 5 completion]`

---

## Table 5: Semantic Similarity (Phase 6)

`[PENDING — Phase 6 completion]`

---

## Figure 4: Semantic Similarity Before/After

`[PENDING — Phase 6 completion]`

---

## Table 6: Classification Results (Phase 7)

`[PENDING — Phase 7 completion (5-fold CV, ~4-8 hours on CPU)]`

---

## Figure 5: Model Performance Comparison

`[PENDING — Phase 7 completion]`

---

## Statistical Significance

`[PENDING — Phase 7 completion]`

---

## Error Analysis

`[PENDING — Phase 7 + error_analysis.py completion]`

---

## Key Findings Summary

*(To be completed once all phases finish)*

1. **Fragmentation**: mBERT fragments 79.3% of Gujlish words vs MuRIL's 70.0% — confirming the research problem.
2. **Vocabulary selection**: 66 high-frequency Gujarati-script words selected with combined scoring (frequency × fragmentation severity).
3. **After adaptation**: `[PENDING]`
4. **Semantic similarity**: `[PENDING]`
5. **Classification**: `[PENDING]`
