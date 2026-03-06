# Operational Probe Comparison — Partner Identity vs Control


Generated: 2026-03-04 15:03


Compares **operational probe** (probing at the natural generation position, no reflective suffix)
accuracy between the Partner Identity dataset (`balanced_gpt`) and the Control dataset
(`nonsense_codeword`). If partner identity information is genuinely represented in the model's
operational context, the Partner Identity version should show above-chance accuracy while the Control version
should remain at chance.


**Column note:** The operational probe corresponds to `control_best_acc` /
`control_final_acc` in the CSV files (column naming predates the probe rename).
Ntest = 400 per layer.


## Statistical Methods


### Data Source


Each dataset version has a CSV file (`layerwise_probe_stats.csv`) containing per-layer probe
accuracy summaries. The **operational probe** is a linear classifier trained on LLaMA-2-13B-Chat
hidden states extracted at the natural generation position (the `[/INST]` token where the model
is about to produce its next response). No reflective suffix is appended — this tests whether partner
identity is represented in the model's ordinary operational context.


We compare two dataset versions:


  - **Partner Identity (`balanced_gpt`):** System prompt tells the model it is
  speaking with a specific human or AI partner. The operational probe should detect partner identity signal
  if the model internalizes this information during normal generation.
  - **Control (`nonsense_codeword`):** The same human/AI tokens appear in the system
  prompt but as a meaningless "session code word" rather than a partner identity. This controls
  for the mere presence of identity-related tokens. The operational probe should perform at chance.


For each version, we use the `control_best_acc` column (best test accuracy across 50 training
epochs) and `control_final_acc` column (final epoch test accuracy) from the CSV. Column names
use the pre-rename convention where "control" refers to the operational probe type.
Each accuracy value is a proportion of correctly classified test samples out of
Ntest = 400 (20% of ~2000 total samples = 50 agents x 40 conversations, stratified
train/test split).


### Test 1: Above-Chance Performance (Binomial Test)


**Question:** At each layer, does the operational probe classify partner type better than
random guessing (50%)?


**Procedure:**


  - Convert each layer's accuracy proportion to a success count:
  `k = round(accuracy x 400)`.
  - Run a one-sided exact binomial test (scipy.stats.binomtest(k, 400, 0.5,
  alternative=&quot;greater&quot;)) testing H1: accuracy > 0.5.
  - Collect all 41 raw p-values and apply Benjamini-Hochberg FDR correction
  (`statsmodels.stats.multitest.multipletests(..., method=&quot;fdr_bh&quot;)`) at &alpha; = 0.05.
  - FDR correction is applied *separately* per version (Partner Identity and Control each get
  their own family of 41 tests).


**Rationale:** The binomial test is the exact test for whether a proportion differs from a
known reference value, and is appropriate here because each test sample is an independent binary
classification. FDR correction controls the expected proportion of false discoveries across the 41
layers tested.


### Test 2: Between-Version Comparison (Two-Proportions Z-Test)


**Question:** At each layer, is the Partner Identity operational probe significantly more
accurate than the Control operational probe?


**Procedure:**


  - For each layer, convert both versions' accuracies to success counts as above.
  - Run a two-sided two-proportions z-test
  (statsmodels.stats.proportion.proportions_ztest([kPI, kCtrl],
  [400, 400], alternative=&quot;two-sided&quot;)).
  - Collect 41 raw p-values and apply Benjamini-Hochberg FDR correction at &alpha; = 0.05.


**Rationale:** The two-proportions z-test compares two independent proportions under the
assumption of large-sample normal approximation to the binomial — appropriate given
Ntest = 400. A two-sided test is used because we do not assume directionality a priori
(although we expect Partner Identity > Control). FDR correction is applied across all 41 layers.


### Test 3: Overall Paired T-Test


**Question:** Across all layers as a whole, does Partner Identity outperform Control?


**Procedure:**


  - Treat the 41 layers as paired observations: for each layer *i*, compute
  di = accPI(i) - accCtrl(i).
  - Run a paired t-test (`scipy.stats.ttest_rel`) on the 41 difference scores.
  - Compute Cohen's d for paired samples:
  d = mean(di) / SD(di).


**Rationale:** The per-layer z-tests above test each layer independently but have limited
power (Ntest = 400 per layer). The paired t-test pools evidence across all layers,
treating the layerwise accuracy profile as the unit of analysis. This is the appropriate omnibus test for
whether one version consistently outperforms the other. Cohen's d provides a standardized effect size.


### Layer Group Means


Layers are grouped into early (0–13, 14 layers), middle (14–27, 14 layers), and late
(28–40, 13 layers). Mean accuracy +/- SEM is reported per group to characterize where in the
network any differences emerge.


## 1. Figure



*[Figure: Operational Probe Comparison — Partner Identity vs Control — see HTML report]*


Figure 1. Operational probe accuracy by layer for Partner Identity (green) and Control (gray)
  datasets. Left: best test accuracy across 50 epochs. Right: final-epoch test accuracy.
  Light green bands mark layers where Partner Identity is significantly above chance (binomial test, FDR q<.05).
  Gold star marks the peak accuracy layer.


## 2. Summary Statistics


| Metric | PI Peak | Ctrl Peak | PI Mean | Ctrl Mean | Diff | Paired t | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 79.5% (L39) | 55.0% (L35) | 73.6% | 51.7% | +21.9% | t(40)=19.27 | <.0001 | 3.01 |
| Final Test Acc | 79.0% (L39) | 53.5% (L13) | 72.8% | 49.8% | +23.0% | t(40)=19.51 | <.0001 | 3.05 |


**Interpretation:** The Partner Identity version shows consistently higher operational probe accuracy
(peak 79.5% at layer 39) compared to the Control version
(peak 55.0% at layer 35). The paired t-test across 41 layers
confirms this difference is highly significant
(t(40)=19.27, p=

## 3. Layer Group Means


| Layer Group | Partner Identity (M +/- SEM) | Control (M +/- SEM) | Difference |
| --- | --- | --- | --- |
| early (0–13) | 67.4% +/- 2.5% | 51.8% +/- 0.3% | +15.7% |
| middle (14–27) | 77.3% +/- 0.3% | 51.1% +/- 0.4% | +26.2% |
| late (28–40) | 76.4% +/- 0.5% | 52.2% +/- 0.5% | +24.1% |


## 4. Above-Chance Performance (Binomial Tests)


One-sided binomial tests (H1: accuracy > 0.5) per layer, FDR corrected (Benjamini-Hochberg)
across 41 layers, separately per version.


**Partner Identity (balanced_gpt):** 41/41 layers above chance (FDR q<.05)


Significant layers: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40


**Control (nonsense_codeword):** 0/41 layers above chance (FDR q<.05)


No layers reached significance, consistent with operational probes learning no meaningful signal from nonsense context.


**Interpretation:** The Partner Identity version shows above-chance operational probe accuracy at 41 layers, concentrated in middle-to-late layers where partner identity information is expected to emerge. The Control version shows no above-chance layers, confirming that the nonsense codeword context does not provide the model with usable partner identity information at the operational position.


## 5. Between-Version Comparison (Z-Tests)


Two-proportions z-tests comparing Partner Identity vs Control operational probe accuracy at each layer,
FDR corrected across 41 layers.


**Significant layers (FDR q<.05):** 36/41


Layers: 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40


| Layer | Partner Identity | Control | z | p (raw) | p (FDR) |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.603 | 0.505 | 2.77 | 0.0055 | 0.0063 |
| 6 | 0.700 | 0.512 | 5.43 | <.0001 | <.0001 |
| 7 | 0.730 | 0.505 | 6.55 | <.0001 | <.0001 |
| 8 | 0.762 | 0.512 | 7.35 | <.0001 | <.0001 |
| 9 | 0.775 | 0.502 | 8.02 | <.0001 | <.0001 |
| 10 | 0.757 | 0.517 | 7.06 | <.0001 | <.0001 |
| 11 | 0.762 | 0.517 | 7.22 | <.0001 | <.0001 |
| 12 | 0.748 | 0.522 | 6.61 | <.0001 | <.0001 |
| 13 | 0.760 | 0.535 | 6.66 | <.0001 | <.0001 |
| 14 | 0.755 | 0.490 | 7.73 | <.0001 | <.0001 |
| 15 | 0.772 | 0.497 | 8.08 | <.0001 | <.0001 |
| 16 | 0.790 | 0.517 | 8.10 | <.0001 | <.0001 |
| 17 | 0.760 | 0.510 | 7.34 | <.0001 | <.0001 |
| 18 | 0.777 | 0.520 | 7.63 | <.0001 | <.0001 |
| 19 | 0.785 | 0.495 | 8.54 | <.0001 | <.0001 |
| 20 | 0.770 | 0.510 | 7.66 | <.0001 | <.0001 |
| 21 | 0.775 | 0.512 | 7.75 | <.0001 | <.0001 |
| 22 | 0.777 | 0.512 | 7.83 | <.0001 | <.0001 |
| 23 | 0.775 | 0.492 | 8.29 | <.0001 | <.0001 |
| 24 | 0.780 | 0.527 | 7.51 | <.0001 | <.0001 |
| 25 | 0.772 | 0.527 | 7.26 | <.0001 | <.0001 |
| 26 | 0.757 | 0.515 | 7.13 | <.0001 | <.0001 |
| 27 | 0.777 | 0.530 | 7.36 | <.0001 | <.0001 |
| 28 | 0.780 | 0.522 | 7.64 | <.0001 | <.0001 |
| 29 | 0.770 | 0.517 | 7.46 | <.0001 | <.0001 |
| 30 | 0.755 | 0.527 | 6.71 | <.0001 | <.0001 |
| 31 | 0.752 | 0.502 | 7.31 | <.0001 | <.0001 |
| 32 | 0.733 | 0.515 | 6.35 | <.0001 | <.0001 |
| 33 | 0.745 | 0.487 | 7.49 | <.0001 | <.0001 |
| 34 | 0.767 | 0.522 | 7.24 | <.0001 | <.0001 |
| 35 | 0.743 | 0.550 | 5.69 | <.0001 | <.0001 |
| 36 | 0.767 | 0.532 | 6.97 | <.0001 | <.0001 |
| 37 | 0.767 | 0.507 | 7.65 | <.0001 | <.0001 |
| 38 | 0.762 | 0.540 | 6.60 | <.0001 | <.0001 |
| 39 | 0.795 | 0.530 | 7.93 | <.0001 | <.0001 |
| 40 | 0.790 | 0.537 | 7.56 | <.0001 | <.0001 |


## 6. Full Layerwise Statistics


Green rows: Partner Identity above chance. Yellow rows: versions significantly different.


| Layer | Partner Identity (balanced_gpt) | Control (nonsense_codeword) | Between-Version |
| --- | --- | --- | --- |
| Acc | p (raw) | p (FDR) | Sig | Acc | p (raw) | p (FDR) | Sig | z | p (raw) | p (FDR) | Sig |
| 0 | 0.560 | 0.0093 | 0.0093 | * | 0.535 | 0.0885 | 0.4614 |  | 0.71 | 0.4775 | 0.4775 |  |
| 1 | 0.568 | 0.0040 | 0.0042 | * | 0.515 | 0.2912 | 0.4614 |  | 1.49 | 0.1362 | 0.1470 |  |
| 2 | 0.562 | 0.0071 | 0.0073 | * | 0.530 | 0.1251 | 0.4614 |  | 0.92 | 0.3559 | 0.3648 |  |
| 3 | 0.570 | 0.0029 | 0.0032 | * | 0.522 | 0.1977 | 0.4614 |  | 1.35 | 0.1772 | 0.1863 |  |
| 4 | 0.585 | 0.0004 | 0.0004 | * | 0.515 | 0.2912 | 0.4614 |  | 1.99 | 0.0466 | 0.0516 |  |
| 5 | 0.603 | <.0001 | <.0001 | * | 0.505 | 0.4404 | 0.5311 |  | 2.77 | 0.0055 | 0.0063 | * |
| 6 | 0.700 | <.0001 | <.0001 | * | 0.512 | 0.3264 | 0.4614 |  | 5.43 | <.0001 | <.0001 | * |
| 7 | 0.730 | <.0001 | <.0001 | * | 0.505 | 0.4404 | 0.5311 |  | 6.55 | <.0001 | <.0001 | * |
| 8 | 0.762 | <.0001 | <.0001 | * | 0.512 | 0.3264 | 0.4614 |  | 7.35 | <.0001 | <.0001 | * |
| 9 | 0.775 | <.0001 | <.0001 | * | 0.502 | 0.4801 | 0.5467 |  | 8.02 | <.0001 | <.0001 | * |
| 10 | 0.757 | <.0001 | <.0001 | * | 0.517 | 0.2579 | 0.4614 |  | 7.06 | <.0001 | <.0001 | * |
| 11 | 0.762 | <.0001 | <.0001 | * | 0.517 | 0.2579 | 0.4614 |  | 7.22 | <.0001 | <.0001 | * |
| 12 | 0.748 | <.0001 | <.0001 | * | 0.522 | 0.1977 | 0.4614 |  | 6.61 | <.0001 | <.0001 | * |
| 13 | 0.760 | <.0001 | <.0001 | * | 0.535 | 0.0885 | 0.4614 |  | 6.66 | <.0001 | <.0001 | * |
| 14 | 0.755 | <.0001 | <.0001 | * | 0.490 | 0.6736 | 0.6905 |  | 7.73 | <.0001 | <.0001 | * |
| 15 | 0.772 | <.0001 | <.0001 | * | 0.497 | 0.5596 | 0.6201 |  | 8.08 | <.0001 | <.0001 | * |
| 16 | 0.790 | <.0001 | <.0001 | * | 0.517 | 0.2579 | 0.4614 |  | 8.10 | <.0001 | <.0001 | * |
| 17 | 0.760 | <.0001 | <.0001 | * | 0.510 | 0.3632 | 0.4804 |  | 7.34 | <.0001 | <.0001 | * |
| 18 | 0.777 | <.0001 | <.0001 | * | 0.520 | 0.2266 | 0.4614 |  | 7.63 | <.0001 | <.0001 | * |
| 19 | 0.785 | <.0001 | <.0001 | * | 0.495 | 0.5987 | 0.6460 |  | 8.54 | <.0001 | <.0001 | * |
| 20 | 0.770 | <.0001 | <.0001 | * | 0.510 | 0.3632 | 0.4804 |  | 7.66 | <.0001 | <.0001 | * |
| 21 | 0.775 | <.0001 | <.0001 | * | 0.512 | 0.3264 | 0.4614 |  | 7.75 | <.0001 | <.0001 | * |
| 22 | 0.777 | <.0001 | <.0001 | * | 0.512 | 0.3264 | 0.4614 |  | 7.83 | <.0001 | <.0001 | * |
| 23 | 0.775 | <.0001 | <.0001 | * | 0.492 | 0.6368 | 0.6695 |  | 8.29 | <.0001 | <.0001 | * |
| 24 | 0.780 | <.0001 | <.0001 | * | 0.527 | 0.1469 | 0.4614 |  | 7.51 | <.0001 | <.0001 | * |
| 25 | 0.772 | <.0001 | <.0001 | * | 0.527 | 0.1469 | 0.4614 |  | 7.26 | <.0001 | <.0001 | * |
| 26 | 0.757 | <.0001 | <.0001 | * | 0.515 | 0.2912 | 0.4614 |  | 7.13 | <.0001 | <.0001 | * |
| 27 | 0.777 | <.0001 | <.0001 | * | 0.530 | 0.1251 | 0.4614 |  | 7.36 | <.0001 | <.0001 | * |
| 28 | 0.780 | <.0001 | <.0001 | * | 0.522 | 0.1977 | 0.4614 |  | 7.64 | <.0001 | <.0001 | * |
| 29 | 0.770 | <.0001 | <.0001 | * | 0.517 | 0.2579 | 0.4614 |  | 7.46 | <.0001 | <.0001 | * |
| 30 | 0.755 | <.0001 | <.0001 | * | 0.527 | 0.1469 | 0.4614 |  | 6.71 | <.0001 | <.0001 | * |
| 31 | 0.752 | <.0001 | <.0001 | * | 0.502 | 0.4801 | 0.5467 |  | 7.31 | <.0001 | <.0001 | * |
| 32 | 0.733 | <.0001 | <.0001 | * | 0.515 | 0.2912 | 0.4614 |  | 6.35 | <.0001 | <.0001 | * |
| 33 | 0.745 | <.0001 | <.0001 | * | 0.487 | 0.7088 | 0.7088 |  | 7.49 | <.0001 | <.0001 | * |
| 34 | 0.767 | <.0001 | <.0001 | * | 0.522 | 0.1977 | 0.4614 |  | 7.24 | <.0001 | <.0001 | * |
| 35 | 0.743 | <.0001 | <.0001 | * | 0.550 | 0.0255 | 0.4614 |  | 5.69 | <.0001 | <.0001 | * |
| 36 | 0.767 | <.0001 | <.0001 | * | 0.532 | 0.1056 | 0.4614 |  | 6.97 | <.0001 | <.0001 | * |
| 37 | 0.767 | <.0001 | <.0001 | * | 0.507 | 0.4013 | 0.5142 |  | 7.65 | <.0001 | <.0001 | * |
| 38 | 0.762 | <.0001 | <.0001 | * | 0.540 | 0.0605 | 0.4614 |  | 6.60 | <.0001 | <.0001 | * |
| 39 | 0.795 | <.0001 | <.0001 | * | 0.530 | 0.1251 | 0.4614 |  | 7.93 | <.0001 | <.0001 | * |
| 40 | 0.790 | <.0001 | <.0001 | * | 0.537 | 0.0735 | 0.4614 |  | 7.56 | <.0001 | <.0001 | * |
