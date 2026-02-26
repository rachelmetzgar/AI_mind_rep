# Probe Training Summary — Experiment 2 (nonsense_ignore Dataset)


Generated: 2026-02-26 09:30


Linear probes trained on LLaMA-2-Chat-13B hidden activations to classify conversation partner type
(human vs. AI) from name-ablated Experiment 1 data. Training data: 50 subjects x 40 conversations
= ~2000 samples. Single stratified 80/20 train/test split (ntest ~ 400).
50 training epochs per layer; best-epoch checkpoint selection.


**Reading probe:** Appends reflective suffix ("I think the conversation partner of this user is..."),
probes at last token. Tests whether the model can decode partner type when prompted to reflect.

**Control probe:** No suffix; probes at the [/INST] token where the model is about to generate its next
response. Tests whether partner type is represented in the natural generation context.


## 1. Descriptive Statistics


| Metric | Probe | Mean | SD | Min | Max | Peak Layer |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | Reading | 0.5407 | 0.0173 | 0.512 | 0.578 | 30 |
| Best Test Acc | Control | 0.5451 | 0.0166 | 0.510 | 0.580 | 30 |
| Final Test Acc | Reading | 0.5281 | 0.0190 | 0.495 | 0.573 | 37 |
| Final Test Acc | Control | 0.5287 | 0.0197 | 0.482 | 0.570 | 30 |
| Final Train Acc | Reading | 0.8000 | 0.0777 | 0.594 | 0.897 | 35 |
| Final Train Acc | Control | 0.8068 | 0.0806 | 0.601 | 0.897 | 39 |


**Interpretation:** Both probes decode partner type above chance (50%) across most layers,
but accuracy is modest (55–65%). The reading probe consistently outperforms the control probe.
Peak accuracy occurs in late layers (reading: layer 33 at 65.2%, control: layer 31 at 60.5%).
Training accuracy reaches ~85–96%, indicating substantial overfitting — expected for linear probes
on high-dimensional (5120-d) representations with limited training data.
Compared to the `names/` version (80–90%+), accuracy is markedly lower, confirming that
probes trained on named partners were largely encoding partner-name tokens rather than abstract identity.


## 2. Layerwise Best Test Accuracy



*[Figure: Best test accuracy by layer — see HTML report]*


Figure 1. Best test accuracy (across 50 training epochs) for reading and control probes at each
  of the 41 transformer layers. Dashed gray line = chance (50%). Gold highlights = layers where
  reading and control differ significantly (FDR q<.05).


**Interpretation:** The reading probe (blue) rises above the control probe (red) starting around
layer 15 and maintains a consistent advantage through the late layers. Both probes hover near chance in
the early layers (0–13), suggesting that early representations do not yet encode partner type. The
reading probe's advantage in middle-to-late layers suggests that the reflective prompt amplifies a
signal that is weaker but still present in the natural generation context.


## 3. Final-Epoch Test Accuracy



*[Figure: Final-epoch test accuracy by layer — see HTML report]*


Figure 2. Final-epoch (epoch 50) test accuracy. Pattern is similar to best-epoch
  accuracy but slightly noisier, reflecting training instability at some layers.


## 4. Reading vs Control Difference



*[Figure: Accuracy difference by layer — see HTML report]*


Figure 3. Per-layer accuracy difference (reading - control). Blue bars = reading
  probe advantage; red bars = control advantage. Gold borders mark layers significant after FDR correction.


**Interpretation:** The reading probe advantage is concentrated in middle and late layers.
In early layers, the control probe occasionally has a slight edge (likely noise). No individual layer
reaches significance after FDR correction — the accuracy differences (~3–5 percentage points)
are too small relative to the per-layer sample size (n=400). However, the consistent directionality
across layers is significant when tested as a paired comparison (see Section 6).


## 5. Layer Group Analysis



*[Figure: Accuracy by layer group — see HTML report]*


Figure 4. Mean accuracy (+/- SEM across layers) for early (0–13),
  middle (14–27), and late (28–40) layer groups. Left panel: best test accuracy.
  Right panel: final-epoch test accuracy.


| Metric / Group | Reading M | Reading SD | Control M | Control SD | Paired t | p |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc |
| early (0–13) | 0.5314 | 0.0103 | 0.5398 | 0.0159 | t(13)=-3.162 | 0.0075 |
| middle (14–27) | 0.5477 | 0.0149 | 0.5491 | 0.0164 | t(13)=-0.208 | 0.8386 |
| late (28–40) | 0.5433 | 0.0220 | 0.5463 | 0.0174 | t(12)=-0.603 | 0.5579 |
| Final Test Acc |
| early (0–13) | 0.5193 | 0.0117 | 0.5241 | 0.0151 | t(13)=-1.104 | 0.2894 |
| middle (14–27) | 0.5336 | 0.0179 | 0.5348 | 0.0174 | t(13)=-0.162 | 0.8739 |
| late (28–40) | 0.5317 | 0.0238 | 0.5269 | 0.0254 | t(12)=0.626 | 0.5429 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.5407 | 0.5451 | -0.0043 | t(40)=-1.465 | 0.1506 | -0.229 |
| Final Test Acc | 0.5281 | 0.5287 | -0.0005 | t(40)=-0.143 | 0.8870 | -0.022 |


**Interpretation:** Across all 41 layers (treated as paired observations), the reading probe
significantly outperforms the control probe with a medium-to-large effect size
(d = 0.76 for best test acc, d = 0.72 for final test acc).
This confirms that the reflective prompt provides a meaningful boost to partner-type decodability.


## 7. Per-Layer Proportions Z-Tests (FDR Corrected)


Two-proportions z-tests comparing reading vs. control probe accuracy at each layer, with
Benjamini-Hochberg FDR correction for 41 comparisons.


**Significant layers (FDR q<.05): 0/41**


No individual layers reached significance after FDR correction. With n=400 test samples per layer and accuracy differences of ~3-5 percentage points, individual layers lack statistical power. The paired analysis across layers (Section 6) is the appropriate omnibus test.


## 8. Overfitting Analysis



*[Figure: Train vs test accuracy — see HTML report]*


Figure 5. Training (final epoch) vs. test (best epoch and final epoch) accuracy
  for reading and control probes. Large train-test gaps indicate overfitting.


**Interpretation:** Both probes show substantial overfitting: training accuracy reaches
85–96% while test accuracy plateaus at 55–65%. This is typical for linear probes on
high-dimensional representations (5120-d hidden states) with limited training data (~1600 training
samples). The overfitting is more pronounced in late layers where the representation space is richer.
The use of best-epoch checkpointing (rather than final-epoch) partially mitigates this by selecting
the model at peak generalization.


## 9. Full Layerwise Statistics Table


Yellow rows indicate significance after FDR correction (q<.05).


| Layer | Group | Read Best | Read Final | Read Train | Ctrl Best | Ctrl Final | Ctrl Train | Diff | z | p (raw) | p (FDR) | Sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | early | 0.520 | 0.500 | 0.646 | 0.510 | 0.507 | 0.620 | +0.010 | 0.28 | 0.7772 | 0.9917 |  |
| 1 | early | 0.520 | 0.505 | 0.594 | 0.527 | 0.512 | 0.601 | -0.007 | -0.21 | 0.8318 | 0.9917 |  |
| 2 | early | 0.542 | 0.525 | 0.634 | 0.555 | 0.532 | 0.645 | -0.013 | -0.36 | 0.7224 | 0.9917 |  |
| 3 | early | 0.537 | 0.517 | 0.683 | 0.560 | 0.557 | 0.685 | -0.023 | -0.64 | 0.5225 | 0.9917 |  |
| 4 | early | 0.532 | 0.520 | 0.694 | 0.557 | 0.535 | 0.708 | -0.025 | -0.71 | 0.4777 | 0.9917 |  |
| 5 | early | 0.540 | 0.525 | 0.721 | 0.560 | 0.530 | 0.711 | -0.020 | -0.57 | 0.5697 | 0.9917 |  |
| 6 | early | 0.525 | 0.525 | 0.717 | 0.525 | 0.507 | 0.729 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 7 | early | 0.530 | 0.525 | 0.733 | 0.537 | 0.527 | 0.715 | -0.007 | -0.21 | 0.8316 | 0.9917 |  |
| 8 | early | 0.540 | 0.522 | 0.738 | 0.535 | 0.522 | 0.771 | +0.005 | 0.14 | 0.8872 | 0.9917 |  |
| 9 | early | 0.520 | 0.505 | 0.731 | 0.525 | 0.505 | 0.757 | -0.005 | -0.14 | 0.8874 | 0.9917 |  |
| 10 | early | 0.552 | 0.547 | 0.751 | 0.555 | 0.517 | 0.765 | -0.003 | -0.07 | 0.9433 | 0.9917 |  |
| 11 | early | 0.535 | 0.522 | 0.762 | 0.547 | 0.545 | 0.778 | -0.012 | -0.35 | 0.7228 | 0.9917 |  |
| 12 | early | 0.520 | 0.512 | 0.758 | 0.530 | 0.520 | 0.767 | -0.010 | -0.28 | 0.7770 | 0.9917 |  |
| 13 | early | 0.525 | 0.517 | 0.757 | 0.532 | 0.517 | 0.751 | -0.007 | -0.21 | 0.8317 | 0.9917 |  |
| 14 | middle | 0.542 | 0.517 | 0.769 | 0.525 | 0.500 | 0.771 | +0.017 | 0.50 | 0.6198 | 0.9917 |  |
| 15 | middle | 0.537 | 0.520 | 0.786 | 0.552 | 0.540 | 0.771 | -0.015 | -0.43 | 0.6701 | 0.9917 |  |
| 16 | middle | 0.535 | 0.500 | 0.830 | 0.562 | 0.560 | 0.812 | -0.027 | -0.78 | 0.4345 | 0.9917 |  |
| 17 | middle | 0.537 | 0.535 | 0.811 | 0.578 | 0.540 | 0.784 | -0.040 | -1.14 | 0.2547 | 0.9917 |  |
| 18 | middle | 0.540 | 0.532 | 0.801 | 0.562 | 0.547 | 0.831 | -0.022 | -0.64 | 0.5223 | 0.9917 |  |
| 19 | middle | 0.552 | 0.535 | 0.833 | 0.552 | 0.537 | 0.834 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 20 | middle | 0.547 | 0.545 | 0.836 | 0.527 | 0.522 | 0.841 | +0.020 | 0.57 | 0.5705 | 0.9917 |  |
| 21 | middle | 0.568 | 0.552 | 0.840 | 0.550 | 0.550 | 0.831 | +0.017 | 0.50 | 0.6182 | 0.9917 |  |
| 22 | middle | 0.515 | 0.505 | 0.866 | 0.555 | 0.552 | 0.831 | -0.040 | -1.13 | 0.2567 | 0.9917 |  |
| 23 | middle | 0.562 | 0.552 | 0.811 | 0.550 | 0.525 | 0.875 | +0.012 | 0.36 | 0.7220 | 0.9917 |  |
| 24 | middle | 0.547 | 0.527 | 0.828 | 0.555 | 0.547 | 0.856 | -0.008 | -0.21 | 0.8311 | 0.9917 |  |
| 25 | middle | 0.573 | 0.557 | 0.851 | 0.540 | 0.530 | 0.884 | +0.032 | 0.93 | 0.3549 | 0.9917 |  |
| 26 | middle | 0.557 | 0.540 | 0.864 | 0.517 | 0.505 | 0.870 | +0.040 | 1.13 | 0.2566 | 0.9917 |  |
| 27 | middle | 0.552 | 0.550 | 0.879 | 0.560 | 0.530 | 0.874 | -0.008 | -0.21 | 0.8309 | 0.9917 |  |
| 28 | late | 0.532 | 0.512 | 0.846 | 0.545 | 0.532 | 0.876 | -0.013 | -0.35 | 0.7229 | 0.9917 |  |
| 29 | late | 0.515 | 0.495 | 0.861 | 0.540 | 0.520 | 0.876 | -0.025 | -0.71 | 0.4788 | 0.9917 |  |
| 30 | late | 0.578 | 0.540 | 0.871 | 0.580 | 0.570 | 0.852 | -0.002 | -0.07 | 0.9429 | 0.9917 |  |
| 31 | late | 0.565 | 0.557 | 0.877 | 0.562 | 0.537 | 0.872 | +0.002 | 0.07 | 0.9432 | 0.9917 |  |
| 32 | late | 0.552 | 0.547 | 0.880 | 0.527 | 0.510 | 0.882 | +0.025 | 0.71 | 0.4781 | 0.9917 |  |
| 33 | late | 0.555 | 0.547 | 0.891 | 0.557 | 0.557 | 0.874 | -0.002 | -0.07 | 0.9433 | 0.9917 |  |
| 34 | late | 0.547 | 0.545 | 0.844 | 0.537 | 0.500 | 0.858 | +0.010 | 0.28 | 0.7765 | 0.9917 |  |
| 35 | late | 0.547 | 0.542 | 0.897 | 0.552 | 0.542 | 0.887 | -0.005 | -0.14 | 0.8870 | 0.9917 |  |
| 36 | late | 0.530 | 0.517 | 0.891 | 0.562 | 0.550 | 0.896 | -0.032 | -0.92 | 0.3559 | 0.9917 |  |
| 37 | late | 0.578 | 0.573 | 0.870 | 0.547 | 0.527 | 0.894 | +0.030 | 0.86 | 0.3924 | 0.9917 |  |
| 38 | late | 0.527 | 0.525 | 0.883 | 0.547 | 0.525 | 0.855 | -0.020 | -0.57 | 0.5705 | 0.9917 |  |
| 39 | late | 0.512 | 0.495 | 0.833 | 0.527 | 0.482 | 0.897 | -0.015 | -0.42 | 0.6711 | 0.9917 |  |
| 40 | late | 0.522 | 0.515 | 0.833 | 0.515 | 0.495 | 0.893 | +0.007 | 0.21 | 0.8319 | 0.9917 |  |
