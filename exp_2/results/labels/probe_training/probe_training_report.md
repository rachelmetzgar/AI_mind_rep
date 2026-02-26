# Probe Training Summary — Experiment 2 (labels Dataset)


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
| Best Test Acc | Reading | 0.5803 | 0.0405 | 0.512 | 0.652 | 33 |
| Best Test Acc | Control | 0.5522 | 0.0201 | 0.512 | 0.605 | 31 |
| Final Test Acc | Reading | 0.5724 | 0.0424 | 0.490 | 0.647 | 33 |
| Final Test Acc | Control | 0.5420 | 0.0206 | 0.505 | 0.593 | 31 |
| Final Train Acc | Reading | 0.8476 | 0.1061 | 0.629 | 0.964 | 36 |
| Final Train Acc | Control | 0.8247 | 0.0918 | 0.607 | 0.942 | 36 |


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
| early (0–13) | 0.5323 | 0.0117 | 0.5404 | 0.0188 | t(13)=-1.251 | 0.2331 |
| middle (14–27) | 0.5961 | 0.0261 | 0.5571 | 0.0135 | t(13)=5.972 | <.0001 |
| late (28–40) | 0.6150 | 0.0173 | 0.5596 | 0.0228 | t(12)=7.075 | <.0001 |
| Final Test Acc |
| early (0–13) | 0.5220 | 0.0154 | 0.5348 | 0.0203 | t(13)=-1.707 | 0.1115 |
| middle (14–27) | 0.5893 | 0.0249 | 0.5471 | 0.0156 | t(13)=6.517 | <.0001 |
| late (28–40) | 0.6087 | 0.0186 | 0.5442 | 0.0247 | t(12)=8.071 | <.0001 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.5803 | 0.5522 | 0.0281 | t(40)=4.880 | <.0001 | 0.762 |
| Final Test Acc | 0.5724 | 0.5420 | 0.0304 | t(40)=4.623 | <.0001 | 0.722 |


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
| 0 | early | 0.547 | 0.535 | 0.652 | 0.515 | 0.510 | 0.642 | +0.032 | 0.92 | 0.3570 | 0.6099 |  |
| 1 | early | 0.535 | 0.535 | 0.634 | 0.540 | 0.537 | 0.607 | -0.005 | -0.14 | 0.8872 | 0.9833 |  |
| 2 | early | 0.537 | 0.532 | 0.629 | 0.515 | 0.505 | 0.642 | +0.022 | 0.64 | 0.5239 | 0.6931 |  |
| 3 | early | 0.527 | 0.527 | 0.657 | 0.532 | 0.522 | 0.661 | -0.005 | -0.14 | 0.8873 | 0.9833 |  |
| 4 | early | 0.532 | 0.527 | 0.677 | 0.535 | 0.527 | 0.674 | -0.003 | -0.07 | 0.9435 | 0.9919 |  |
| 5 | early | 0.542 | 0.522 | 0.716 | 0.550 | 0.550 | 0.715 | -0.008 | -0.21 | 0.8313 | 0.9741 |  |
| 6 | early | 0.530 | 0.522 | 0.705 | 0.570 | 0.565 | 0.733 | -0.040 | -1.14 | 0.2555 | 0.4989 |  |
| 7 | early | 0.515 | 0.512 | 0.718 | 0.562 | 0.562 | 0.741 | -0.047 | -1.35 | 0.1778 | 0.4860 |  |
| 8 | early | 0.540 | 0.530 | 0.762 | 0.570 | 0.565 | 0.782 | -0.030 | -0.85 | 0.3933 | 0.6202 |  |
| 9 | early | 0.512 | 0.490 | 0.756 | 0.537 | 0.527 | 0.752 | -0.025 | -0.71 | 0.4790 | 0.6771 |  |
| 10 | early | 0.540 | 0.540 | 0.796 | 0.512 | 0.507 | 0.790 | +0.028 | 0.78 | 0.4360 | 0.6385 |  |
| 11 | early | 0.512 | 0.500 | 0.781 | 0.535 | 0.535 | 0.773 | -0.023 | -0.64 | 0.5240 | 0.6931 |  |
| 12 | early | 0.535 | 0.500 | 0.757 | 0.545 | 0.535 | 0.780 | -0.010 | -0.28 | 0.7766 | 0.9649 |  |
| 13 | early | 0.545 | 0.532 | 0.771 | 0.545 | 0.537 | 0.780 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 14 | middle | 0.532 | 0.530 | 0.815 | 0.540 | 0.525 | 0.797 | -0.008 | -0.21 | 0.8316 | 0.9741 |  |
| 15 | middle | 0.565 | 0.562 | 0.875 | 0.565 | 0.550 | 0.811 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 16 | middle | 0.590 | 0.583 | 0.880 | 0.547 | 0.537 | 0.823 | +0.042 | 1.21 | 0.2249 | 0.4989 |  |
| 17 | middle | 0.570 | 0.562 | 0.854 | 0.557 | 0.557 | 0.851 | +0.012 | 0.36 | 0.7215 | 0.9244 |  |
| 18 | middle | 0.603 | 0.593 | 0.895 | 0.562 | 0.540 | 0.811 | +0.040 | 1.15 | 0.2513 | 0.4989 |  |
| 19 | middle | 0.605 | 0.603 | 0.893 | 0.573 | 0.568 | 0.836 | +0.032 | 0.93 | 0.3503 | 0.6099 |  |
| 20 | middle | 0.598 | 0.588 | 0.864 | 0.550 | 0.537 | 0.838 | +0.047 | 1.36 | 0.1743 | 0.4860 |  |
| 21 | middle | 0.593 | 0.583 | 0.912 | 0.552 | 0.550 | 0.877 | +0.040 | 1.14 | 0.2528 | 0.4989 |  |
| 22 | middle | 0.625 | 0.615 | 0.911 | 0.570 | 0.570 | 0.863 | +0.055 | 1.59 | 0.1127 | 0.4622 |  |
| 23 | middle | 0.620 | 0.620 | 0.911 | 0.547 | 0.537 | 0.875 | +0.073 | 2.08 | 0.0375 | 0.3226 |  |
| 24 | middle | 0.627 | 0.615 | 0.911 | 0.562 | 0.537 | 0.877 | +0.065 | 1.87 | 0.0611 | 0.3226 |  |
| 25 | middle | 0.608 | 0.600 | 0.904 | 0.535 | 0.532 | 0.911 | +0.073 | 2.07 | 0.0383 | 0.3226 |  |
| 26 | middle | 0.593 | 0.588 | 0.931 | 0.552 | 0.540 | 0.886 | +0.040 | 1.14 | 0.2528 | 0.4989 |  |
| 27 | middle | 0.618 | 0.610 | 0.910 | 0.585 | 0.578 | 0.929 | +0.033 | 0.94 | 0.3479 | 0.6099 |  |
| 28 | late | 0.618 | 0.595 | 0.925 | 0.590 | 0.562 | 0.935 | +0.028 | 0.80 | 0.4265 | 0.6385 |  |
| 29 | late | 0.608 | 0.590 | 0.939 | 0.565 | 0.545 | 0.908 | +0.043 | 1.22 | 0.2223 | 0.4989 |  |
| 30 | late | 0.620 | 0.618 | 0.953 | 0.590 | 0.588 | 0.940 | +0.030 | 0.87 | 0.3855 | 0.6202 |  |
| 31 | late | 0.608 | 0.608 | 0.932 | 0.605 | 0.593 | 0.839 | +0.003 | 0.07 | 0.9423 | 0.9919 |  |
| 32 | late | 0.610 | 0.610 | 0.912 | 0.557 | 0.547 | 0.919 | +0.052 | 1.51 | 0.1320 | 0.4860 |  |
| 33 | late | 0.652 | 0.647 | 0.958 | 0.542 | 0.532 | 0.906 | +0.110 | 3.17 | 0.0015 | 0.0620 |  |
| 34 | late | 0.640 | 0.637 | 0.961 | 0.542 | 0.535 | 0.912 | +0.098 | 2.80 | 0.0050 | 0.1032 |  |
| 35 | late | 0.615 | 0.608 | 0.956 | 0.547 | 0.535 | 0.925 | +0.068 | 1.93 | 0.0530 | 0.3226 |  |
| 36 | late | 0.620 | 0.615 | 0.964 | 0.555 | 0.530 | 0.942 | +0.065 | 1.87 | 0.0619 | 0.3226 |  |
| 37 | late | 0.618 | 0.613 | 0.953 | 0.568 | 0.552 | 0.888 | +0.050 | 1.44 | 0.1501 | 0.4860 |  |
| 38 | late | 0.585 | 0.580 | 0.946 | 0.535 | 0.517 | 0.911 | +0.050 | 1.42 | 0.1543 | 0.4860 |  |
| 39 | late | 0.608 | 0.600 | 0.953 | 0.542 | 0.527 | 0.902 | +0.065 | 1.86 | 0.0630 | 0.3226 |  |
| 40 | late | 0.595 | 0.593 | 0.926 | 0.535 | 0.510 | 0.829 | +0.060 | 1.71 | 0.0870 | 0.3962 |  |
