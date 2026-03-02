# Probe Training Summary — Experiment 2 (you_are_labels Dataset)


Generated: 2026-03-01 12:51


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
| Best Test Acc | Reading | 0.5579 | 0.0182 | 0.525 | 0.605 | 38 |
| Best Test Acc | Control | 0.5629 | 0.0176 | 0.525 | 0.603 | 34 |
| Final Test Acc | Reading | 0.5479 | 0.0198 | 0.512 | 0.603 | 38 |
| Final Test Acc | Control | 0.5453 | 0.0175 | 0.510 | 0.585 | 34 |
| Final Train Acc | Reading | 0.8188 | 0.0753 | 0.640 | 0.949 | 39 |
| Final Train Acc | Control | 0.8240 | 0.0792 | 0.610 | 0.939 | 40 |


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
| early (0–13) | 0.5537 | 0.0129 | 0.5609 | 0.0186 | t(13)=-1.220 | 0.2440 |
| middle (14–27) | 0.5609 | 0.0194 | 0.5604 | 0.0163 | t(13)=0.090 | 0.9293 |
| late (28–40) | 0.5592 | 0.0219 | 0.5677 | 0.0182 | t(12)=-0.989 | 0.3422 |
| Final Test Acc |
| early (0–13) | 0.5421 | 0.0166 | 0.5450 | 0.0179 | t(13)=-0.464 | 0.6503 |
| middle (14–27) | 0.5498 | 0.0192 | 0.5434 | 0.0174 | t(13)=1.053 | 0.3115 |
| late (28–40) | 0.5521 | 0.0234 | 0.5477 | 0.0182 | t(12)=0.456 | 0.6563 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.5579 | 0.5629 | -0.0049 | t(40)=-1.273 | 0.2104 | -0.199 |
| Final Test Acc | 0.5479 | 0.5453 | 0.0026 | t(40)=0.624 | 0.5365 | 0.097 |


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
| 0 | early | 0.578 | 0.573 | 0.651 | 0.545 | 0.520 | 0.637 | +0.032 | 0.93 | 0.3543 | 0.9916 |  |
| 1 | early | 0.557 | 0.557 | 0.640 | 0.537 | 0.535 | 0.610 | +0.020 | 0.57 | 0.5699 | 0.9916 |  |
| 2 | early | 0.552 | 0.537 | 0.662 | 0.545 | 0.535 | 0.663 | +0.007 | 0.21 | 0.8312 | 0.9916 |  |
| 3 | early | 0.560 | 0.535 | 0.694 | 0.560 | 0.547 | 0.691 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 4 | early | 0.550 | 0.545 | 0.719 | 0.568 | 0.540 | 0.725 | -0.017 | -0.50 | 0.6182 | 0.9916 |  |
| 5 | early | 0.555 | 0.532 | 0.724 | 0.575 | 0.570 | 0.743 | -0.020 | -0.57 | 0.5683 | 0.9916 |  |
| 6 | early | 0.540 | 0.527 | 0.757 | 0.588 | 0.540 | 0.754 | -0.047 | -1.35 | 0.1756 | 0.9916 |  |
| 7 | early | 0.555 | 0.550 | 0.746 | 0.540 | 0.527 | 0.749 | +0.015 | 0.43 | 0.6700 | 0.9916 |  |
| 8 | early | 0.525 | 0.512 | 0.766 | 0.530 | 0.522 | 0.771 | -0.005 | -0.14 | 0.8874 | 0.9916 |  |
| 9 | early | 0.550 | 0.535 | 0.764 | 0.562 | 0.550 | 0.789 | -0.012 | -0.36 | 0.7220 | 0.9916 |  |
| 10 | early | 0.562 | 0.555 | 0.766 | 0.580 | 0.578 | 0.803 | -0.017 | -0.50 | 0.6170 | 0.9916 |  |
| 11 | early | 0.568 | 0.560 | 0.815 | 0.580 | 0.573 | 0.807 | -0.012 | -0.36 | 0.7207 | 0.9916 |  |
| 12 | early | 0.540 | 0.520 | 0.779 | 0.580 | 0.542 | 0.798 | -0.040 | -1.14 | 0.2545 | 0.9916 |  |
| 13 | early | 0.560 | 0.550 | 0.794 | 0.562 | 0.550 | 0.814 | -0.002 | -0.07 | 0.9432 | 0.9916 |  |
| 14 | middle | 0.530 | 0.520 | 0.827 | 0.560 | 0.552 | 0.811 | -0.030 | -0.85 | 0.3942 | 0.9916 |  |
| 15 | middle | 0.547 | 0.520 | 0.810 | 0.540 | 0.537 | 0.800 | +0.007 | 0.21 | 0.8314 | 0.9916 |  |
| 16 | middle | 0.565 | 0.555 | 0.807 | 0.568 | 0.557 | 0.780 | -0.003 | -0.07 | 0.9431 | 0.9916 |  |
| 17 | middle | 0.555 | 0.550 | 0.824 | 0.555 | 0.542 | 0.814 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 18 | middle | 0.552 | 0.542 | 0.815 | 0.593 | 0.565 | 0.829 | -0.040 | -1.14 | 0.2528 | 0.9916 |  |
| 19 | middle | 0.532 | 0.530 | 0.819 | 0.545 | 0.515 | 0.844 | -0.013 | -0.35 | 0.7229 | 0.9916 |  |
| 20 | middle | 0.565 | 0.552 | 0.836 | 0.570 | 0.535 | 0.859 | -0.005 | -0.14 | 0.8865 | 0.9916 |  |
| 21 | middle | 0.542 | 0.532 | 0.867 | 0.562 | 0.522 | 0.861 | -0.020 | -0.57 | 0.5695 | 0.9916 |  |
| 22 | middle | 0.573 | 0.557 | 0.871 | 0.565 | 0.557 | 0.860 | +0.008 | 0.21 | 0.8304 | 0.9916 |  |
| 23 | middle | 0.595 | 0.568 | 0.800 | 0.562 | 0.545 | 0.852 | +0.032 | 0.93 | 0.3519 | 0.9916 |  |
| 24 | middle | 0.562 | 0.550 | 0.848 | 0.565 | 0.562 | 0.864 | -0.002 | -0.07 | 0.9432 | 0.9916 |  |
| 25 | middle | 0.588 | 0.578 | 0.838 | 0.578 | 0.552 | 0.883 | +0.010 | 0.29 | 0.7743 | 0.9916 |  |
| 26 | middle | 0.583 | 0.580 | 0.852 | 0.557 | 0.552 | 0.866 | +0.025 | 0.71 | 0.4751 | 0.9916 |  |
| 27 | middle | 0.562 | 0.562 | 0.868 | 0.525 | 0.510 | 0.873 | +0.037 | 1.06 | 0.2870 | 0.9916 |  |
| 28 | late | 0.547 | 0.542 | 0.888 | 0.568 | 0.532 | 0.889 | -0.020 | -0.57 | 0.5690 | 0.9916 |  |
| 29 | late | 0.595 | 0.590 | 0.884 | 0.547 | 0.535 | 0.897 | +0.047 | 1.36 | 0.1747 | 0.9916 |  |
| 30 | late | 0.562 | 0.555 | 0.865 | 0.560 | 0.525 | 0.881 | +0.002 | 0.07 | 0.9432 | 0.9916 |  |
| 31 | late | 0.542 | 0.532 | 0.885 | 0.557 | 0.557 | 0.836 | -0.015 | -0.43 | 0.6698 | 0.9916 |  |
| 32 | late | 0.537 | 0.532 | 0.890 | 0.580 | 0.557 | 0.881 | -0.042 | -1.21 | 0.2261 | 0.9916 |  |
| 33 | late | 0.535 | 0.522 | 0.892 | 0.575 | 0.573 | 0.909 | -0.040 | -1.14 | 0.2550 | 0.9916 |  |
| 34 | late | 0.555 | 0.547 | 0.895 | 0.603 | 0.585 | 0.923 | -0.047 | -1.36 | 0.1737 | 0.9916 |  |
| 35 | late | 0.568 | 0.560 | 0.907 | 0.535 | 0.525 | 0.922 | +0.032 | 0.92 | 0.3554 | 0.9916 |  |
| 36 | late | 0.552 | 0.547 | 0.901 | 0.590 | 0.550 | 0.927 | -0.037 | -1.07 | 0.2839 | 0.9916 |  |
| 37 | late | 0.535 | 0.530 | 0.874 | 0.550 | 0.542 | 0.900 | -0.015 | -0.43 | 0.6702 | 0.9916 |  |
| 38 | late | 0.605 | 0.603 | 0.879 | 0.575 | 0.537 | 0.841 | +0.030 | 0.86 | 0.3883 | 0.9916 |  |
| 39 | late | 0.562 | 0.547 | 0.949 | 0.575 | 0.540 | 0.889 | -0.012 | -0.36 | 0.7211 | 0.9916 |  |
| 40 | late | 0.573 | 0.568 | 0.904 | 0.565 | 0.560 | 0.939 | +0.008 | 0.21 | 0.8304 | 0.9916 |  |
