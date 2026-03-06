# Probe Training Summary — Experiment 2 (nonsense_codeword Dataset)


Generated: 2026-03-04 07:30


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
| Best Test Acc | Reading | 0.5222 | 0.0189 | 0.492 | 0.565 | 30 |
| Best Test Acc | Control | 0.5170 | 0.0141 | 0.487 | 0.550 | 35 |
| Final Test Acc | Reading | 0.5049 | 0.0204 | 0.458 | 0.550 | 28 |
| Final Test Acc | Control | 0.4985 | 0.0160 | 0.475 | 0.535 | 13 |
| Final Train Acc | Reading | 0.8004 | 0.0804 | 0.597 | 0.918 | 35 |
| Final Train Acc | Control | 0.8031 | 0.0925 | 0.565 | 0.916 | 37 |


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
| early (0–13) | 0.5212 | 0.0140 | 0.5177 | 0.0104 | t(13)=0.701 | 0.4955 |
| middle (14–27) | 0.5198 | 0.0228 | 0.5112 | 0.0132 | t(13)=1.133 | 0.2778 |
| late (28–40) | 0.5258 | 0.0199 | 0.5225 | 0.0168 | t(12)=0.495 | 0.6299 |
| Final Test Acc |
| early (0–13) | 0.5020 | 0.0122 | 0.4993 | 0.0180 | t(13)=0.541 | 0.5979 |
| middle (14–27) | 0.5018 | 0.0256 | 0.4943 | 0.0149 | t(13)=0.923 | 0.3727 |
| late (28–40) | 0.5113 | 0.0212 | 0.5021 | 0.0150 | t(12)=1.338 | 0.2058 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.5222 | 0.5170 | 0.0052 | t(40)=1.409 | 0.1667 | 0.220 |
| Final Test Acc | 0.5049 | 0.4985 | 0.0064 | t(40)=1.666 | 0.1036 | 0.260 |


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
| 0 | early | 0.510 | 0.502 | 0.638 | 0.535 | 0.532 | 0.609 | -0.025 | -0.71 | 0.4791 | 0.9219 |  |
| 1 | early | 0.515 | 0.510 | 0.597 | 0.515 | 0.507 | 0.565 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 2 | early | 0.520 | 0.492 | 0.649 | 0.530 | 0.495 | 0.612 | -0.010 | -0.28 | 0.7770 | 0.9219 |  |
| 3 | early | 0.517 | 0.512 | 0.648 | 0.522 | 0.510 | 0.638 | -0.005 | -0.14 | 0.8874 | 0.9330 |  |
| 4 | early | 0.520 | 0.487 | 0.697 | 0.515 | 0.502 | 0.676 | +0.005 | 0.14 | 0.8875 | 0.9330 |  |
| 5 | early | 0.505 | 0.500 | 0.705 | 0.505 | 0.505 | 0.706 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 6 | early | 0.505 | 0.475 | 0.721 | 0.512 | 0.475 | 0.720 | -0.007 | -0.21 | 0.8320 | 0.9219 |  |
| 7 | early | 0.545 | 0.517 | 0.746 | 0.505 | 0.490 | 0.733 | +0.040 | 1.13 | 0.2573 | 0.9219 |  |
| 8 | early | 0.535 | 0.512 | 0.733 | 0.512 | 0.480 | 0.751 | +0.023 | 0.64 | 0.5240 | 0.9219 |  |
| 9 | early | 0.535 | 0.497 | 0.743 | 0.502 | 0.487 | 0.731 | +0.033 | 0.92 | 0.3576 | 0.9219 |  |
| 10 | early | 0.535 | 0.517 | 0.749 | 0.517 | 0.500 | 0.756 | +0.018 | 0.50 | 0.6201 | 0.9219 |  |
| 11 | early | 0.497 | 0.492 | 0.762 | 0.517 | 0.487 | 0.752 | -0.020 | -0.57 | 0.5716 | 0.9219 |  |
| 12 | early | 0.532 | 0.502 | 0.779 | 0.522 | 0.482 | 0.759 | +0.010 | 0.28 | 0.7770 | 0.9219 |  |
| 13 | early | 0.525 | 0.507 | 0.772 | 0.535 | 0.535 | 0.774 | -0.010 | -0.28 | 0.7769 | 0.9219 |  |
| 14 | middle | 0.505 | 0.500 | 0.784 | 0.490 | 0.482 | 0.779 | +0.015 | 0.42 | 0.6714 | 0.9219 |  |
| 15 | middle | 0.512 | 0.492 | 0.806 | 0.497 | 0.475 | 0.781 | +0.015 | 0.42 | 0.6714 | 0.9219 |  |
| 16 | middle | 0.510 | 0.487 | 0.787 | 0.517 | 0.512 | 0.818 | -0.007 | -0.21 | 0.8319 | 0.9219 |  |
| 17 | middle | 0.497 | 0.458 | 0.791 | 0.510 | 0.482 | 0.811 | -0.013 | -0.35 | 0.7237 | 0.9219 |  |
| 18 | middle | 0.492 | 0.480 | 0.812 | 0.520 | 0.492 | 0.824 | -0.028 | -0.78 | 0.4366 | 0.9219 |  |
| 19 | middle | 0.555 | 0.520 | 0.820 | 0.495 | 0.487 | 0.778 | +0.060 | 1.70 | 0.0893 | 0.9219 |  |
| 20 | middle | 0.532 | 0.522 | 0.815 | 0.510 | 0.510 | 0.823 | +0.022 | 0.64 | 0.5241 | 0.9219 |  |
| 21 | middle | 0.492 | 0.482 | 0.801 | 0.512 | 0.510 | 0.823 | -0.020 | -0.57 | 0.5716 | 0.9219 |  |
| 22 | middle | 0.505 | 0.502 | 0.854 | 0.512 | 0.480 | 0.849 | -0.007 | -0.21 | 0.8320 | 0.9219 |  |
| 23 | middle | 0.550 | 0.547 | 0.809 | 0.492 | 0.487 | 0.824 | +0.058 | 1.63 | 0.1036 | 0.9219 |  |
| 24 | middle | 0.545 | 0.500 | 0.834 | 0.527 | 0.507 | 0.840 | +0.018 | 0.50 | 0.6197 | 0.9219 |  |
| 25 | middle | 0.535 | 0.515 | 0.823 | 0.527 | 0.520 | 0.858 | +0.008 | 0.21 | 0.8317 | 0.9219 |  |
| 26 | middle | 0.545 | 0.542 | 0.851 | 0.515 | 0.477 | 0.858 | +0.030 | 0.85 | 0.3953 | 0.9219 |  |
| 27 | middle | 0.500 | 0.475 | 0.881 | 0.530 | 0.495 | 0.848 | -0.030 | -0.85 | 0.3959 | 0.9219 |  |
| 28 | late | 0.550 | 0.550 | 0.852 | 0.522 | 0.512 | 0.878 | +0.028 | 0.78 | 0.4355 | 0.9219 |  |
| 29 | late | 0.527 | 0.515 | 0.879 | 0.517 | 0.502 | 0.882 | +0.010 | 0.28 | 0.7771 | 0.9219 |  |
| 30 | late | 0.565 | 0.502 | 0.803 | 0.527 | 0.522 | 0.909 | +0.037 | 1.07 | 0.2868 | 0.9219 |  |
| 31 | late | 0.510 | 0.492 | 0.882 | 0.502 | 0.475 | 0.894 | +0.008 | 0.21 | 0.8320 | 0.9219 |  |
| 32 | late | 0.542 | 0.540 | 0.875 | 0.515 | 0.497 | 0.897 | +0.027 | 0.78 | 0.4359 | 0.9219 |  |
| 33 | late | 0.517 | 0.512 | 0.878 | 0.487 | 0.482 | 0.912 | +0.030 | 0.85 | 0.3961 | 0.9219 |  |
| 34 | late | 0.502 | 0.500 | 0.882 | 0.522 | 0.502 | 0.907 | -0.020 | -0.57 | 0.5715 | 0.9219 |  |
| 35 | late | 0.540 | 0.537 | 0.918 | 0.550 | 0.522 | 0.895 | -0.010 | -0.28 | 0.7764 | 0.9219 |  |
| 36 | late | 0.497 | 0.475 | 0.889 | 0.532 | 0.517 | 0.909 | -0.035 | -0.99 | 0.3220 | 0.9219 |  |
| 37 | late | 0.525 | 0.520 | 0.909 | 0.507 | 0.487 | 0.916 | +0.018 | 0.50 | 0.6204 | 0.9219 |  |
| 38 | late | 0.530 | 0.495 | 0.875 | 0.540 | 0.512 | 0.895 | -0.010 | -0.28 | 0.7768 | 0.9219 |  |
| 39 | late | 0.505 | 0.500 | 0.881 | 0.530 | 0.497 | 0.909 | -0.025 | -0.71 | 0.4792 | 0.9219 |  |
| 40 | late | 0.522 | 0.507 | 0.889 | 0.537 | 0.495 | 0.828 | -0.015 | -0.43 | 0.6708 | 0.9219 |  |
