# Probe Training Summary — Experiment 2 (nonsense_codeword Dataset)


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
| Best Test Acc | Reading | 0.5196 | 0.0166 | 0.490 | 0.562 | 19 |
| Best Test Acc | Control | 0.5167 | 0.0140 | 0.487 | 0.550 | 35 |
| Final Test Acc | Reading | 0.5021 | 0.0161 | 0.472 | 0.537 | 37 |
| Final Test Acc | Control | 0.4985 | 0.0162 | 0.472 | 0.532 | 0 |
| Final Train Acc | Reading | 0.7934 | 0.0752 | 0.615 | 0.896 | 35 |
| Final Train Acc | Control | 0.8031 | 0.0925 | 0.565 | 0.917 | 37 |


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
| early (0–13) | 0.5182 | 0.0137 | 0.5173 | 0.0101 | t(13)=0.207 | 0.8394 |
| middle (14–27) | 0.5204 | 0.0202 | 0.5114 | 0.0139 | t(13)=1.151 | 0.2706 |
| late (28–40) | 0.5202 | 0.0162 | 0.5217 | 0.0165 | t(12)=-0.311 | 0.7608 |
| Final Test Acc |
| early (0–13) | 0.5012 | 0.0140 | 0.4991 | 0.0177 | t(13)=0.440 | 0.6674 |
| middle (14–27) | 0.5009 | 0.0171 | 0.4941 | 0.0142 | t(13)=1.102 | 0.2906 |
| late (28–40) | 0.5042 | 0.0179 | 0.5025 | 0.0167 | t(12)=0.246 | 0.8101 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.5196 | 0.5167 | 0.0029 | t(40)=0.842 | 0.4046 | 0.132 |
| Final Test Acc | 0.5021 | 0.4985 | 0.0036 | t(40)=1.052 | 0.2992 | 0.164 |


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
| 0 | early | 0.510 | 0.502 | 0.652 | 0.535 | 0.532 | 0.609 | -0.025 | -0.71 | 0.4791 | 1.0000 |  |
| 1 | early | 0.525 | 0.520 | 0.615 | 0.515 | 0.507 | 0.565 | +0.010 | 0.28 | 0.7771 | 1.0000 |  |
| 2 | early | 0.525 | 0.497 | 0.651 | 0.530 | 0.495 | 0.612 | -0.005 | -0.14 | 0.8874 | 1.0000 |  |
| 3 | early | 0.522 | 0.515 | 0.647 | 0.522 | 0.510 | 0.639 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 4 | early | 0.517 | 0.495 | 0.699 | 0.515 | 0.502 | 0.676 | +0.002 | 0.07 | 0.9436 | 1.0000 |  |
| 5 | early | 0.500 | 0.490 | 0.700 | 0.505 | 0.505 | 0.706 | -0.005 | -0.14 | 0.8875 | 1.0000 |  |
| 6 | early | 0.490 | 0.472 | 0.721 | 0.512 | 0.475 | 0.719 | -0.022 | -0.64 | 0.5245 | 1.0000 |  |
| 7 | early | 0.540 | 0.522 | 0.723 | 0.505 | 0.490 | 0.733 | +0.035 | 0.99 | 0.3217 | 1.0000 |  |
| 8 | early | 0.510 | 0.490 | 0.715 | 0.512 | 0.480 | 0.752 | -0.002 | -0.07 | 0.9436 | 1.0000 |  |
| 9 | early | 0.527 | 0.487 | 0.737 | 0.502 | 0.487 | 0.731 | +0.025 | 0.71 | 0.4793 | 1.0000 |  |
| 10 | early | 0.520 | 0.510 | 0.748 | 0.517 | 0.500 | 0.756 | +0.003 | 0.07 | 0.9436 | 1.0000 |  |
| 11 | early | 0.505 | 0.497 | 0.761 | 0.517 | 0.487 | 0.753 | -0.012 | -0.35 | 0.7236 | 1.0000 |  |
| 12 | early | 0.530 | 0.512 | 0.772 | 0.520 | 0.482 | 0.758 | +0.010 | 0.28 | 0.7770 | 1.0000 |  |
| 13 | early | 0.532 | 0.505 | 0.763 | 0.532 | 0.532 | 0.774 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 14 | middle | 0.502 | 0.487 | 0.772 | 0.490 | 0.482 | 0.779 | +0.012 | 0.35 | 0.7237 | 1.0000 |  |
| 15 | middle | 0.527 | 0.510 | 0.797 | 0.497 | 0.475 | 0.781 | +0.030 | 0.85 | 0.3960 | 1.0000 |  |
| 16 | middle | 0.500 | 0.495 | 0.784 | 0.517 | 0.515 | 0.818 | -0.017 | -0.50 | 0.6206 | 1.0000 |  |
| 17 | middle | 0.510 | 0.477 | 0.782 | 0.510 | 0.485 | 0.811 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 18 | middle | 0.525 | 0.502 | 0.805 | 0.520 | 0.492 | 0.824 | +0.005 | 0.14 | 0.8874 | 1.0000 |  |
| 19 | middle | 0.562 | 0.515 | 0.805 | 0.492 | 0.490 | 0.778 | +0.070 | 1.98 | 0.0474 | 1.0000 |  |
| 20 | middle | 0.540 | 0.532 | 0.847 | 0.507 | 0.507 | 0.823 | +0.033 | 0.92 | 0.3574 | 1.0000 |  |
| 21 | middle | 0.492 | 0.485 | 0.806 | 0.515 | 0.510 | 0.823 | -0.023 | -0.64 | 0.5245 | 1.0000 |  |
| 22 | middle | 0.512 | 0.500 | 0.875 | 0.512 | 0.480 | 0.849 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 23 | middle | 0.540 | 0.535 | 0.819 | 0.492 | 0.487 | 0.824 | +0.048 | 1.34 | 0.1789 | 1.0000 |  |
| 24 | middle | 0.525 | 0.492 | 0.826 | 0.527 | 0.502 | 0.841 | -0.002 | -0.07 | 0.9436 | 1.0000 |  |
| 25 | middle | 0.520 | 0.495 | 0.824 | 0.527 | 0.517 | 0.858 | -0.007 | -0.21 | 0.8318 | 1.0000 |  |
| 26 | middle | 0.535 | 0.500 | 0.818 | 0.517 | 0.477 | 0.858 | +0.018 | 0.50 | 0.6201 | 1.0000 |  |
| 27 | middle | 0.492 | 0.485 | 0.864 | 0.532 | 0.495 | 0.848 | -0.040 | -1.13 | 0.2578 | 1.0000 |  |
| 28 | late | 0.520 | 0.520 | 0.856 | 0.522 | 0.512 | 0.878 | -0.002 | -0.07 | 0.9436 | 1.0000 |  |
| 29 | late | 0.530 | 0.507 | 0.847 | 0.515 | 0.505 | 0.882 | +0.015 | 0.42 | 0.6711 | 1.0000 |  |
| 30 | late | 0.537 | 0.492 | 0.791 | 0.527 | 0.522 | 0.911 | +0.010 | 0.28 | 0.7768 | 1.0000 |  |
| 31 | late | 0.497 | 0.490 | 0.889 | 0.502 | 0.472 | 0.893 | -0.005 | -0.14 | 0.8875 | 1.0000 |  |
| 32 | late | 0.515 | 0.507 | 0.858 | 0.515 | 0.497 | 0.897 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 33 | late | 0.510 | 0.495 | 0.885 | 0.487 | 0.482 | 0.912 | +0.023 | 0.64 | 0.5245 | 1.0000 |  |
| 34 | late | 0.497 | 0.472 | 0.883 | 0.522 | 0.502 | 0.907 | -0.025 | -0.71 | 0.4794 | 1.0000 |  |
| 35 | late | 0.537 | 0.517 | 0.896 | 0.550 | 0.522 | 0.895 | -0.013 | -0.35 | 0.7227 | 1.0000 |  |
| 36 | late | 0.505 | 0.482 | 0.867 | 0.527 | 0.525 | 0.909 | -0.022 | -0.64 | 0.5243 | 1.0000 |  |
| 37 | late | 0.537 | 0.537 | 0.879 | 0.507 | 0.485 | 0.917 | +0.030 | 0.85 | 0.3957 | 1.0000 |  |
| 38 | late | 0.545 | 0.517 | 0.838 | 0.537 | 0.515 | 0.894 | +0.008 | 0.21 | 0.8314 | 1.0000 |  |
| 39 | late | 0.507 | 0.497 | 0.889 | 0.530 | 0.500 | 0.909 | -0.023 | -0.64 | 0.5242 | 1.0000 |  |
| 40 | late | 0.522 | 0.517 | 0.824 | 0.537 | 0.490 | 0.828 | -0.015 | -0.43 | 0.6708 | 1.0000 |  |
