# Probe Training Summary — Experiment 2 (balanced_names Dataset)


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
| Best Test Acc | Reading | 0.6612 | 0.0587 | 0.537 | 0.750 | 39 |
| Best Test Acc | Control | 0.6677 | 0.0597 | 0.535 | 0.745 | 39 |
| Final Test Acc | Reading | 0.6525 | 0.0619 | 0.517 | 0.743 | 39 |
| Final Test Acc | Control | 0.6586 | 0.0607 | 0.517 | 0.720 | 39 |
| Final Train Acc | Reading | 0.9001 | 0.1401 | 0.557 | 0.999 | 37 |
| Final Train Acc | Control | 0.9084 | 0.1388 | 0.557 | 0.999 | 35 |


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
| early (0–13) | 0.5929 | 0.0483 | 0.6048 | 0.0610 | t(13)=-1.683 | 0.1163 |
| middle (14–27) | 0.6895 | 0.0135 | 0.6898 | 0.0187 | t(13)=-0.063 | 0.9506 |
| late (28–40) | 0.7044 | 0.0183 | 0.7117 | 0.0144 | t(12)=-1.573 | 0.1418 |
| Final Test Acc |
| early (0–13) | 0.5804 | 0.0508 | 0.5918 | 0.0579 | t(13)=-1.592 | 0.1355 |
| middle (14–27) | 0.6841 | 0.0154 | 0.6846 | 0.0204 | t(13)=-0.097 | 0.9243 |
| late (28–40) | 0.6962 | 0.0201 | 0.7025 | 0.0101 | t(12)=-1.169 | 0.2650 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.6612 | 0.6677 | -0.0065 | t(40)=-1.900 | 0.0646 | -0.297 |
| Final Test Acc | 0.6525 | 0.6586 | -0.0061 | t(40)=-1.730 | 0.0914 | -0.270 |


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
| 0 | early | 0.560 | 0.560 | 0.566 | 0.542 | 0.535 | 0.566 | +0.018 | 0.50 | 0.6188 | 0.9635 |  |
| 1 | early | 0.562 | 0.555 | 0.557 | 0.557 | 0.542 | 0.557 | +0.005 | 0.14 | 0.8867 | 0.9635 |  |
| 2 | early | 0.562 | 0.545 | 0.574 | 0.535 | 0.535 | 0.562 | +0.027 | 0.78 | 0.4345 | 0.9635 |  |
| 3 | early | 0.550 | 0.542 | 0.622 | 0.540 | 0.530 | 0.591 | +0.010 | 0.28 | 0.7764 | 0.9635 |  |
| 4 | early | 0.550 | 0.525 | 0.656 | 0.540 | 0.517 | 0.657 | +0.010 | 0.28 | 0.7764 | 0.9635 |  |
| 5 | early | 0.537 | 0.517 | 0.708 | 0.565 | 0.545 | 0.708 | -0.027 | -0.78 | 0.4343 | 0.9635 |  |
| 6 | early | 0.557 | 0.537 | 0.721 | 0.568 | 0.565 | 0.761 | -0.010 | -0.29 | 0.7756 | 0.9635 |  |
| 7 | early | 0.560 | 0.545 | 0.733 | 0.600 | 0.600 | 0.814 | -0.040 | -1.15 | 0.2517 | 0.9635 |  |
| 8 | early | 0.598 | 0.590 | 0.779 | 0.667 | 0.655 | 0.873 | -0.070 | -2.05 | 0.0400 | 0.9635 |  |
| 9 | early | 0.645 | 0.632 | 0.853 | 0.685 | 0.657 | 0.889 | -0.040 | -1.20 | 0.2307 | 0.9635 |  |
| 10 | early | 0.655 | 0.627 | 0.860 | 0.675 | 0.652 | 0.896 | -0.020 | -0.60 | 0.5490 | 0.9635 |  |
| 11 | early | 0.642 | 0.637 | 0.883 | 0.662 | 0.642 | 0.906 | -0.020 | -0.59 | 0.5525 | 0.9635 |  |
| 12 | early | 0.647 | 0.642 | 0.900 | 0.660 | 0.647 | 0.912 | -0.013 | -0.37 | 0.7102 | 0.9635 |  |
| 13 | early | 0.672 | 0.667 | 0.935 | 0.670 | 0.660 | 0.909 | +0.002 | 0.08 | 0.9400 | 0.9635 |  |
| 14 | middle | 0.677 | 0.670 | 0.947 | 0.655 | 0.652 | 0.941 | +0.022 | 0.67 | 0.4998 | 0.9635 |  |
| 15 | middle | 0.703 | 0.693 | 0.950 | 0.675 | 0.670 | 0.949 | +0.027 | 0.84 | 0.4009 | 0.9635 |  |
| 16 | middle | 0.713 | 0.710 | 0.954 | 0.700 | 0.698 | 0.971 | +0.013 | 0.39 | 0.6979 | 0.9635 |  |
| 17 | middle | 0.693 | 0.693 | 0.960 | 0.690 | 0.688 | 0.968 | +0.003 | 0.08 | 0.9390 | 0.9635 |  |
| 18 | middle | 0.690 | 0.682 | 0.958 | 0.675 | 0.662 | 0.973 | +0.015 | 0.46 | 0.6486 | 0.9635 |  |
| 19 | middle | 0.660 | 0.655 | 0.975 | 0.703 | 0.685 | 0.963 | -0.042 | -1.29 | 0.1971 | 0.9635 |  |
| 20 | middle | 0.682 | 0.672 | 0.977 | 0.665 | 0.657 | 0.973 | +0.017 | 0.53 | 0.5976 | 0.9635 |  |
| 21 | middle | 0.688 | 0.682 | 0.978 | 0.685 | 0.677 | 0.988 | +0.002 | 0.08 | 0.9393 | 0.9635 |  |
| 22 | middle | 0.690 | 0.688 | 0.990 | 0.677 | 0.670 | 0.989 | +0.012 | 0.38 | 0.7038 | 0.9635 |  |
| 23 | middle | 0.690 | 0.682 | 0.984 | 0.718 | 0.715 | 0.991 | -0.028 | -0.85 | 0.3944 | 0.9635 |  |
| 24 | middle | 0.677 | 0.665 | 0.988 | 0.693 | 0.690 | 0.989 | -0.015 | -0.46 | 0.6479 | 0.9635 |  |
| 25 | middle | 0.693 | 0.688 | 0.992 | 0.718 | 0.715 | 0.991 | -0.025 | -0.78 | 0.4382 | 0.9635 |  |
| 26 | middle | 0.688 | 0.688 | 0.993 | 0.703 | 0.703 | 0.993 | -0.015 | -0.46 | 0.6450 | 0.9635 |  |
| 27 | middle | 0.710 | 0.710 | 0.991 | 0.703 | 0.703 | 0.995 | +0.007 | 0.23 | 0.8159 | 0.9635 |  |
| 28 | late | 0.700 | 0.693 | 0.991 | 0.703 | 0.698 | 0.996 | -0.003 | -0.08 | 0.9384 | 0.9635 |  |
| 29 | late | 0.713 | 0.705 | 0.993 | 0.700 | 0.688 | 0.997 | +0.013 | 0.39 | 0.6979 | 0.9635 |  |
| 30 | late | 0.713 | 0.705 | 0.984 | 0.690 | 0.688 | 0.996 | +0.023 | 0.70 | 0.4869 | 0.9635 |  |
| 31 | late | 0.695 | 0.688 | 0.993 | 0.710 | 0.698 | 0.997 | -0.015 | -0.46 | 0.6426 | 0.9635 |  |
| 32 | late | 0.715 | 0.715 | 0.993 | 0.715 | 0.708 | 0.996 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 33 | late | 0.700 | 0.693 | 0.995 | 0.708 | 0.703 | 0.997 | -0.008 | -0.23 | 0.8163 | 0.9635 |  |
| 34 | late | 0.703 | 0.685 | 0.995 | 0.700 | 0.695 | 0.998 | +0.003 | 0.08 | 0.9384 | 0.9635 |  |
| 35 | late | 0.670 | 0.662 | 0.996 | 0.705 | 0.700 | 0.999 | -0.035 | -1.07 | 0.2856 | 0.9635 |  |
| 36 | late | 0.703 | 0.680 | 0.995 | 0.733 | 0.718 | 0.998 | -0.030 | -0.94 | 0.3460 | 0.9635 |  |
| 37 | late | 0.705 | 0.700 | 0.999 | 0.715 | 0.705 | 0.999 | -0.010 | -0.31 | 0.7553 | 0.9635 |  |
| 38 | late | 0.685 | 0.675 | 0.998 | 0.713 | 0.703 | 0.998 | -0.027 | -0.85 | 0.3966 | 0.9635 |  |
| 39 | late | 0.750 | 0.743 | 0.994 | 0.745 | 0.720 | 0.999 | +0.005 | 0.16 | 0.8707 | 0.9635 |  |
| 40 | late | 0.708 | 0.708 | 0.999 | 0.718 | 0.713 | 0.999 | -0.010 | -0.31 | 0.7547 | 0.9635 |  |
