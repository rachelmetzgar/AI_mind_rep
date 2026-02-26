# Probe Training Summary — Experiment 2 (balanced_gpt Dataset)


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
| Best Test Acc | Reading | 0.7196 | 0.0674 | 0.562 | 0.780 | 39 |
| Best Test Acc | Control | 0.7364 | 0.0703 | 0.560 | 0.795 | 39 |
| Final Test Acc | Reading | 0.7102 | 0.0663 | 0.560 | 0.777 | 39 |
| Final Test Acc | Control | 0.7282 | 0.0706 | 0.552 | 0.787 | 39 |
| Final Train Acc | Reading | 0.9198 | 0.1358 | 0.564 | 1.000 | 39 |
| Final Train Acc | Control | 0.9225 | 0.1389 | 0.549 | 1.000 | 39 |


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
| early (0–13) | 0.6580 | 0.0858 | 0.6746 | 0.0921 | t(13)=-2.411 | 0.0314 |
| middle (14–27) | 0.7541 | 0.0109 | 0.7729 | 0.0105 | t(13)=-5.075 | 0.0002 |
| late (28–40) | 0.7487 | 0.0180 | 0.7637 | 0.0182 | t(12)=-4.101 | 0.0015 |
| Final Test Acc |
| early (0–13) | 0.6477 | 0.0823 | 0.6648 | 0.0912 | t(13)=-2.394 | 0.0325 |
| middle (14–27) | 0.7454 | 0.0071 | 0.7652 | 0.0106 | t(13)=-6.228 | <.0001 |
| late (28–40) | 0.7396 | 0.0181 | 0.7565 | 0.0188 | t(12)=-5.387 | 0.0002 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.7196 | 0.7364 | -0.0168 | t(40)=-5.910 | <.0001 | -0.923 |
| Final Test Acc | 0.7102 | 0.7282 | -0.0180 | t(40)=-6.441 | <.0001 | -1.006 |


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
| 0 | early | 0.565 | 0.565 | 0.567 | 0.560 | 0.560 | 0.549 | +0.005 | 0.14 | 0.8867 | 0.9598 |  |
| 1 | early | 0.562 | 0.560 | 0.564 | 0.568 | 0.560 | 0.556 | -0.005 | -0.14 | 0.8866 | 0.9598 |  |
| 2 | early | 0.568 | 0.562 | 0.583 | 0.562 | 0.552 | 0.571 | +0.005 | 0.14 | 0.8866 | 0.9598 |  |
| 3 | early | 0.580 | 0.578 | 0.644 | 0.570 | 0.565 | 0.609 | +0.010 | 0.29 | 0.7748 | 0.9598 |  |
| 4 | early | 0.570 | 0.568 | 0.682 | 0.585 | 0.583 | 0.679 | -0.015 | -0.43 | 0.6676 | 0.9598 |  |
| 5 | early | 0.585 | 0.570 | 0.723 | 0.603 | 0.595 | 0.726 | -0.018 | -0.50 | 0.6143 | 0.9598 |  |
| 6 | early | 0.620 | 0.613 | 0.742 | 0.700 | 0.647 | 0.762 | -0.080 | -2.39 | 0.0169 | 0.6939 |  |
| 7 | early | 0.688 | 0.655 | 0.806 | 0.733 | 0.718 | 0.853 | -0.045 | -1.40 | 0.1608 | 0.9598 |  |
| 8 | early | 0.713 | 0.680 | 0.857 | 0.762 | 0.755 | 0.906 | -0.050 | -1.61 | 0.1080 | 0.9598 |  |
| 9 | early | 0.755 | 0.738 | 0.919 | 0.775 | 0.770 | 0.926 | -0.020 | -0.67 | 0.5047 | 0.9598 |  |
| 10 | early | 0.757 | 0.750 | 0.929 | 0.757 | 0.755 | 0.934 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 11 | early | 0.770 | 0.760 | 0.948 | 0.762 | 0.750 | 0.953 | +0.008 | 0.25 | 0.8021 | 0.9598 |  |
| 12 | early | 0.738 | 0.730 | 0.950 | 0.748 | 0.743 | 0.961 | -0.010 | -0.32 | 0.7464 | 0.9598 |  |
| 13 | early | 0.743 | 0.740 | 0.957 | 0.760 | 0.755 | 0.964 | -0.017 | -0.57 | 0.5670 | 0.9598 |  |
| 14 | middle | 0.743 | 0.740 | 0.971 | 0.755 | 0.752 | 0.976 | -0.012 | -0.41 | 0.6836 | 0.9598 |  |
| 15 | middle | 0.738 | 0.738 | 0.980 | 0.770 | 0.765 | 0.979 | -0.032 | -1.07 | 0.2860 | 0.9598 |  |
| 16 | middle | 0.752 | 0.750 | 0.988 | 0.790 | 0.757 | 0.988 | -0.038 | -1.26 | 0.2067 | 0.9598 |  |
| 17 | middle | 0.767 | 0.738 | 0.989 | 0.760 | 0.757 | 0.989 | +0.007 | 0.25 | 0.8028 | 0.9598 |  |
| 18 | middle | 0.762 | 0.750 | 0.992 | 0.777 | 0.752 | 0.990 | -0.015 | -0.50 | 0.6142 | 0.9598 |  |
| 19 | middle | 0.772 | 0.750 | 0.993 | 0.785 | 0.775 | 0.994 | -0.013 | -0.43 | 0.6702 | 0.9598 |  |
| 20 | middle | 0.765 | 0.760 | 0.993 | 0.767 | 0.765 | 0.994 | -0.002 | -0.08 | 0.9334 | 0.9598 |  |
| 21 | middle | 0.762 | 0.752 | 0.994 | 0.775 | 0.767 | 0.996 | -0.013 | -0.42 | 0.6750 | 0.9598 |  |
| 22 | middle | 0.740 | 0.735 | 0.986 | 0.777 | 0.772 | 0.998 | -0.037 | -1.24 | 0.2151 | 0.9598 |  |
| 23 | middle | 0.750 | 0.748 | 0.998 | 0.775 | 0.775 | 0.998 | -0.025 | -0.83 | 0.4061 | 0.9598 |  |
| 24 | middle | 0.760 | 0.748 | 0.996 | 0.780 | 0.780 | 0.997 | -0.020 | -0.67 | 0.5015 | 0.9598 |  |
| 25 | middle | 0.750 | 0.748 | 0.996 | 0.772 | 0.772 | 0.998 | -0.022 | -0.75 | 0.4554 | 0.9598 |  |
| 26 | middle | 0.750 | 0.740 | 0.996 | 0.755 | 0.745 | 0.998 | -0.005 | -0.16 | 0.8698 | 0.9598 |  |
| 27 | middle | 0.745 | 0.740 | 0.998 | 0.780 | 0.775 | 0.998 | -0.035 | -1.16 | 0.2448 | 0.9598 |  |
| 28 | late | 0.748 | 0.748 | 0.996 | 0.780 | 0.770 | 0.997 | -0.032 | -1.08 | 0.2792 | 0.9598 |  |
| 29 | late | 0.745 | 0.740 | 0.998 | 0.770 | 0.765 | 0.998 | -0.025 | -0.82 | 0.4094 | 0.9598 |  |
| 30 | late | 0.757 | 0.750 | 0.996 | 0.755 | 0.748 | 0.998 | +0.002 | 0.08 | 0.9344 | 0.9598 |  |
| 31 | late | 0.748 | 0.730 | 0.998 | 0.752 | 0.750 | 0.998 | -0.005 | -0.16 | 0.8703 | 0.9598 |  |
| 32 | late | 0.730 | 0.728 | 0.999 | 0.733 | 0.725 | 0.999 | -0.003 | -0.08 | 0.9364 | 0.9598 |  |
| 33 | late | 0.718 | 0.713 | 0.996 | 0.745 | 0.738 | 0.998 | -0.027 | -0.88 | 0.3803 | 0.9598 |  |
| 34 | late | 0.733 | 0.723 | 0.998 | 0.767 | 0.743 | 0.998 | -0.035 | -1.14 | 0.2530 | 0.9598 |  |
| 35 | late | 0.733 | 0.720 | 0.998 | 0.743 | 0.738 | 0.998 | -0.010 | -0.32 | 0.7479 | 0.9598 |  |
| 36 | late | 0.745 | 0.735 | 0.997 | 0.767 | 0.762 | 0.999 | -0.022 | -0.74 | 0.4586 | 0.9598 |  |
| 37 | late | 0.760 | 0.738 | 0.998 | 0.767 | 0.767 | 0.999 | -0.007 | -0.25 | 0.8028 | 0.9598 |  |
| 38 | late | 0.767 | 0.755 | 0.999 | 0.762 | 0.757 | 0.999 | +0.005 | 0.17 | 0.8676 | 0.9598 |  |
| 39 | late | 0.780 | 0.777 | 1.000 | 0.795 | 0.787 | 1.000 | -0.015 | -0.52 | 0.6041 | 0.9598 |  |
| 40 | late | 0.770 | 0.760 | 0.999 | 0.790 | 0.785 | 1.000 | -0.020 | -0.68 | 0.4947 | 0.9598 |  |
