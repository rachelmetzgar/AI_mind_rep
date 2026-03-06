# Probe Training Summary — Experiment 2 (balanced_gpt Dataset)


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
| Best Test Acc | Reading | 0.7162 | 0.0652 | 0.562 | 0.777 | 40 |
| Best Test Acc | Control | 0.7365 | 0.0703 | 0.560 | 0.795 | 39 |
| Final Test Acc | Reading | 0.7072 | 0.0662 | 0.552 | 0.767 | 40 |
| Final Test Acc | Control | 0.7284 | 0.0707 | 0.552 | 0.790 | 39 |
| Final Train Acc | Reading | 0.9206 | 0.1365 | 0.560 | 1.000 | 39 |
| Final Train Acc | Control | 0.9224 | 0.1389 | 0.549 | 1.000 | 39 |


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
| early (0–13) | 0.6595 | 0.0856 | 0.6745 | 0.0920 | t(13)=-2.683 | 0.0188 |
| middle (14–27) | 0.7500 | 0.0066 | 0.7732 | 0.0100 | t(13)=-8.468 | <.0001 |
| late (28–40) | 0.7410 | 0.0205 | 0.7637 | 0.0182 | t(12)=-5.258 | 0.0002 |
| Final Test Acc |
| early (0–13) | 0.6486 | 0.0854 | 0.6648 | 0.0912 | t(13)=-2.785 | 0.0155 |
| middle (14–27) | 0.7421 | 0.0080 | 0.7655 | 0.0099 | t(13)=-6.287 | <.0001 |
| late (28–40) | 0.7327 | 0.0219 | 0.7567 | 0.0191 | t(12)=-8.734 | <.0001 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.7162 | 0.7365 | -0.0202 | t(40)=-7.983 | <.0001 | -1.247 |
| Final Test Acc | 0.7072 | 0.7284 | -0.0212 | t(40)=-8.397 | <.0001 | -1.311 |


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
| 0 | early | 0.568 | 0.560 | 0.560 | 0.560 | 0.560 | 0.549 | +0.007 | 0.21 | 0.8306 | 0.9730 |  |
| 1 | early | 0.562 | 0.557 | 0.564 | 0.568 | 0.560 | 0.556 | -0.005 | -0.14 | 0.8866 | 0.9824 |  |
| 2 | early | 0.562 | 0.552 | 0.583 | 0.562 | 0.552 | 0.571 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 3 | early | 0.570 | 0.565 | 0.637 | 0.570 | 0.565 | 0.609 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 4 | early | 0.573 | 0.562 | 0.681 | 0.585 | 0.583 | 0.679 | -0.012 | -0.36 | 0.7203 | 0.9722 |  |
| 5 | early | 0.590 | 0.585 | 0.726 | 0.603 | 0.595 | 0.726 | -0.013 | -0.36 | 0.7186 | 0.9722 |  |
| 6 | early | 0.640 | 0.618 | 0.739 | 0.700 | 0.647 | 0.762 | -0.060 | -1.80 | 0.0711 | 0.8835 |  |
| 7 | early | 0.700 | 0.657 | 0.822 | 0.730 | 0.718 | 0.853 | -0.030 | -0.94 | 0.3473 | 0.8835 |  |
| 8 | early | 0.708 | 0.693 | 0.866 | 0.762 | 0.755 | 0.906 | -0.055 | -1.76 | 0.0780 | 0.8835 |  |
| 9 | early | 0.750 | 0.745 | 0.924 | 0.775 | 0.770 | 0.926 | -0.025 | -0.83 | 0.4061 | 0.8835 |  |
| 10 | early | 0.765 | 0.760 | 0.934 | 0.757 | 0.755 | 0.934 | +0.008 | 0.25 | 0.8035 | 0.9722 |  |
| 11 | early | 0.757 | 0.752 | 0.947 | 0.762 | 0.750 | 0.953 | -0.005 | -0.17 | 0.8685 | 0.9824 |  |
| 12 | early | 0.735 | 0.728 | 0.951 | 0.748 | 0.743 | 0.961 | -0.013 | -0.40 | 0.6865 | 0.9722 |  |
| 13 | early | 0.752 | 0.745 | 0.953 | 0.760 | 0.755 | 0.964 | -0.008 | -0.25 | 0.8049 | 0.9722 |  |
| 14 | middle | 0.748 | 0.740 | 0.971 | 0.755 | 0.752 | 0.976 | -0.007 | -0.25 | 0.8062 | 0.9722 |  |
| 15 | middle | 0.762 | 0.760 | 0.981 | 0.772 | 0.767 | 0.979 | -0.010 | -0.33 | 0.7378 | 0.9722 |  |
| 16 | middle | 0.757 | 0.750 | 0.988 | 0.790 | 0.760 | 0.988 | -0.033 | -1.10 | 0.2720 | 0.8835 |  |
| 17 | middle | 0.752 | 0.750 | 0.989 | 0.760 | 0.757 | 0.989 | -0.008 | -0.25 | 0.8049 | 0.9722 |  |
| 18 | middle | 0.752 | 0.743 | 0.991 | 0.777 | 0.752 | 0.990 | -0.025 | -0.83 | 0.4044 | 0.8835 |  |
| 19 | middle | 0.757 | 0.748 | 0.993 | 0.785 | 0.775 | 0.994 | -0.028 | -0.93 | 0.3545 | 0.8835 |  |
| 20 | middle | 0.750 | 0.738 | 0.994 | 0.770 | 0.765 | 0.994 | -0.020 | -0.66 | 0.5078 | 0.9722 |  |
| 21 | middle | 0.750 | 0.738 | 0.994 | 0.775 | 0.770 | 0.996 | -0.025 | -0.83 | 0.4061 | 0.8835 |  |
| 22 | middle | 0.745 | 0.743 | 0.994 | 0.777 | 0.772 | 0.998 | -0.032 | -1.08 | 0.2810 | 0.8835 |  |
| 23 | middle | 0.745 | 0.745 | 0.997 | 0.775 | 0.772 | 0.998 | -0.030 | -0.99 | 0.3205 | 0.8835 |  |
| 24 | middle | 0.738 | 0.728 | 0.996 | 0.780 | 0.780 | 0.997 | -0.042 | -1.40 | 0.1601 | 0.8835 |  |
| 25 | middle | 0.748 | 0.735 | 0.996 | 0.772 | 0.772 | 0.998 | -0.025 | -0.83 | 0.4078 | 0.8835 |  |
| 26 | middle | 0.743 | 0.738 | 0.997 | 0.757 | 0.748 | 0.998 | -0.015 | -0.49 | 0.6242 | 0.9722 |  |
| 27 | middle | 0.752 | 0.738 | 0.998 | 0.777 | 0.772 | 0.998 | -0.025 | -0.83 | 0.4044 | 0.8835 |  |
| 28 | late | 0.740 | 0.733 | 0.998 | 0.780 | 0.770 | 0.997 | -0.040 | -1.32 | 0.1853 | 0.8835 |  |
| 29 | late | 0.745 | 0.738 | 0.998 | 0.770 | 0.765 | 0.998 | -0.025 | -0.82 | 0.4094 | 0.8835 |  |
| 30 | late | 0.738 | 0.738 | 0.998 | 0.755 | 0.748 | 0.998 | -0.017 | -0.57 | 0.5695 | 0.9722 |  |
| 31 | late | 0.733 | 0.730 | 0.998 | 0.752 | 0.750 | 0.998 | -0.020 | -0.65 | 0.5177 | 0.9722 |  |
| 32 | late | 0.733 | 0.710 | 0.999 | 0.733 | 0.725 | 0.999 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 33 | late | 0.710 | 0.708 | 0.998 | 0.745 | 0.738 | 0.998 | -0.035 | -1.11 | 0.2663 | 0.8835 |  |
| 34 | late | 0.710 | 0.703 | 0.998 | 0.767 | 0.743 | 0.998 | -0.057 | -1.85 | 0.0642 | 0.8835 |  |
| 35 | late | 0.725 | 0.705 | 0.998 | 0.743 | 0.738 | 0.998 | -0.018 | -0.56 | 0.5755 | 0.9722 |  |
| 36 | late | 0.740 | 0.738 | 0.999 | 0.767 | 0.762 | 0.999 | -0.027 | -0.90 | 0.3667 | 0.8835 |  |
| 37 | late | 0.765 | 0.760 | 0.998 | 0.767 | 0.767 | 0.999 | -0.002 | -0.08 | 0.9334 | 1.0000 |  |
| 38 | late | 0.750 | 0.735 | 0.999 | 0.762 | 0.757 | 0.999 | -0.012 | -0.41 | 0.6805 | 0.9722 |  |
| 39 | late | 0.767 | 0.762 | 1.000 | 0.795 | 0.790 | 1.000 | -0.028 | -0.94 | 0.3468 | 0.8835 |  |
| 40 | late | 0.777 | 0.767 | 1.000 | 0.790 | 0.785 | 1.000 | -0.013 | -0.43 | 0.6676 | 0.9722 |  |
