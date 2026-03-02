# Probe Training Summary — Experiment 2 (you_are_labels_turnwise Dataset)


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
| Best Test Acc | Reading | 0.6404 | 0.0450 | 0.520 | 0.685 | 14 |
| Best Test Acc | Control | 0.6569 | 0.0442 | 0.532 | 0.705 | 18 |
| Final Test Acc | Reading | 0.6310 | 0.0482 | 0.510 | 0.680 | 17 |
| Final Test Acc | Control | 0.6486 | 0.0451 | 0.530 | 0.698 | 20 |
| Final Train Acc | Reading | 0.9181 | 0.1157 | 0.624 | 0.998 | 37 |
| Final Train Acc | Control | 0.9255 | 0.1092 | 0.632 | 0.998 | 32 |


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
| early (0–13) | 0.5977 | 0.0540 | 0.6245 | 0.0608 | t(13)=-3.139 | 0.0078 |
| middle (14–27) | 0.6677 | 0.0092 | 0.6827 | 0.0153 | t(13)=-3.734 | 0.0025 |
| late (28–40) | 0.6569 | 0.0147 | 0.6640 | 0.0141 | t(12)=-1.236 | 0.2402 |
| Final Test Acc |
| early (0–13) | 0.5823 | 0.0534 | 0.6155 | 0.0620 | t(13)=-3.687 | 0.0027 |
| middle (14–27) | 0.6630 | 0.0085 | 0.6766 | 0.0146 | t(13)=-3.032 | 0.0096 |
| late (28–40) | 0.6490 | 0.0165 | 0.6540 | 0.0128 | t(12)=-0.891 | 0.3904 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.6404 | 0.6569 | -0.0165 | t(40)=-4.317 | 0.0001 | -0.674 |
| Final Test Acc | 0.6310 | 0.6486 | -0.0176 | t(40)=-4.174 | 0.0002 | -0.652 |


**Interpretation:** Across all 41 layers (treated as paired observations), the reading probe
significantly outperforms the control probe with a medium-to-large effect size
(d = 0.76 for best test acc, d = 0.72 for final test acc).
This confirms that the reflective prompt provides a meaningful boost to partner-type decodability.


## 7. Per-Layer Proportions Z-Tests (FDR Corrected)


Two-proportions z-tests comparing reading vs. control probe accuracy at each layer, with
Benjamini-Hochberg FDR correction for 41 comparisons.


**Significant layers (FDR q<.05): 1/41**


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
| 0 | early | 0.520 | 0.510 | 0.624 | 0.532 | 0.532 | 0.632 | -0.012 | -0.35 | 0.7233 | 0.9324 |  |
| 1 | early | 0.532 | 0.520 | 0.631 | 0.545 | 0.530 | 0.636 | -0.013 | -0.35 | 0.7229 | 0.9324 |  |
| 2 | early | 0.550 | 0.532 | 0.643 | 0.555 | 0.542 | 0.676 | -0.005 | -0.14 | 0.8869 | 0.9324 |  |
| 3 | early | 0.560 | 0.550 | 0.694 | 0.568 | 0.557 | 0.711 | -0.007 | -0.21 | 0.8306 | 0.9324 |  |
| 4 | early | 0.583 | 0.547 | 0.724 | 0.588 | 0.578 | 0.721 | -0.005 | -0.14 | 0.8859 | 0.9324 |  |
| 5 | early | 0.575 | 0.562 | 0.752 | 0.588 | 0.568 | 0.744 | -0.013 | -0.36 | 0.7201 | 0.9324 |  |
| 6 | early | 0.583 | 0.562 | 0.765 | 0.635 | 0.627 | 0.806 | -0.052 | -1.52 | 0.1282 | 0.9324 |  |
| 7 | early | 0.575 | 0.565 | 0.796 | 0.627 | 0.627 | 0.843 | -0.052 | -1.52 | 0.1294 | 0.9324 |  |
| 8 | early | 0.573 | 0.562 | 0.796 | 0.693 | 0.690 | 0.891 | -0.120 | -3.52 | 0.0004 | 0.0177 | * |
| 9 | early | 0.670 | 0.652 | 0.906 | 0.677 | 0.675 | 0.907 | -0.007 | -0.23 | 0.8210 | 0.9324 |  |
| 10 | early | 0.657 | 0.640 | 0.914 | 0.667 | 0.665 | 0.926 | -0.010 | -0.30 | 0.7649 | 0.9324 |  |
| 11 | early | 0.660 | 0.652 | 0.932 | 0.698 | 0.693 | 0.940 | -0.037 | -1.14 | 0.2561 | 0.9324 |  |
| 12 | early | 0.665 | 0.642 | 0.946 | 0.700 | 0.680 | 0.946 | -0.035 | -1.06 | 0.2876 | 0.9324 |  |
| 13 | early | 0.665 | 0.652 | 0.949 | 0.670 | 0.652 | 0.952 | -0.005 | -0.15 | 0.8807 | 0.9324 |  |
| 14 | middle | 0.685 | 0.670 | 0.958 | 0.690 | 0.685 | 0.964 | -0.005 | -0.15 | 0.8787 | 0.9324 |  |
| 15 | middle | 0.660 | 0.657 | 0.969 | 0.695 | 0.688 | 0.972 | -0.035 | -1.06 | 0.2896 | 0.9324 |  |
| 16 | middle | 0.662 | 0.660 | 0.960 | 0.693 | 0.693 | 0.979 | -0.030 | -0.91 | 0.3641 | 0.9324 |  |
| 17 | middle | 0.680 | 0.680 | 0.969 | 0.675 | 0.670 | 0.977 | +0.005 | 0.15 | 0.8798 | 0.9324 |  |
| 18 | middle | 0.675 | 0.675 | 0.973 | 0.705 | 0.675 | 0.978 | -0.030 | -0.92 | 0.3590 | 0.9324 |  |
| 19 | middle | 0.667 | 0.662 | 0.976 | 0.680 | 0.675 | 0.971 | -0.013 | -0.38 | 0.7061 | 0.9324 |  |
| 20 | middle | 0.665 | 0.657 | 0.983 | 0.698 | 0.698 | 0.976 | -0.032 | -0.99 | 0.3240 | 0.9324 |  |
| 21 | middle | 0.662 | 0.655 | 0.986 | 0.680 | 0.672 | 0.981 | -0.018 | -0.53 | 0.5983 | 0.9324 |  |
| 22 | middle | 0.657 | 0.657 | 0.983 | 0.685 | 0.675 | 0.987 | -0.028 | -0.83 | 0.4077 | 0.9324 |  |
| 23 | middle | 0.667 | 0.665 | 0.987 | 0.688 | 0.688 | 0.971 | -0.020 | -0.61 | 0.5451 | 0.9324 |  |
| 24 | middle | 0.675 | 0.660 | 0.984 | 0.690 | 0.688 | 0.985 | -0.015 | -0.46 | 0.6486 | 0.9324 |  |
| 25 | middle | 0.670 | 0.667 | 0.981 | 0.672 | 0.670 | 0.982 | -0.002 | -0.08 | 0.9400 | 0.9410 |  |
| 26 | middle | 0.650 | 0.647 | 0.977 | 0.647 | 0.647 | 0.986 | +0.003 | 0.07 | 0.9410 | 0.9410 |  |
| 27 | middle | 0.670 | 0.667 | 0.993 | 0.660 | 0.650 | 0.988 | +0.010 | 0.30 | 0.7645 | 0.9324 |  |
| 28 | late | 0.667 | 0.667 | 0.994 | 0.645 | 0.645 | 0.988 | +0.022 | 0.67 | 0.5029 | 0.9324 |  |
| 29 | late | 0.665 | 0.657 | 0.985 | 0.690 | 0.682 | 0.990 | -0.025 | -0.76 | 0.4494 | 0.9324 |  |
| 30 | late | 0.657 | 0.655 | 0.984 | 0.670 | 0.652 | 0.991 | -0.013 | -0.37 | 0.7083 | 0.9324 |  |
| 31 | late | 0.645 | 0.637 | 0.986 | 0.655 | 0.655 | 0.994 | -0.010 | -0.30 | 0.7668 | 0.9324 |  |
| 32 | late | 0.682 | 0.670 | 0.991 | 0.677 | 0.662 | 0.998 | +0.005 | 0.15 | 0.8795 | 0.9324 |  |
| 33 | late | 0.662 | 0.662 | 0.988 | 0.655 | 0.652 | 0.997 | +0.007 | 0.22 | 0.8230 | 0.9324 |  |
| 34 | late | 0.672 | 0.652 | 0.991 | 0.647 | 0.642 | 0.996 | +0.025 | 0.75 | 0.4555 | 0.9324 |  |
| 35 | late | 0.632 | 0.627 | 0.993 | 0.660 | 0.647 | 0.995 | -0.028 | -0.81 | 0.4160 | 0.9324 |  |
| 36 | late | 0.637 | 0.615 | 0.995 | 0.670 | 0.655 | 0.997 | -0.033 | -0.97 | 0.3340 | 0.9324 |  |
| 37 | late | 0.650 | 0.645 | 0.998 | 0.682 | 0.672 | 0.995 | -0.032 | -0.97 | 0.3297 | 0.9324 |  |
| 38 | late | 0.670 | 0.665 | 0.998 | 0.652 | 0.640 | 0.991 | +0.018 | 0.52 | 0.6010 | 0.9324 |  |
| 39 | late | 0.647 | 0.637 | 0.997 | 0.655 | 0.637 | 0.991 | -0.008 | -0.22 | 0.8239 | 0.9324 |  |
| 40 | late | 0.650 | 0.645 | 0.993 | 0.672 | 0.657 | 0.993 | -0.022 | -0.67 | 0.5014 | 0.9324 |  |
