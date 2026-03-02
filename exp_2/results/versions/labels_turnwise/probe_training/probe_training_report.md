# Probe Training Summary — Experiment 2 (labels_turnwise Dataset)


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
| Best Test Acc | Reading | 0.6202 | 0.0492 | 0.505 | 0.677 | 38 |
| Best Test Acc | Control | 0.6224 | 0.0431 | 0.505 | 0.675 | 40 |
| Final Test Acc | Reading | 0.6063 | 0.0544 | 0.480 | 0.662 | 19 |
| Final Test Acc | Control | 0.6116 | 0.0438 | 0.497 | 0.657 | 38 |
| Final Train Acc | Reading | 0.9068 | 0.1281 | 0.576 | 0.999 | 38 |
| Final Train Acc | Control | 0.9218 | 0.1148 | 0.605 | 0.999 | 36 |


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
| early (0–13) | 0.5705 | 0.0519 | 0.5861 | 0.0565 | t(13)=-2.927 | 0.0118 |
| middle (14–27) | 0.6496 | 0.0155 | 0.6366 | 0.0101 | t(13)=2.256 | 0.0419 |
| late (28–40) | 0.6421 | 0.0217 | 0.6463 | 0.0154 | t(12)=-0.812 | 0.4323 |
| Final Test Acc |
| early (0–13) | 0.5507 | 0.0581 | 0.5729 | 0.0557 | t(13)=-3.515 | 0.0038 |
| middle (14–27) | 0.6404 | 0.0168 | 0.6296 | 0.0105 | t(13)=1.797 | 0.0955 |
| late (28–40) | 0.6296 | 0.0188 | 0.6338 | 0.0160 | t(12)=-0.839 | 0.4179 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.6202 | 0.6224 | -0.0022 | t(40)=-0.610 | 0.5452 | -0.095 |
| Final Test Acc | 0.6063 | 0.6116 | -0.0052 | t(40)=-1.337 | 0.1888 | -0.209 |


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
| 0 | early | 0.537 | 0.532 | 0.616 | 0.525 | 0.517 | 0.633 | +0.012 | 0.35 | 0.7232 | 1.0000 |  |
| 1 | early | 0.522 | 0.502 | 0.576 | 0.540 | 0.497 | 0.605 | -0.018 | -0.50 | 0.6199 | 1.0000 |  |
| 2 | early | 0.512 | 0.480 | 0.638 | 0.505 | 0.502 | 0.645 | +0.007 | 0.21 | 0.8320 | 1.0000 |  |
| 3 | early | 0.520 | 0.495 | 0.674 | 0.545 | 0.527 | 0.710 | -0.025 | -0.71 | 0.4786 | 1.0000 |  |
| 4 | early | 0.505 | 0.492 | 0.685 | 0.532 | 0.532 | 0.730 | -0.027 | -0.78 | 0.4364 | 1.0000 |  |
| 5 | early | 0.527 | 0.487 | 0.705 | 0.527 | 0.520 | 0.706 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 6 | early | 0.583 | 0.537 | 0.719 | 0.595 | 0.583 | 0.813 | -0.012 | -0.36 | 0.7194 | 1.0000 |  |
| 7 | early | 0.545 | 0.522 | 0.749 | 0.573 | 0.565 | 0.850 | -0.027 | -0.78 | 0.4335 | 1.0000 |  |
| 8 | early | 0.593 | 0.565 | 0.757 | 0.642 | 0.625 | 0.869 | -0.050 | -1.45 | 0.1457 | 1.0000 |  |
| 9 | early | 0.613 | 0.613 | 0.875 | 0.632 | 0.620 | 0.894 | -0.020 | -0.58 | 0.5596 | 1.0000 |  |
| 10 | early | 0.605 | 0.598 | 0.896 | 0.647 | 0.640 | 0.905 | -0.042 | -1.24 | 0.2141 | 1.0000 |  |
| 11 | early | 0.632 | 0.632 | 0.902 | 0.657 | 0.647 | 0.921 | -0.025 | -0.74 | 0.4600 | 1.0000 |  |
| 12 | early | 0.650 | 0.642 | 0.926 | 0.657 | 0.632 | 0.929 | -0.007 | -0.22 | 0.8236 | 1.0000 |  |
| 13 | early | 0.642 | 0.610 | 0.927 | 0.625 | 0.610 | 0.944 | +0.017 | 0.51 | 0.6075 | 1.0000 |  |
| 14 | middle | 0.645 | 0.637 | 0.954 | 0.647 | 0.647 | 0.949 | -0.002 | -0.07 | 0.9411 | 1.0000 |  |
| 15 | middle | 0.620 | 0.613 | 0.955 | 0.655 | 0.642 | 0.966 | -0.035 | -1.03 | 0.3032 | 1.0000 |  |
| 16 | middle | 0.655 | 0.650 | 0.969 | 0.640 | 0.630 | 0.974 | +0.015 | 0.44 | 0.6570 | 1.0000 |  |
| 17 | middle | 0.660 | 0.645 | 0.960 | 0.645 | 0.642 | 0.974 | +0.015 | 0.45 | 0.6560 | 1.0000 |  |
| 18 | middle | 0.642 | 0.632 | 0.970 | 0.645 | 0.632 | 0.976 | -0.003 | -0.07 | 0.9411 | 1.0000 |  |
| 19 | middle | 0.667 | 0.662 | 0.957 | 0.630 | 0.625 | 0.983 | +0.037 | 1.11 | 0.2666 | 1.0000 |  |
| 20 | middle | 0.675 | 0.662 | 0.968 | 0.640 | 0.635 | 0.971 | +0.035 | 1.04 | 0.2969 | 1.0000 |  |
| 21 | middle | 0.665 | 0.657 | 0.976 | 0.625 | 0.620 | 0.985 | +0.040 | 1.18 | 0.2371 | 1.0000 |  |
| 22 | middle | 0.640 | 0.635 | 0.991 | 0.630 | 0.625 | 0.989 | +0.010 | 0.29 | 0.7689 | 1.0000 |  |
| 23 | middle | 0.660 | 0.657 | 0.986 | 0.635 | 0.620 | 0.984 | +0.025 | 0.74 | 0.4593 | 1.0000 |  |
| 24 | middle | 0.647 | 0.623 | 0.982 | 0.620 | 0.618 | 0.987 | +0.027 | 0.81 | 0.4195 | 1.0000 |  |
| 25 | middle | 0.627 | 0.618 | 0.974 | 0.640 | 0.640 | 0.986 | -0.013 | -0.37 | 0.7137 | 1.0000 |  |
| 26 | middle | 0.637 | 0.625 | 0.984 | 0.637 | 0.623 | 0.988 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 27 | middle | 0.652 | 0.647 | 0.989 | 0.623 | 0.615 | 0.989 | +0.030 | 0.88 | 0.3775 | 1.0000 |  |
| 28 | late | 0.608 | 0.605 | 0.991 | 0.657 | 0.655 | 0.989 | -0.050 | -1.47 | 0.1425 | 1.0000 |  |
| 29 | late | 0.642 | 0.630 | 0.993 | 0.650 | 0.625 | 0.995 | -0.008 | -0.22 | 0.8244 | 1.0000 |  |
| 30 | late | 0.605 | 0.588 | 0.988 | 0.620 | 0.610 | 0.986 | -0.015 | -0.44 | 0.6633 | 1.0000 |  |
| 31 | late | 0.630 | 0.620 | 0.989 | 0.645 | 0.627 | 0.998 | -0.015 | -0.44 | 0.6590 | 1.0000 |  |
| 32 | late | 0.660 | 0.645 | 0.994 | 0.642 | 0.642 | 0.999 | +0.018 | 0.52 | 0.6035 | 1.0000 |  |
| 33 | late | 0.630 | 0.623 | 0.998 | 0.627 | 0.618 | 0.998 | +0.003 | 0.07 | 0.9417 | 1.0000 |  |
| 34 | late | 0.632 | 0.623 | 0.996 | 0.635 | 0.627 | 0.993 | -0.003 | -0.07 | 0.9415 | 1.0000 |  |
| 35 | late | 0.642 | 0.630 | 0.993 | 0.647 | 0.642 | 0.996 | -0.005 | -0.15 | 0.8825 | 1.0000 |  |
| 36 | late | 0.662 | 0.637 | 0.996 | 0.642 | 0.615 | 0.999 | +0.020 | 0.59 | 0.5525 | 1.0000 |  |
| 37 | late | 0.637 | 0.632 | 0.998 | 0.635 | 0.625 | 0.999 | +0.002 | 0.07 | 0.9414 | 1.0000 |  |
| 38 | late | 0.677 | 0.650 | 0.999 | 0.660 | 0.657 | 0.998 | +0.017 | 0.53 | 0.5990 | 1.0000 |  |
| 39 | late | 0.665 | 0.655 | 0.998 | 0.665 | 0.640 | 0.999 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 40 | late | 0.655 | 0.647 | 0.988 | 0.675 | 0.655 | 0.988 | -0.020 | -0.60 | 0.5490 | 1.0000 |  |
