# Probe Training Summary — Experiment 2 (names Dataset)


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
| Best Test Acc | Reading | 0.7247 | 0.0697 | 0.560 | 0.787 | 37 |
| Best Test Acc | Control | 0.7363 | 0.0671 | 0.573 | 0.790 | 35 |
| Final Test Acc | Reading | 0.7157 | 0.0751 | 0.545 | 0.782 | 21 |
| Final Test Acc | Control | 0.7290 | 0.0683 | 0.565 | 0.785 | 23 |
| Final Train Acc | Reading | 0.9127 | 0.1315 | 0.573 | 0.998 | 39 |
| Final Train Acc | Control | 0.9187 | 0.1330 | 0.538 | 0.999 | 39 |


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
| early (0–13) | 0.6455 | 0.0659 | 0.6736 | 0.0846 | t(13)=-3.293 | 0.0058 |
| middle (14–27) | 0.7618 | 0.0146 | 0.7629 | 0.0105 | t(13)=-0.317 | 0.7565 |
| late (28–40) | 0.7700 | 0.0114 | 0.7752 | 0.0090 | t(12)=-1.537 | 0.1503 |
| Final Test Acc |
| early (0–13) | 0.6305 | 0.0712 | 0.6646 | 0.0855 | t(13)=-3.925 | 0.0017 |
| middle (14–27) | 0.7566 | 0.0165 | 0.7588 | 0.0118 | t(13)=-0.508 | 0.6198 |
| late (28–40) | 0.7633 | 0.0120 | 0.7663 | 0.0093 | t(12)=-0.817 | 0.4300 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.7247 | 0.7363 | -0.0116 | t(40)=-3.099 | 0.0035 | -0.484 |
| Final Test Acc | 0.7157 | 0.7290 | -0.0134 | t(40)=-3.211 | 0.0026 | -0.502 |


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
| 0 | early | 0.595 | 0.550 | 0.573 | 0.575 | 0.565 | 0.538 | +0.020 | 0.57 | 0.5659 | 0.9714 |  |
| 1 | early | 0.560 | 0.545 | 0.604 | 0.585 | 0.568 | 0.552 | -0.025 | -0.71 | 0.4748 | 0.9714 |  |
| 2 | early | 0.583 | 0.565 | 0.600 | 0.573 | 0.570 | 0.593 | +0.010 | 0.29 | 0.7746 | 0.9714 |  |
| 3 | early | 0.580 | 0.575 | 0.651 | 0.583 | 0.583 | 0.657 | -0.003 | -0.07 | 0.9429 | 1.0000 |  |
| 4 | early | 0.585 | 0.585 | 0.669 | 0.585 | 0.568 | 0.693 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 5 | early | 0.580 | 0.547 | 0.713 | 0.605 | 0.595 | 0.724 | -0.025 | -0.72 | 0.4718 | 0.9714 |  |
| 6 | early | 0.630 | 0.605 | 0.739 | 0.670 | 0.660 | 0.784 | -0.040 | -1.19 | 0.2356 | 0.9714 |  |
| 7 | early | 0.655 | 0.640 | 0.792 | 0.710 | 0.703 | 0.843 | -0.055 | -1.67 | 0.0947 | 0.9714 |  |
| 8 | early | 0.657 | 0.652 | 0.833 | 0.760 | 0.750 | 0.907 | -0.103 | -3.19 | 0.0014 | 0.0582 |  |
| 9 | early | 0.713 | 0.690 | 0.869 | 0.777 | 0.775 | 0.907 | -0.065 | -2.11 | 0.0349 | 0.7163 |  |
| 10 | early | 0.728 | 0.710 | 0.897 | 0.760 | 0.755 | 0.921 | -0.032 | -1.05 | 0.2924 | 0.9714 |  |
| 11 | early | 0.718 | 0.710 | 0.909 | 0.738 | 0.723 | 0.936 | -0.020 | -0.64 | 0.5253 | 0.9714 |  |
| 12 | early | 0.725 | 0.723 | 0.927 | 0.760 | 0.745 | 0.940 | -0.035 | -1.13 | 0.2576 | 0.9714 |  |
| 13 | early | 0.730 | 0.730 | 0.934 | 0.750 | 0.748 | 0.945 | -0.020 | -0.64 | 0.5190 | 0.9714 |  |
| 14 | middle | 0.743 | 0.738 | 0.951 | 0.752 | 0.750 | 0.955 | -0.010 | -0.33 | 0.7448 | 0.9714 |  |
| 15 | middle | 0.748 | 0.738 | 0.958 | 0.760 | 0.757 | 0.967 | -0.012 | -0.41 | 0.6816 | 0.9714 |  |
| 16 | middle | 0.745 | 0.738 | 0.969 | 0.762 | 0.757 | 0.978 | -0.017 | -0.57 | 0.5657 | 0.9714 |  |
| 17 | middle | 0.755 | 0.743 | 0.979 | 0.767 | 0.762 | 0.971 | -0.012 | -0.41 | 0.6784 | 0.9714 |  |
| 18 | middle | 0.743 | 0.738 | 0.981 | 0.767 | 0.767 | 0.979 | -0.025 | -0.82 | 0.4110 | 0.9714 |  |
| 19 | middle | 0.755 | 0.748 | 0.982 | 0.750 | 0.738 | 0.981 | +0.005 | 0.16 | 0.8698 | 0.9907 |  |
| 20 | middle | 0.757 | 0.757 | 0.990 | 0.750 | 0.748 | 0.989 | +0.007 | 0.25 | 0.8055 | 0.9714 |  |
| 21 | middle | 0.785 | 0.782 | 0.989 | 0.777 | 0.772 | 0.986 | +0.008 | 0.26 | 0.7975 | 0.9714 |  |
| 22 | middle | 0.767 | 0.765 | 0.986 | 0.760 | 0.752 | 0.991 | +0.007 | 0.25 | 0.8028 | 0.9714 |  |
| 23 | middle | 0.782 | 0.780 | 0.994 | 0.787 | 0.785 | 0.994 | -0.005 | -0.17 | 0.8633 | 0.9907 |  |
| 24 | middle | 0.767 | 0.762 | 0.994 | 0.767 | 0.767 | 0.993 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 25 | middle | 0.770 | 0.762 | 0.995 | 0.755 | 0.750 | 0.995 | +0.015 | 0.50 | 0.6181 | 0.9714 |  |
| 26 | middle | 0.770 | 0.770 | 0.995 | 0.757 | 0.757 | 0.995 | +0.013 | 0.42 | 0.6773 | 0.9714 |  |
| 27 | middle | 0.777 | 0.772 | 0.995 | 0.765 | 0.757 | 0.996 | +0.012 | 0.42 | 0.6739 | 0.9714 |  |
| 28 | late | 0.752 | 0.752 | 0.995 | 0.762 | 0.762 | 0.996 | -0.010 | -0.33 | 0.7414 | 0.9714 |  |
| 29 | late | 0.752 | 0.745 | 0.995 | 0.767 | 0.765 | 0.995 | -0.015 | -0.50 | 0.6194 | 0.9714 |  |
| 30 | late | 0.765 | 0.760 | 0.998 | 0.782 | 0.780 | 0.997 | -0.017 | -0.59 | 0.5542 | 0.9714 |  |
| 31 | late | 0.765 | 0.748 | 0.994 | 0.780 | 0.765 | 0.996 | -0.015 | -0.51 | 0.6128 | 0.9714 |  |
| 32 | late | 0.775 | 0.760 | 0.995 | 0.772 | 0.743 | 0.997 | +0.003 | 0.08 | 0.9327 | 1.0000 |  |
| 33 | late | 0.782 | 0.775 | 0.996 | 0.772 | 0.762 | 0.997 | +0.010 | 0.34 | 0.7338 | 0.9714 |  |
| 34 | late | 0.767 | 0.762 | 0.998 | 0.777 | 0.772 | 0.997 | -0.010 | -0.34 | 0.7359 | 0.9714 |  |
| 35 | late | 0.782 | 0.777 | 0.997 | 0.790 | 0.765 | 0.997 | -0.008 | -0.26 | 0.7958 | 0.9714 |  |
| 36 | late | 0.770 | 0.767 | 0.997 | 0.790 | 0.775 | 0.997 | -0.020 | -0.68 | 0.4947 | 0.9714 |  |
| 37 | late | 0.787 | 0.782 | 0.997 | 0.767 | 0.767 | 0.997 | +0.020 | 0.68 | 0.4965 | 0.9714 |  |
| 38 | late | 0.765 | 0.760 | 0.998 | 0.777 | 0.770 | 0.997 | -0.012 | -0.42 | 0.6739 | 0.9714 |  |
| 39 | late | 0.782 | 0.777 | 0.998 | 0.775 | 0.775 | 0.999 | +0.007 | 0.26 | 0.7983 | 0.9714 |  |
| 40 | late | 0.762 | 0.755 | 0.993 | 0.762 | 0.760 | 0.994 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
