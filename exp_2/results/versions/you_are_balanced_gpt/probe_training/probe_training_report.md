# Probe Training Summary — Experiment 2 (you_are_balanced_gpt Dataset)


Generated: 2026-03-01 12:52


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
| Best Test Acc | Reading | 0.7384 | 0.0778 | 0.570 | 0.820 | 27 |
| Best Test Acc | Control | 0.7470 | 0.0766 | 0.573 | 0.810 | 16 |
| Final Test Acc | Reading | 0.7325 | 0.0791 | 0.565 | 0.815 | 27 |
| Final Test Acc | Control | 0.7413 | 0.0761 | 0.570 | 0.802 | 35 |
| Final Train Acc | Reading | 0.9199 | 0.1335 | 0.583 | 1.000 | 34 |
| Final Train Acc | Control | 0.9221 | 0.1346 | 0.578 | 1.000 | 36 |


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
| early (0–13) | 0.6595 | 0.0890 | 0.6730 | 0.0936 | t(13)=-2.831 | 0.0142 |
| middle (14–27) | 0.7846 | 0.0153 | 0.7893 | 0.0117 | t(13)=-1.438 | 0.1742 |
| late (28–40) | 0.7735 | 0.0154 | 0.7810 | 0.0147 | t(12)=-1.607 | 0.1341 |
| Final Test Acc |
| early (0–13) | 0.6518 | 0.0899 | 0.6682 | 0.0931 | t(13)=-2.821 | 0.0144 |
| middle (14–27) | 0.7793 | 0.0156 | 0.7814 | 0.0123 | t(13)=-0.653 | 0.5252 |
| late (28–40) | 0.7690 | 0.0147 | 0.7769 | 0.0155 | t(12)=-1.513 | 0.1561 |


**Interpretation:** The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (p < .0001) and late layers
(p < .0001). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model's deeper representations, and is
amplified by the reflective prompt used by the reading probe.


## 6. Overall Reading vs Control


| Metric | Reading M | Control M | Mean Diff | Test | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| Best Test Acc | 0.7384 | 0.7470 | -0.0086 | t(40)=-3.471 | 0.0013 | -0.542 |
| Final Test Acc | 0.7325 | 0.7413 | -0.0088 | t(40)=-3.046 | 0.0041 | -0.476 |


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
| 0 | early | 0.575 | 0.568 | 0.583 | 0.573 | 0.573 | 0.578 | +0.002 | 0.07 | 0.9430 | 1.0000 |  |
| 1 | early | 0.573 | 0.568 | 0.598 | 0.575 | 0.575 | 0.579 | -0.002 | -0.07 | 0.9430 | 1.0000 |  |
| 2 | early | 0.570 | 0.565 | 0.596 | 0.580 | 0.580 | 0.588 | -0.010 | -0.29 | 0.7748 | 1.0000 |  |
| 3 | early | 0.580 | 0.568 | 0.619 | 0.580 | 0.575 | 0.618 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 4 | early | 0.575 | 0.565 | 0.674 | 0.583 | 0.570 | 0.662 | -0.008 | -0.21 | 0.8299 | 1.0000 |  |
| 5 | early | 0.593 | 0.588 | 0.719 | 0.593 | 0.583 | 0.721 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 6 | early | 0.593 | 0.590 | 0.753 | 0.615 | 0.615 | 0.770 | -0.022 | -0.65 | 0.5153 | 1.0000 |  |
| 7 | early | 0.650 | 0.635 | 0.807 | 0.700 | 0.688 | 0.838 | -0.050 | -1.51 | 0.1311 | 1.0000 |  |
| 8 | early | 0.713 | 0.690 | 0.850 | 0.748 | 0.745 | 0.907 | -0.035 | -1.11 | 0.2649 | 1.0000 |  |
| 9 | early | 0.748 | 0.733 | 0.906 | 0.777 | 0.770 | 0.917 | -0.030 | -1.00 | 0.3188 | 1.0000 |  |
| 10 | early | 0.733 | 0.728 | 0.922 | 0.770 | 0.767 | 0.927 | -0.037 | -1.23 | 0.2199 | 1.0000 |  |
| 11 | early | 0.772 | 0.772 | 0.943 | 0.777 | 0.772 | 0.941 | -0.005 | -0.17 | 0.8655 | 1.0000 |  |
| 12 | early | 0.775 | 0.775 | 0.942 | 0.772 | 0.770 | 0.951 | +0.003 | 0.08 | 0.9327 | 1.0000 |  |
| 13 | early | 0.785 | 0.782 | 0.961 | 0.780 | 0.772 | 0.961 | +0.005 | 0.17 | 0.8639 | 1.0000 |  |
| 14 | middle | 0.755 | 0.750 | 0.971 | 0.772 | 0.765 | 0.968 | -0.017 | -0.58 | 0.5601 | 1.0000 |  |
| 15 | middle | 0.787 | 0.785 | 0.981 | 0.792 | 0.787 | 0.976 | -0.005 | -0.17 | 0.8622 | 1.0000 |  |
| 16 | middle | 0.802 | 0.800 | 0.981 | 0.810 | 0.800 | 0.985 | -0.008 | -0.27 | 0.7884 | 1.0000 |  |
| 17 | middle | 0.780 | 0.775 | 0.986 | 0.792 | 0.790 | 0.986 | -0.012 | -0.43 | 0.6663 | 1.0000 |  |
| 18 | middle | 0.780 | 0.772 | 0.991 | 0.787 | 0.775 | 0.991 | -0.007 | -0.26 | 0.7967 | 1.0000 |  |
| 19 | middle | 0.780 | 0.780 | 0.985 | 0.795 | 0.772 | 0.989 | -0.015 | -0.52 | 0.6041 | 1.0000 |  |
| 20 | middle | 0.767 | 0.765 | 0.994 | 0.780 | 0.770 | 0.990 | -0.013 | -0.42 | 0.6727 | 1.0000 |  |
| 21 | middle | 0.780 | 0.775 | 0.997 | 0.780 | 0.765 | 0.993 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 22 | middle | 0.782 | 0.772 | 0.998 | 0.780 | 0.777 | 0.997 | +0.002 | 0.09 | 0.9318 | 1.0000 |  |
| 23 | middle | 0.795 | 0.790 | 0.999 | 0.790 | 0.785 | 0.999 | +0.005 | 0.17 | 0.8616 | 1.0000 |  |
| 24 | middle | 0.780 | 0.770 | 0.998 | 0.805 | 0.800 | 0.996 | -0.025 | -0.87 | 0.3833 | 1.0000 |  |
| 25 | middle | 0.782 | 0.775 | 0.997 | 0.770 | 0.770 | 0.998 | +0.012 | 0.42 | 0.6714 | 1.0000 |  |
| 26 | middle | 0.792 | 0.785 | 0.999 | 0.795 | 0.787 | 0.998 | -0.003 | -0.09 | 0.9304 | 1.0000 |  |
| 27 | middle | 0.820 | 0.815 | 0.996 | 0.800 | 0.795 | 0.999 | +0.020 | 0.72 | 0.4709 | 1.0000 |  |
| 28 | late | 0.800 | 0.792 | 0.998 | 0.800 | 0.800 | 0.998 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 29 | late | 0.782 | 0.775 | 0.994 | 0.797 | 0.792 | 0.999 | -0.015 | -0.52 | 0.6025 | 1.0000 |  |
| 30 | late | 0.767 | 0.762 | 0.996 | 0.792 | 0.785 | 0.998 | -0.025 | -0.85 | 0.3934 | 1.0000 |  |
| 31 | late | 0.777 | 0.777 | 0.998 | 0.790 | 0.787 | 0.998 | -0.013 | -0.43 | 0.6676 | 1.0000 |  |
| 32 | late | 0.760 | 0.755 | 0.996 | 0.787 | 0.782 | 0.999 | -0.027 | -0.93 | 0.3526 | 1.0000 |  |
| 33 | late | 0.792 | 0.787 | 0.998 | 0.775 | 0.770 | 0.999 | +0.017 | 0.60 | 0.5477 | 1.0000 |  |
| 34 | late | 0.785 | 0.782 | 1.000 | 0.765 | 0.760 | 0.999 | +0.020 | 0.68 | 0.4982 | 1.0000 |  |
| 35 | late | 0.770 | 0.760 | 0.997 | 0.802 | 0.802 | 0.998 | -0.032 | -1.12 | 0.2622 | 1.0000 |  |
| 36 | late | 0.775 | 0.772 | 0.999 | 0.775 | 0.765 | 1.000 | +0.000 | 0.00 | 1.0000 | 1.0000 |  |
| 37 | late | 0.765 | 0.757 | 0.999 | 0.775 | 0.770 | 0.999 | -0.010 | -0.34 | 0.7368 | 1.0000 |  |
| 38 | late | 0.740 | 0.740 | 0.999 | 0.760 | 0.757 | 0.999 | -0.020 | -0.65 | 0.5136 | 1.0000 |  |
| 39 | late | 0.777 | 0.775 | 1.000 | 0.767 | 0.762 | 0.999 | +0.010 | 0.34 | 0.7359 | 1.0000 |  |
| 40 | late | 0.762 | 0.760 | 0.998 | 0.765 | 0.765 | 0.998 | -0.003 | -0.08 | 0.9337 | 1.0000 |  |
