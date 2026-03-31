# Experiment 5: Residual Activation Probing — Methods

## 1. Overview

We used **residual activation probing** to test whether LLaMA-2-13B-Chat maintains a representation of mental state attribution (C1: "He [mental verb] the [object]") that is irreducible to three binary confound features: subject presence, mental verb presence, and grammaticality. This analysis complements the 5-predictor RSA (which operates in distance space) and the Gram-Schmidt critical tests (which operate on probe weight vectors) by directly deconfounding the activation data via OLS regression before training a new probe from scratch.

The logic is straightforward: if the model's representation of C1 is fully explained by the additive combination of subject + mental verb + grammaticality, then removing these features from the activation vectors should eliminate any ability to distinguish C1 from other conditions. If the residual probe still classifies C1 above chance, the model encodes attribution-specific information beyond the three features.

## 2. Stimuli

We used the same 4-condition subset (C1–C4, 224 sentences) as the 5-predictor RSA analysis. See the 5-predictor methods document for full stimulus description. Briefly:

- **C1 (mental_state):** "He notices the crack." — Full mental state attribution
- **C2 (dis_mental):** "Notice the crack." — Mental verb + object, no subject
- **C3 (scr_mental):** "The crack to notice." — Scrambled mental verb + object
- **C4 (action):** "He fills the crack." — Subject + action verb + object

56 items × 4 conditions = 224 sentences.

## 3. Activation Extraction

Activations were extracted from LLaMA-2-13B-Chat at three token positions (verb, object, period) across all 41 layers (embedding through layer 40), producing a (224, 3, 41, 5120) activation tensor. See the main methods document for extraction details.

## 4. Confound Feature Matrix

We constructed a (224 × 4) design matrix **F** with an intercept column and three binary feature columns:

| Feature | C1 | C2 | C3 | C4 |
|---------|----|----|----|----|
| Subject presence | 1 | 0 | 0 | 1 |
| Mental verb | 1 | 1 | 1 | 0 |
| Grammaticality | 1 | 1 | 0 | 1 |

**Action verb** was excluded because it is perfectly anti-correlated with mental verb in the C1–C4 subset (every sentence has exactly one verb type), so including both would create a rank-deficient design matrix.

The design matrix has rank 4 (intercept + 3 features), verified computationally. Note that C1 is the *unique* condition satisfying subject=1, mental_verb=1, grammaticality=1. No single feature or feature pair uniquely identifies C1; only the conjunction of all three does. This is precisely why OLS residualization is informative: it removes all linear information about these features, and anything that survives must encode a higher-order or nonlinearly-combined representation.

## 5. OLS Residualization

For each (position, layer) combination, we applied OLS residualization to the raw activation matrix:

**X_resid = M · X_raw**

where **M** is the residual-maker (hat complement) matrix:

**M = I − F(F'F)^{−1}F'**

This 224 × 224 matrix is computed once and reused across all (position, layer) combinations. For any activation column vector **x**, the product **Mx** removes the component of **x** that lies in the column space of **F** — i.e., any linear dependence on the intercept and the three confound features. The residual **Mx** is guaranteed to be orthogonal to all columns of **F**, which we verified computationally (max|F' · X_resid| < 10^{−6}).

## 6. Probe Training

At each (position, layer), we trained a logistic regression classifier (L2-regularized, C=1.0, LBFGS solver, balanced class weights) to discriminate C1 sentences (label=1, n=56) from C2+C3+C4 sentences (label=0, n=168). The `class_weight="balanced"` parameter adjusts for the 1:3 class imbalance by weighting each class inversely proportional to its frequency.

Two probes were trained at each (position, layer):
- **Residual probe:** Trained on OLS-residualized activations (confound features removed)
- **Raw baseline probe:** Trained on unresidualized activations (same C1-C4 subset)

The raw baseline provides a within-analysis comparison showing how much signal survives deconfounding.

## 7. Cross-Validation

We used **leave-one-item-out (LOIO) cross-validation** with 56 folds. Each fold holds out all 4 sentences belonging to one item (one sentence per condition). This ensures:

1. **Item-level generalization:** The probe must generalize to items not seen during training, not just interpolate within the same item's conditions.
2. **Balanced test sets:** Each held-out fold contains exactly 1 C1 sentence and 3 non-C1 sentences.
3. **No information leakage:** Sentences from the same item (which share the same verb and object) never appear in both training and test sets.

## 8. Metrics

- **Accuracy:** Proportion of correctly classified sentences across all held-out folds.
- **ROC AUC:** Area under the receiver operating characteristic curve, computed from predicted probabilities. AUC is the primary metric because it is threshold-independent and handles class imbalance gracefully.

## 9. Permutation Testing

Statistical significance was assessed via permutation testing (200 iterations per position × layer combination). On each iteration, the label vector was randomly shuffled (destroying the C1-vs-rest structure while preserving the marginal distribution), and the full LOIO CV procedure was repeated. The p-value is the proportion of null AUC values ≥ the observed AUC.

To accelerate the permutation test, activations were PCA-reduced to min(256, n−4) = 220 components before probe training. Because n=224 < d=5120, PCA with 220 components preserves all variance in the data (the remaining components have zero variance), making this a lossless dimensionality reduction that eliminates redundant zero-variance dimensions without discarding any information.

## 10. FDR Correction

Benjamini-Hochberg False Discovery Rate correction was applied at q = 0.05, separately within each (probe_type, position) group across the 41 layers. This controls the expected proportion of false positives among rejected null hypotheses.

## 11. Raw Baseline Comparison

**Signal retention** quantifies how much of the raw probe's above-chance signal survives deconfounding:

**retention = (AUC_residual − 0.5) / (AUC_raw − 0.5)**

computed at each position's peak layer. A retention of 1.0 means all signal survives; 0 means the confound features fully explain the raw signal.

## 12. Interpretation

**Significant residual probe:** C1 is distinguishable from C2+C3+C4 even after removing subject presence, mental verb, and grammaticality from the activations. This constitutes evidence for an irreducible mental state attribution representation — the model encodes something about the bound proposition {subject + mental verb + object} that cannot be reconstructed from its parts.

**Non-significant residual probe:** The raw probe's success was entirely driven by the additive combination of the three confound features. The model does not maintain a dedicated attribution representation beyond what these features predict.

**Caveats:** (1) OLS residualization removes *linear* effects of the confounds; nonlinear confound encoding could survive. (2) With n=224 (56 positive), statistical power is limited — a null result does not prove absence. (3) The three features chosen do not exhaust all possible confounds (e.g., sentence length differences remain).

---

*Rachel C. Metzgar · March 2026*
