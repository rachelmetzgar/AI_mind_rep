# Experiment 5: Representational Similarity Analysis of Mental State Attribution Binding

## Methods — 5-Predictor Reduced Design (C1–C4)

---

### 1. Overview

We used Representational Similarity Analysis (RSA) to test whether LLaMA-2-13B-Chat maintains a dedicated representational structure for **mental state attributions** — the bound proposition *{subject + mental verb + object}* — that is distinct from representations of the component parts in isolation. Five binary model RDMs, entered simultaneously into a multiple regression at each layer, isolate the unique contribution of full mental state attribution geometry after partialing out confounds for sub-propositional features (verb-object binding, mental vocabulary, syntactic frame).

### 2. Stimuli

#### 2.1 Design

We constructed 56 stimulus items, each consisting of 4 sentences (conditions C1–C4), for a total of 224 sentences. Each item pairs a **mental state verb** with a matched **action verb** (maximally concrete/physical, selected to carry no mental state connotation) and a shared **object noun** that appears across all conditions. The grammatical subject is held constant as "He" in conditions that include a subject.

#### 2.2 Mental State Verb Categories

The 56 mental state verbs are drawn from 7 psychological categories (8 verbs each), following established folk psychology taxonomies:

| Category | Example Verbs |
|----------|--------------|
| Attention (8) | notices, observes, watches, sees, detects, examines, inspects, distinguishes |
| Memory (8) | remembers, recalls, forgets, recognizes, misremembers, reminisces, retains, recollects |
| Sensation (8) | feels, senses, perceives, tastes, smells, hears, touches, experiences |
| Belief (8) | believes, knows, assumes, trusts, doubts, thinks, suspects, supposes |
| Desire (8) | wants, craves, desires, needs, yearns for, pursues, seeks, prefers |
| Emotion (8) | fears, loves, dreads, envies, admires, hates, resents, cherishes |
| Intention (8) | contemplates, plans, expects, anticipates, ponders, decides, chooses, considers |

Each of the 56 action verbs is unique across items, and all action verbs pass a "security camera test" — the denoted action could be identified from silent video footage, ruling out implicit mentalistic content.

#### 2.3 The 4 Conditions

For item 1 (mental verb: *notices*, action verb: *fills*, object: *the crack*):

| Condition | Label | Template | Example |
|-----------|-------|----------|---------|
| C1 | mental_state | He [mental verb] the [object]. | *He notices the crack.* |
| C2 | dis_mental | [Mental verb] the [object]. | *Notice the crack.* |
| C3 | scr_mental | The [object] to [mental verb]. | *The crack to notice.* |
| C4 | action | He [action verb] the [object]. | *He fills the crack.* |

These four conditions are a reduced set from a full 6-condition design. Conditions C5 (dis_action: imperative action sentence without subject) and C6 (scr_action: scrambled action sentence) were excluded to increase the proportion of hypothesis-relevant C1×C1 pairs in the RDM from 2.7% to 6.2%, thereby improving statistical sensitivity to the target geometry.

#### 2.4 Condition Feature Matrix

Each condition controls a specific subset of the features present in the full mental state attribution (C1):

| Feature | C1 | C2 | C3 | C4 |
|---------|----|----|----|----|
| Subject "He" present | Yes | No | No | Yes |
| Mental verb present | Yes | Yes | Yes | No |
| Action verb present | No | No | No | Yes |
| Grammatical word order | Yes | Yes | No | Yes |
| Scrambled form | No | No | Yes | No |
| Object noun present | Yes | Yes | Yes | Yes |
| Mental verb + object (grammatical) | Yes | Yes | No | No |
| Subject + verb + object frame | Yes | No | No | Yes |

This feature matrix motivates the five model RDMs described below.

### 3. Activation Extraction

#### 3.1 Model

All activations were extracted from **LLaMA-2-13B-Chat** (Meta; 40 transformer layers plus the embedding layer = 41 layers total; hidden dimensionality = 5,120). The model was loaded in float16 precision with HuggingFace Transformers.

#### 3.2 Procedure

Each of the 224 sentences was presented to the model as a bare string (no chat template, no system prompt, no instruction tags). For each sentence, we performed a single forward pass and extracted the hidden state vector at the **last token position** from each of the 41 layers (embedding layer through layer 40). This produced an activation matrix of shape (224, 41, 5120).

Last-token position was chosen because autoregressive language models concentrate contextual integration at the final token — the position from which the next-token prediction is generated. At this position, the model's representation reflects its full processing of the entire sentence.

### 4. Neural RDM Construction

At each of the 41 layers, we constructed a 224 × 224 **neural representational dissimilarity matrix (RDM)**. Each cell (i, j) of the neural RDM contains the **correlation distance** (1 − Pearson *r*) between the 5,120-dimensional activation vectors for sentences *i* and *j*. Correlation distance was chosen over Euclidean distance because it is invariant to differences in activation magnitude across layers and focuses on the pattern of relative activation values — the representational geometry rather than the scale.

The lower triangle of each RDM was extracted as a vector of 24,976 unique pairwise distances (224 × 223 / 2) for all subsequent analyses.

### 5. Model RDMs

We constructed five binary theoretical RDMs (224 × 224), each encoding a hypothesis about which sentence pairs should be represented similarly. In each model RDM, pairs predicted to be similar receive a value of 0 and all other pairs receive a value of 1. The diagonal is 0.

The key design principle: each model RDM predicts similarity for all sentence pairs where **both sentences belong to a specified set of conditions**, regardless of item identity. This means predictions apply across items (e.g., "He notices the crack" is predicted similar to "He believes the story" by Model A, because both are C1 sentences), not just within items. This cross-item similarity structure is what allows us to test for condition-level representational geometry.

#### 5.1 Model A: Full Attribution

**Predicts similarity when both sentences ∈ {C1}.**

Model A is the target hypothesis. It predicts that C1 sentences — and only C1 sentences — cluster together in representational space. A significant positive beta for Model A (after partialing out all confounds) would indicate that the model maintains a representational geometry unique to the bound proposition *{He + mental verb + object}*, above and beyond the contributions of any sub-propositional feature.

#### 5.2 Model B: Mental Verb + Object Binding

**Predicts similarity when both sentences ∈ {C1, C2}.**

C1 and C2 share the same mental verb and object in grammatical order (verb-object binding), but C2 lacks a subject. A significant beta for Model B indicates that the model represents grammatical mental-verb+object binding, regardless of whether a subject is present. By including B in the regression, any beta for A is guaranteed to reflect structure beyond verb-object binding.

#### 5.3 Model C: Mental Verb Presence

**Predicts similarity when both sentences ∈ {C1, C2, C3}.**

C1, C2, and C3 all contain the same mental verb (and object noun). C3 presents these words in scrambled order. A significant beta for C would indicate that the mere presence of mental vocabulary drives clustering, regardless of syntactic structure. Including C ensures that any Model A effect is not explained by lexical co-occurrence of mental state words.

#### 5.4 Model D: Verb + Object Binding (Regardless of Verb Type)

**Predicts similarity when both sentences ∈ {C1, C2, C4}.**

C1, C2, and C4 all present a verb followed by an object in grammatical order. C4 uses an action verb rather than a mental verb. A significant beta for D indicates that the model represents generic grammatical verb-object frames, independent of whether the verb is mental or physical. Including D partials out the contribution of syntactic structure (SVO or VO frame) from Model A.

#### 5.5 Model E: Subject + Verb + Object Frame

**Predicts similarity when both sentences ∈ {C1, C4}.**

C1 and C4 both have the full subject-verb-object structure ("He [verb] the [object]"), but they differ in verb type (mental vs. action). A significant beta for E indicates that sentences with a grammatical agent share representational structure. Including E ensures that any Model A effect is not merely an artifact of shared syntactic frame (subject presence + grammatical order).

#### 5.6 Regression Logic

The five models are entered simultaneously into an OLS multiple regression:

> neural_RDM = β_A · Model_A + β_B · Model_B + β_C · Model_C + β_D · Model_D + β_E · Model_E + ε

Because the models are entered simultaneously (not sequentially), each beta reflects the **unique** contribution of that model after accounting for all others. A significant β_A therefore means: C1 sentences share a representational geometry that cannot be explained by any additive combination of mental verb+object binding (B), mental vocabulary presence (C), generic verb+object frame (D), or subject+verb+object structure (E).

This is the critical test. If the model has internalized mental state attribution as a structured operation, the full bound proposition should produce a representational signature that is irreducible to its components.

### 6. Multicollinearity Check

Because the model RDMs share overlapping condition sets (all include C1; B is nested within C; etc.), multicollinearity is a concern. Before running the regression, we computed **Variance Inflation Factors (VIF)** for all five predictors.

VIF is computed for each predictor by regressing that predictor on all remaining predictors and computing 1 / (1 − R²). The standard interpretation is:

- VIF < 5: acceptable
- VIF 5–10: caution warranted
- VIF > 10: regression coefficients unreliable

Observed VIFs: Model A = 2.02, Model B = 4.57, Model C = 2.75, Model D = 3.92, Model E = 2.49. All VIFs fall below 5, indicating acceptable multicollinearity. The highest VIF (Model B = 4.57) reflects B's nesting within C (both include C1 and C2), but remains within conventional bounds.

We also computed the full pairwise Spearman correlation matrix between model RDM vectors to verify the collinearity structure and confirm no perfect or near-perfect correlations.

### 7. Regression Procedure

#### 7.1 Standardization

Both the neural RDM vector (dependent variable) and all model RDM vectors (predictors) were z-scored (mean-subtracted and divided by standard deviation) before regression. This ensures that betas are standardized regression coefficients, interpretable as the effect of each predictor in standard-deviation units.

#### 7.2 OLS Estimation

Betas were estimated via ordinary least squares: **β = (X'X)^{−1} X'y**, where X is the (24,976 × 5) design matrix and y is the (24,976 × 1) neural RDM vector.

#### 7.3 Semi-Partial Correlation and Unique Variance (ΔR²)

For each predictor, we computed the **unique variance** (ΔR²) — the increment in R² when that predictor is added to a model already containing all other predictors. This was computed as:

> ΔR²_i = (SS_res,reduced − SS_res,full) / SS_total

where *SS_res,reduced* is the residual sum of squares from the model excluding predictor *i*, and *SS_res,full* is the residual sum of squares from the full model.

The **semi-partial correlation** (sr) was computed as the signed square root of ΔR²:

> sr_i = sign(β_i) · √(ΔR²_i)

Semi-partial r indicates the correlation between the neural RDM and the unique (non-shared) portion of predictor *i*, preserving sign information that ΔR² discards.

### 8. Permutation Testing

Statistical significance was assessed via permutation testing (10,000 permutations per layer), which avoids parametric assumptions about the distribution of RDM correlations that are violated by the non-independence of entries in a distance matrix.

#### 8.1 Permutation Scheme

On each permutation iteration, we shuffled **condition labels within items**. That is, for each of the 56 items, the assignment of the item's 4 activation vectors to conditions C1–C4 was randomly permuted, while item membership was preserved. This procedure destroys condition-level structure while maintaining the item-level correlation structure of the neural data.

Formally, let π be a permutation of {0, 1, 2, 3} applied independently to each item's 4 rows. The permuted neural RDM was constructed by reindexing both rows and columns of the original neural RDM according to the concatenated permutation vector, yielding RDM_perm = RDM[π, π]. The full 5-predictor regression was then re-run on the permuted neural RDM, and the beta for each model was recorded.

Each layer used a unique random seed (42 + layer number) to ensure reproducibility while avoiding correlated null distributions across layers.

#### 8.2 P-Value Computation

For each predictor at each layer, the two-tailed p-value was computed as:

> p = proportion of permutation |β_null| ≥ |β_observed|

This tests the null hypothesis that the observed beta (positive or negative) is no larger in magnitude than expected by chance.

### 9. Multiple Comparisons Correction

Because the regression is run independently at each of 41 layers, we applied **Benjamini-Hochberg False Discovery Rate (FDR) correction** at q = 0.05, separately for each model predictor across layers. This controls the expected proportion of false positives among rejected null hypotheses, rather than the family-wise error rate, providing a balance between Type I error control and statistical power.

FDR correction was applied after all layers were computed. The procedure ranks the 41 p-values for each predictor from smallest to largest, then adjusts each p-value as: p_adjusted,i = p_i × 41 / rank_i, with monotonicity enforced (each adjusted p is at least as large as any smaller-ranked adjusted p).

### 10. Interpretation of Results

#### 10.1 What a Significant Positive β_A Means

A significant positive β_A at a given layer indicates that, at that layer, C1 sentence pairs (both sentences are full mental state attributions) are **more similar** to each other than predicted by the combined effects of all confound models. This would constitute evidence for a bound mental state attribution representation.

#### 10.2 What a Significant Negative β_A Means

A significant negative β_A indicates that C1 sentence pairs are **more dissimilar** to each other than expected — that is, full mental state attributions *individuate* in representational space. This is an interpretable finding: it suggests that the model differentiates among specific attributions (e.g., "He notices the crack" vs. "He believes the story") more than it differentiates among the control conditions. In other words, full attributions occupy a larger region of representational space, consistent with finer-grained semantic encoding for propositions that bind a mental state to an agent and object.

#### 10.3 Layer Profile

The analysis yields a complete profile of each predictor's beta across all 41 layers (embedding layer through final transformer layer), revealing which layers carry representational structure relevant to each feature. Significant effects at early layers may reflect lexical or syntactic processing, while effects at middle and late layers are more likely to reflect compositional semantic structure.

### 11. Software and Reproducibility

- **Model**: LLaMA-2-13B-Chat (Meta), loaded via HuggingFace Transformers in float16 precision
- **Distance metric**: Correlation distance (1 − Pearson r) computed via `scipy.spatial.distance.pdist`
- **Regression**: Custom OLS implementation (NumPy `linalg.solve`), z-scored predictors
- **Permutation testing**: 10,000 iterations per layer, condition-label shuffling within items
- **FDR correction**: Benjamini-Hochberg at q = 0.05
- **Random number generation**: NumPy `default_rng` with seed = 42 + layer_number
- **All code**: `exp_5/code/rsa/5_predictors/`

---

*Rachel C. Metzgar · March 2026*
