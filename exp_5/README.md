# Experiment 5 — Mental State Attribution Bound Representation

## Overview

Tests whether LLaMA-2 maintains a dedicated representational structure for **mental state attributions** — the bound proposition {subject + mental state verb + object} — that is distinct from representations of the component parts in isolation.

The core claim: if the model has genuine mental state attribution machinery, then sentences like "He believes the story" should produce a representational geometry that is NOT explained by:
- The presence of mental state vocabulary alone
- The syntactic frame alone
- The lexical identity of any individual word

The geometry should ONLY emerge when all three components (subject, mental state verb, object) are bound together in grammatical order.

## Theoretical Motivation

Mental state attribution (also called "mentalizing" or "theory of mind") requires representing a structured proposition: AGENT + MENTAL STATE + CONTENT. This is distinct from:
- Knowing what mental state words mean (lexical knowledge)
- Processing subject-verb-object syntax (syntactic parsing)
- Representing mental states without attributing them to an agent

If the model has internalized mental state attribution as a structured operation, we should find a representational signature that is unique to the full bound attribution and absent when any component is removed or scrambled.

## Stimuli

### Design

56 items x 6 conditions = 336 sentences total.

**Subject** is fixed to "He" throughout (eliminates subject as a variable).

Each item has a **mental state verb** and a matched **action verb** (maximally concrete/physical, no mental state leakage). The same object noun appears across all 6 conditions for a given item.

### The 6 Conditions

For item 1 (verb: "notices", action: "fills", object: "the crack"):

| Condition | Label | Example | What it controls |
|-----------|-------|---------|-----------------|
| 1 | `mental_state` | He notices the crack. | Full attribution: subject + mental verb + object |
| 2 | `dis_mental` | Notice the crack. | Mental verb + object, NO subject |
| 3 | `scr_mental` | The crack to notice. | Same words as cond 2, scrambled order |
| 4 | `action` | He fills the crack. | Subject + action verb + object (same syntactic frame, no mental state) |
| 5 | `dis_action` | Fill the crack. | Action verb + object, NO subject |
| 6 | `scr_action` | The crack to fill. | Same words as cond 5, scrambled order |

### Mental State Verb Categories (7 categories x 8 verbs = 56 items)

| Category | Verbs |
|----------|-------|
| **Attention** (8) | notices, observes, watches, sees, detects, examines, inspects, distinguishes |
| **Memory** (8) | remembers, recalls, forgets, recognizes, misremembers, reminisces, retains, recollects |
| **Sensation** (8) | feels, senses, perceives, tastes, smells, hears, touches, experiences |
| **Belief** (8) | believes, knows, assumes, trusts, doubts, thinks, suspects, supposes |
| **Desire** (8) | wants, craves, desires, needs, yearns for, pursues, seeks, prefers |
| **Emotion** (8) | fears, loves, dreads, envies, admires, hates, resents, cherishes |
| **Intention** (8) | contemplates, plans, expects, anticipates, ponders, decides, chooses, considers |

### Action Verb Constraints
- Every action verb is unique (no repeats across the 56 items)
- All action verbs are maximally concrete/physical ("security camera test": you could identify the action from silent video)
- Action verbs should NOT imply mental states (no "decided," "ignored," "avoided," etc.)

### Condition Feature Matrix

| Feature | C1 (mental_state) | C2 (dis_mental) | C3 (scr_mental) | C4 (action) | C5 (dis_action) | C6 (scr_action) |
|---|---|---|---|---|---|---|
| Has subject "He" | yes | no | no | yes | no | no |
| Mental verb present | yes | yes | yes | no | no | no |
| Action verb present | no | no | no | yes | yes | yes |
| Grammatical word order | yes | yes | no | yes | yes | no |
| Scrambled form ("The X to Y") | no | no | yes | no | no | yes |
| Shared object noun (within item) | yes | yes | yes | yes | yes | yes |

## Three Analysis Approaches

We test the core claim with three independent methods at increasing levels of causal strength:

1. **RSA** (correlational) — Does the overall representational geometry across 336 sentences match attribution-specific structure, after partialing out confounds like verb type, subject presence, and word overlap?
2. **Probes** (supervised) — Can we find specific linear directions that encode the bound attribution, and do those directions survive Gram-Schmidt removal of simpler feature directions?
3. **Interchange intervention** (causal) — When we swap verb activations between sentences, do within-type swaps (mental→mental) transfer better than cross-type swaps (mental→action), indicating a dedicated binding mechanism?

Each approach has its own code subdirectory (`code/rsa/`, `code/probes/`, `code/interchange/`) and can be run independently.

---

## Approach 1: RSA Analysis Pipeline

### Step 1: Extract Activations

Run all 336 sentences through LLaMA-2-13B-Chat. For each sentence, extract the activation vector at each layer at the **last token position** (and optionally mean across all token positions). Produces a matrix of shape `(336, hidden_dim)` per layer.

### Step 2: Compute Neural RDM

For each layer, compute pairwise distances between all 336 activation vectors using **correlation distance** (1 - Pearson r). Yields a 336x336 symmetric matrix. Extract the lower triangle (56,280 unique pairs) as a vector.

---

### Analysis 1: Simple RSA

**Model A — Full Attribution:**
- Condition 1 x Condition 1 pairs: **0** (similar)
- All other pairs: **1** (dissimilar)

Procedure: Spearman correlation between neural RDM vector and Model A vector. Test via permutation (shuffle condition labels within items, 10,000 iterations). Run at each layer, BH-FDR correct across layers.

If significant: condition 1 items cluster together more than items from any other condition. Because conditions 2-3 share mental verbs (predicted dissimilar by Model A), and condition 4 shares syntactic frame (also predicted dissimilar), the stimulus design implicitly controls for lexical and syntactic confounds.

---

### Analysis 2: Partial RSA

Multiple regression of the neural RDM on Model A plus all confound models simultaneously, asking whether Model A explains unique variance above and beyond all confounds.

**7 Model RDMs (each 336x336, binary 0/1):**

| Model | Predicts similarity for | What it controls |
|-------|------------------------|-----------------|
| **A** — Full Attribution | C1xC1 pairs only | *The hypothesis*: bound subject + mental verb + object |
| **B** — Mental Verb Presence | C1xC1, C2xC2, C3xC3, C1xC2, C1xC3, C2xC3 | Lexical confound: mental vocabulary clusters |
| **C** — Subject Presence | C1xC1, C4xC4, C1xC4 | Syntactic confound: having a subject/agent |
| **D** — Item Identity | All 6 versions of same item with each other | Word overlap confound: shared object noun |
| **F** — Grammatical Order | C1xC1, C2xC2, C4xC4, C5xC5, cross-pairs of these | Grammaticality confound: well-formed vs scrambled |
| **G** — Scrambled Form | C3xC3, C6xC6, C3xC6 | Scrambled confound: "The X to Y" structure |
| **H** — Action Verb Presence | C4xC4, C5xC5, C6xC6, C4xC5, C4xC6, C5xC6 | Action lexical confound: mirror of Model B |

**Primary analysis:** Regression `neural_RDM ~ A + B + C + D + F + G + H`. Semi-partial correlation for A = unique variance for full attribution after partialing out all confounds.

**Secondary analysis:** Swap Model A for Model E:
- **E** — Mental Verb + Object, Subject-Optional: C1xC1, C2xC2, C1xC2 pairs similar.
- Tests whether mental verb + object in grammatical order suffices without subject.

Significance via permutation (10,000 iterations, shuffle condition labels within items). Run at each layer, BH-FDR correct.

**Interpreting A vs E:**
- beta_A sig, beta_E not: subject is necessary for attribution structure
- beta_E sig, beta_A not: mental verb + object binding suffices without subject
- Both sig: core verb+object structure that subject enriches
- beta_E sig, beta_A not beyond confounds: subject adds nothing

---

### Analysis 3: Within-Condition Category Structure RSA

Within condition 1 only (56 sentences), test whether the model organizes mental state attributions by the 7 verb categories.

**Model Cat — Mental State Category Structure:**
- Same-category pairs: **0** (similar) — 28 within-category pairs per category, 196 total
- Different-category pairs: **1** (dissimilar) — 1,344 pairs

Spearman correlation between 56x56 neural RDM and Model Cat. Permutation test (shuffle category labels, 10,000 iterations). Run at each layer, BH-FDR correct.

**Optional extensions:**
- **Graded category similarity** using Gray et al.'s Experience x Agency framework (Attention/Sensation = Experience-heavy; Belief/Intention/Desire = Agency-heavy; Emotion = Experience-heavy; Memory = mixed). Tests whether the model's mental state geometry aligns with folk-psychological dimensional structure (connects to Exp 4).
- **Cross-condition comparison:** Run category RSA on conditions 2-6 independently. If category structure appears in C1 but not C2-C3, it requires the full attribution form. If also in C2, verb semantics alone organize the space.

## Expected Results

**If model has mental state attribution machinery:**
- Simple RSA: Model A significant at middle-to-late layers
- Partial RSA: beta_A significant after partialing out all confounds
- Layer profile for beta_A peaks in layers associated with semantic/conceptual processing

**If result is just lexical:** beta_B (mental verb presence) significant, beta_A NOT significant after partialing out B.

**If result is just syntactic:** beta_C (subject presence) or beta_F (grammatical order) significant, beta_A not.

**If result is just word overlap:** beta_D (item identity) significant, beta_A not.

**If mental state attributions are internally structured by category (Analysis 3):**
- Model Cat significant in C1: model organizes attributions by type
- Model Cat significant in C1 but NOT C2-C6: category structure requires full attribution form
- Graded Experience/Agency model fits better than binary: aligns with folk-psychological dimensions (connects to Exp 4)

## Model

- Primary: LLaMA-2-13B-Chat (for comparability with Exps 2-4)

## RSA Variant Analyses

Three additional variant analyses cross distance metric x stimulus type to test robustness:

| Variant | Distance Metric | Stimuli | Purpose |
|---------|----------------|---------|---------|
| **baseline** | correlation (1 - Pearson r) | original | Primary analysis |
| **cosine** | cosine | original | Test metric sensitivity |
| **corr_you** | correlation | "You" (C2/C5 get "You" subject) | Test whether adding subject to imperatives changes structure |
| **cosine_you** | cosine | "You" | Combined metric + stimuli change |

"You" stimuli: C2 "Notice the crack." becomes "You notice the crack." C5 "Fill the crack." becomes "You fill the crack." All other conditions unchanged.

Results saved to `results/{model}/rsa/data/{variant}/` subfolders, with the same CSV file names as the baseline analysis.

## Reduced 4-Condition RSA

A supplementary analysis that drops C5 (dis_action) and C6 (scr_action) entirely. The motivation: C5 and C6 never directly test Model A — their "similar" blocks don't overlap with C1×C1 pairs. Removing them:

- Increases C1×C1 pair proportion from 2.7% to 6.2% of total pairs
- Drops Models G and H (which rely on C5/C6)
- Reduces multicollinearity (4 confound regressors instead of 7)
- Sharpens standard errors on β_A

**Conditions retained:** C1 (mental_state), C2 (dis_mental), C3 (scr_mental), C4 (action).

**Combined regression:**
```
RDM_neural = β_A·A + β_E·E + β_B·B + β_C·C + β_D·D + β_F·F + ε
```

Both A and E are in the same regression. β_A = unique variance for full attribution BEYOND verb+object binding (E) and all confounds. β_E = unique variance for verb+object binding BEYOND full attribution (A).

**Model F redefined for 4 conditions:** C1, C2, C4 are grammatical (was C1, C2, C4, C5 in the 6-condition design).

Results in `results/{model}/rsa/data/reduced_4cond/`.

---

## Approach 2: Probe Training

### Overview and Key Questions

The RSA analyses above test whether the *overall* representational geometry matches attribution-specific structure. The probe pipeline asks a complementary question: can we find specific *directions* in the activation space that encode attribution-relevant features, and do those directions compose into an attribution-specific signal?

Two distinct claims are at stake:

**Claim 1: The model can distinguish full attributions (C1) from everything else.**
A c1_vs_all binary classifier that reliably discriminates C1 from C2-C6 demonstrates this. Because the stimulus design already controls for confounds (C2 shares mental verb + object; C4 shares subject + syntactic frame; C3/C5/C6 share various subsets), any probe that successfully classifies C1 must be picking up on the *conjunction* of features — subject + mental verb + object in grammatical order.

**Claim 2: The model has a dedicated attribution direction that is NOT reducible to the sum of individual feature directions.**
Even if a c1_vs_all probe works, the direction it finds might be a linear combination of simpler features (subject presence, mental verb presence, grammaticality). Claim 2 asks: after Gram-Schmidt orthogonalization removes the directions that encode these individual features, does a *residual* attribution direction remain? If yes, the model has a representation that cannot be decomposed into its component parts — a genuinely emergent attribution signal.

### Method: Binary Linear Probes

All probes use **logistic regression** (sklearn `LogisticRegression`, `C=1.0`, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=10000`).

**Cross-validation:** Leave-one-item-out (LOIO). 56 folds, each holding out all 6 conditions for one item. This ensures train/test splits never share the same object noun, preventing lexical leakage.

**Evaluation:** Accuracy and AUC (area under ROC curve) on held-out folds.

**Permutation testing:** Shuffle class labels at the *item* level (all 6 conditions for an item move together), retrain and evaluate. This preserves the within-item correlation structure. 200 iterations for feature probes, 10,000 for critical tests.

**FDR correction:** Benjamini-Hochberg across layers within each (probe_type, token_position) combination.

### Phase 1: Feature Probes

Four binary probes that test whether the model encodes individual features of the stimulus design. Trained at all 3 token positions (verb, object, period) × 41 layers = 123 combinations.

| Probe | Positive class | Negative class | What it detects |
|-------|---------------|----------------|-----------------|
| **subject_presence** | C1, C4 (112 samples) | C2, C3, C5, C6 (224 samples) | Whether "He" is present |
| **mental_verb** | C1, C2, C3 (168 samples) | C4, C5, C6 (168 samples) | Whether the verb is a mental state verb |
| **grammaticality** | C1, C2, C4, C5 (224 samples) | C3, C6 (112 samples) | Whether words are in grammatical order |
| **action_verb** | C4, C5, C6 (168 samples) | C1, C2, C3 (168 samples) | Whether the verb is an action verb |

These probes serve two purposes:
1. Establish that individual features are linearly decodable (sanity check)
2. Provide weight vectors for Gram-Schmidt orthogonalization in the critical tests

### Phase 2: Attribution Probes

Three binary probes that target the attribution signal at different levels of specificity.

| Probe | Positive | Negative | What it tests |
|-------|----------|----------|---------------|
| **c1_vs_all** | C1 (56) | C2-C6 (280) | Full attribution vs everything else — **Claim 1** |
| **c1_vs_c2** | C1 (56) | C2 (56) | Subject's contribution to mental verb binding |
| **c4_vs_c5** | C4 (56) | C5 (56) | Subject's contribution to action verb binding (control) |

**Interpretation of c1_vs_all:** If this probe works (AUC >> 0.5, p < 0.05), the model distinguishes C1 from all other conditions. Given the stimulus design — where every individual feature (subject, mental verb, grammaticality) is shared with at least one other condition — the probe must be sensitive to the conjunction. This is sufficient for **Claim 1**.

**Interpretation of c1_vs_c2 vs c4_vs_c5:** Both c1_vs_c2 and c4_vs_c5 test the effect of adding a subject ("He") while keeping everything else constant. If c1_vs_c2 works better than c4_vs_c5, the subject matters more in mental verb contexts than in action verb contexts — an interaction that suggests attribution-specific binding.

### Phase 3: Critical Tests

Performed at the peak (position, layer) identified from the attribution probes.

#### Test 3a: Residual Probe Direction (Gram-Schmidt)

1. Load trained weight vectors: **w_attr** (c1_vs_all), **w_subj** (subject_presence), **w_mental** (mental_verb), **w_gram** (grammaticality)
2. Gram-Schmidt orthogonalization: sequentially project out w_subj, w_mental, and w_gram from w_attr:
   ```
   w_residual_1 = w_attr - (w_attr · ŵ_subj) ŵ_subj
   w_residual_2 = w_residual_1 - (w_residual_1 · ŵ_mental) ŵ_mental
   w_residual   = w_residual_2 - (w_residual_2 · ŵ_gram) ŵ_gram
   ```
   where ŵ denotes a unit vector.
3. Project all 336 activations onto w_residual
4. Compute AUC for classifying C1 vs rest using the 1D projection
5. Permutation test: 10,000 iterations, shuffle C1/rest labels

**Interpretation:** If AUC is significantly above chance after removing individual feature directions, the model has an attribution direction that cannot be decomposed into subject + mental verb + grammaticality. This is **Claim 2**.

If AUC drops to chance, the c1_vs_all probe's success was entirely explained by a linear combination of simpler features.

#### Test 3b: Interaction Direction

A model-free approach to finding the attribution-specific direction:

```
delta_mental = mean(C1 activations) - mean(C2 activations)
delta_action = mean(C4 activations) - mean(C5 activations)
w_interaction = delta_mental - delta_action
```

delta_mental captures the effect of adding "He" to a mental verb sentence. delta_action captures the same for action verbs. Their *difference* isolates the attribution-specific component of subject addition — the part that is unique to mental state contexts.

Project 336 activations onto w_interaction, compute AUC + permutation test.

#### Test 3c: Direction Comparison

Cosine similarity between w_c1vc2 and w_c4vc5 (the probe weight vectors). Bootstrap 10,000 iterations: resample items, retrain both probes, compute cosine, get 95% CI.

**Interpretation:** If cosine ≈ 1: subject adds the same information regardless of verb type (generic syntactic effect). If cosine << 1: subject integration is verb-type-specific — evidence of attribution-specific binding.

### Phase 4: Probe-Projected RSA

If the critical tests reveal significant attribution directions, use them to focus the RSA:

1. Collect significant directions (w_residual, w_interaction) from Phase 3
2. Orthogonalize into a projection matrix **P** (k × 5120)
3. Project all 336 activations: z_proj = activations @ P.T → (336, k)
4. Compute RDM from projected activations
5. Run full partial RSA with all model RDMs (same as Analysis 2)
6. Run category RSA on projected C1 activations (same as Analysis 3)

**Rationale:** If Model A was non-significant in the full partial RSA because the attribution signal lives in a thin subspace overwhelmed by other variance, projecting onto the probe-identified directions should amplify it.

**Interpretation:** If β_A becomes significant after projection: the attribution structure exists but is low-dimensional (thin subspace). If still non-significant: the probe was picking up on something other than the attribution geometry.

### Phase 5: Category Probes

Test whether the model organizes mental state attributions by the 7 verb categories (Attention, Memory, Sensation, Belief, Desire, Emotion, Intention).

| Analysis | Training data | Test data | What it tests |
|----------|--------------|-----------|---------------|
| **5a** | C1 only (56 samples, 7 classes) | LOIO CV | Category structure within full attributions |
| **5b** | C2 only (56 samples, 7 classes) | LOIO CV | Category structure in verb+object (no subject) |
| **5c** | All C1 | All C2 (and vice versa) | Cross-condition generalization of category structure |

7-way multinomial logistic regression. Chance = 14.3%.

**Interpretation:** If 5a > chance but 5b ≈ chance: category organization requires the full attribution form. If both work: category structure comes from verb semantics alone. If 5c works (train on C1, test on C2): category representation generalizes across sentence forms — shared verb-semantic substrate.

### Summary: Which Results Support Which Claims

| Result Pattern | Claim 1 | Claim 2 | Interpretation |
|---|---|---|---|
| c1_vs_all significant | yes | — | Model distinguishes C1 from rest |
| c1_vs_all significant + residual AUC at chance | yes | no | Attribution probe = sum of feature probes |
| c1_vs_all significant + residual AUC significant | yes | yes | Dedicated attribution direction exists |
| c1_vs_c2 >> c4_vs_c5 | — | supports | Subject binding is attribution-specific |
| cosine(w_c1vc2, w_c4vc5) << 1 | — | supports | Different binding mechanisms for mental vs action |
| Projected RSA: β_A significant | — | supports | Attribution geometry exists in thin subspace |
| Category probe: 5a > chance, 5b ≈ chance | — | — | Category structure requires full attribution form |

---

## Approach 3: Interchange Intervention Pipeline

### Overview

Tests whether the model uses a DIFFERENT binding mechanism for mental state attributions vs action sentences, adapted from the entity binding literature (Feng & Steinhardt, ICLR 2024; Gur-Arieh et al., 2025).

The core idea: swap verb activations between sentences and measure how well the swap transfers. If mental state verbs are bound differently from action verbs, within-condition swaps (mental→mental) should work better than cross-condition swaps (mental→action). This is direct causal evidence of a dedicated attribution binding mechanism.

### Activation Extraction

For each of 336 sentences, extract the FULL residual stream at EVERY token position and EVERY layer. Save per-sentence `.npz` files `(n_tokens, 41, 5120)` in float16. Also save a token-position map identifying verb, object, subject (C1/C4 only), and period token indices.

**Token position rules by condition:**
- C1/C4 ("He [verb] the [obj]."): subject=token after BOS, verb=next token, period=last
- C2/C5 ("[Verb] the [obj]."): verb=first token after BOS, period=last
- C3/C6 ("The [obj] to [verb]."): object=after BOS, verb=before period

Multi-subword verbs use the LAST subword token index.

### Step 1: Verb Swap Interventions

For each pair of sentences (A, B) at each layer L:
1. Register a forward hook on layer L's transformer block
2. Run sentence A's forward pass; the hook replaces A's verb activation with B's cached verb activation at that layer
3. Collect the intervened final representation at A's period token

**Swap success metric:**
```
swap_success = cos(rep_A_intervened, rep_B_original) - cos(rep_A_intervened, rep_A_original)
```

**Which pairs to swap:**
- **Within-condition:** 56 × 10 random partners = 560 swaps per condition (6 conditions)
- **Cross-condition, same item:** 56 swaps per condition pair (30 ordered pairs) — controls for object
- **Cross-condition, different item:** 56 swaps per condition pair (1 random partner per item)

**Layer subsampling:** Start with 8 layers (5, 10, 15, 20, 25, 30, 35, 40), then fill in around the peak.

### Step 2: Transfer Matrix Analysis

Build a 6×6 transfer matrix at each layer: `transfer_matrix[source, target] = mean(swap_success)`.

**Key contrasts (permutation tests, 10K iterations):**

| Test | Contrast | What it tests |
|------|----------|---------------|
| 1 | C1→C1 vs C1→C4 | Mental verbs swap better into mental contexts? |
| 2 | C4→C4 vs C4→C1 | Action verbs swap better into action contexts? |
| 3 | C1→C1 vs C1→C2 | Does the subject matter for mental verb binding? |
| 4 | C1→C2 vs C1→C4 | Verb type vs subject presence for binding compatibility |
| 5 | (C1→C1 − C1→C2) vs (C4→C4 − C4→C5) | **Attribution interaction**: subject matters MORE for mental binding? |

**Block structure analysis:** Test whether the transfer matrix has a 2×2 block structure (mental C1-C3 vs action C4-C6). Block index = within-type mean − cross-type mean. Positive and significant = verb-type-specific binding.

**Verb similarity control:** Regress `swap_success ~ verb_embedding_similarity + same_condition + same_verb_type` using layer-0 embeddings. If `same_condition` coefficient is significant after controlling for embedding similarity, binding is condition-specific, not just driven by verb semantics.

### Step 3: Subject Token Swaps

Swap the "He" activation between C1 and C4 sentences (both have "He" at the same position). Since "He" is the same word, any difference in its activation between C1 and C4 reflects verb-type-specific information written back to the subject position by attention.

**Metric:** `subject_swap_effect = 1 - cos(rep_intervened, rep_original)`

**Control:** Swap "He" between two C1 sentences (within-condition, 56 × 10 random partners per layer).

**Key comparison:** If cross-type effect (C1↔C4) > within-type effect (C1↔C1), the model writes verb-type-specific information to the subject position — evidence that "He" is bound differently in mental state vs action contexts. This effect should emerge at mid-to-late layers (after attention propagates verb info backward).

### Expected Results

| Hypothesis | Prediction |
|---|---|
| **Dedicated attribution binding** | Block structure significant; Test 5 interaction significant; subject swap cross > within at mid-late layers |
| **Generic binding** | Transfer matrix roughly uniform; no block structure; subject swap equal across conditions |
| **Verb semantics only** | Block structure exists BUT verb similarity control regression shows same_condition is NOT significant |

---

## Code Organization

Three independent analysis pipelines, each in its own subdirectory. Shared infrastructure (`config.py`, `stimuli.py`, `utils/`) lives in `code/`.

```
code/
  config.py             # paths, model config — single source of truth
  stimuli.py            # 56 items × 6 conditions = 336 sentences
  utils/
    rsa.py              # RDM construction, model RDMs, permutation tests, FDR
    probes.py           # LOIO CV, permutation test, Gram-Schmidt utilities
    token_positions.py  # verb/object/subject/period token identification

  rsa/                  # Pipeline A: Representational Similarity Analysis
    1_extract_activations.py        # GPU — extract last-token activations
    2_simple_rsa.py                 # CPU — Model A vs neural RDM
    3_partial_rsa.py                # CPU — partial RSA (A + confounds, E + confounds)
    4_category_rsa.py               # CPU — within-condition category structure
    5_reduced_rsa.py                # CPU — reduced 4-condition RSA (combined A+E)
    6_extract_you_activations.py    # GPU — "You" variant activations
    7_variant_rsa.py                # CPU — variant analyses (cosine, corr_you, cosine_you)
    8a_report_generator.py          # login — HTML report
    slurm/

  probes/               # Pipeline B: Probe Training
    1_extract_multipos_activations.py   # GPU — verb/object/period activations
    2_feature_probes.py                 # CPU — subject, mental verb, grammaticality, action
    3_attribution_probes.py             # CPU — c1_vs_all, c1_vs_c2, c4_vs_c5
    4_critical_tests.py                 # CPU — Gram-Schmidt residual, interaction direction
    5_projected_rsa.py                  # CPU — RSA on probe-projected activations
    6_category_probes.py                # CPU — 7-way category classification
    7a_report_generator.py              # login — HTML report
    slurm/

  interchange/          # Pipeline C: Interchange Intervention
    1_extract_full_activations.py       # GPU — all tokens × all layers
    2_verb_swap_interventions.py        # GPU — swap verb activations between sentences
    3_subject_swap_interventions.py     # GPU — swap "He" between C1/C4
    4_transfer_analysis.py              # CPU — 6×6 transfer matrix, contrasts, controls
    5a_report_generator.py              # login — HTML report
    slurm/
```

## Implementation Notes

- Extraction: Use forward hooks or `output_hidden_states=True` to capture activations at each layer
- Distance metric: Correlation distance (1 - Pearson r) primary; cosine distance as variant
- Token position: Last token position (RSA), verb/object/period (probes), all tokens (interchange)
- Permutation scheme: Permute condition labels within items (preserves item structure)
- Correction: BH-FDR across layers
- Full stimulus set defined in `code/stimuli.py`

## Relation to Other Experiments

| Experiment | What it shows | What Exp 5 adds |
|---|---|---|
| Exp 2 | Partner identity is linearly decodable and causally active | Exp 5 tests whether structured mental state attribution exists beyond identity |
| Exp 3 | LLMs have concept-level representations of mental properties | Exp 5 tests whether those properties are organized into attribution-bound propositions |
| Exp 4 | LLMs have implicit mind perception geometry | Exp 5 tests whether the Experience/Agency structure appears within mental state attributions |
| Exp 7 | Tests concept deployment during ToM reasoning | Exp 5 tests the underlying attribution structure that ToM reasoning would depend on |

## Key References

- Kriegeskorte et al. (2008). Representational similarity analysis.
- Nili et al. (2014). A toolbox for representational similarity analysis.
- Gray et al. (2007). Dimensions of mind perception (Experience x Agency).
- Zhu et al. (2024). Belief representation probing in LLMs.
