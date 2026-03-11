# Experiment 5 — Mental State Attribution RSA

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

## Analysis Pipeline

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

## Implementation Notes

- Extraction: Use forward hooks or `output_hidden_states=True` to capture activations at each layer
- Distance metric: Correlation distance (1 - Pearson r) primary; cosine distance as variant
- Token position: Last token position
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
