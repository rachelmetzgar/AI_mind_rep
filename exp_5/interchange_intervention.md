# Mental State Attribution: Interchange Intervention Analysis

## Overview

This analysis tests whether LLaMA-2-13B-Chat uses a DIFFERENT binding mechanism for mental state attributions (He + mental verb + object) compared to action sentences (He + action verb + object). The method is adapted from the entity binding literature (Feng & Steinhardt, ICLR 2024; Gur-Arieh et al., 2025).

The core idea: swap verb activations between sentences and measure how well the swap transfers. If mental state verbs are bound differently from action verbs, within-condition swaps (mental→mental, action→action) should work better than cross-condition swaps (mental→action, action→mental). This would be direct evidence of a dedicated attribution binding mechanism.

## Model and Data

- **Model:** LLaMA-2-13B-Chat
- **Stimuli:** 56 items × 6 conditions = 336 sentences from Experiment 5
- **Conditions:**
  - C1: mental_state — "He notices the crack."
  - C2: dis_mental — "Notice the crack."
  - C3: scr_mental — "The crack to notice."
  - C4: action — "He fills the crack."
  - C5: dis_action — "Fill the crack."
  - C6: scr_action — "The crack to fill."

## Step 0: Activation Extraction

For each of the 336 sentences:

1. Tokenize the sentence
2. Run a forward pass through the model
3. Save the FULL residual stream activations at EVERY token position and EVERY layer
   - Shape per sentence: (num_tokens, num_layers, hidden_dim)
   - For LLaMA-2-13B: hidden_dim = 5120, num_layers = 41 (including embedding layer 0)
4. Also save a token-position map for each sentence identifying:
   - verb_token_idx: the token index of the verb (or last subword token if multi-token)
   - object_token_idx: the token index of the last content word before the period
   - subject_token_idx: token index of "He" (only for C1 and C4, None for others)
   - period_token_idx: token index of "."

**Token position identification per condition:**
- C1 ("He notices the crack."): subject=0, verb=1, object varies, period=last
- C2 ("Notice the crack."): subject=None, verb=0, object varies, period=last
- C3 ("The crack to notice."): subject=None, verb=second-to-last (before period), object=1
- C4 ("He fills the crack."): subject=0, verb=1, object varies, period=last
- C5 ("Fill the crack."): subject=None, verb=0, object varies, period=last
- C6 ("The crack to fill."): subject=None, verb=second-to-last (before period), object=1

IMPORTANT: Tokenize each sentence individually and verify indices. Multi-subword verbs like "distinguishes" or "photographs" should use the LAST subword token index.

Save all activations to disk. They will be reused many times.

## Step 1: Verb Swap Interchange Interventions

### Procedure

For each pair of sentences (A, B):
    For each layer L in range(0, 41):
        1. Load the saved activations for sentence A: acts_A (shape: num_tokens_A × 41 × 5120)
        2. Load the saved activations for sentence B: acts_B (shape: num_tokens_B × 41 × 5120)
        3. Create intervened activations for sentence A:
           - Copy acts_A
           - At layer L, at the verb_token_idx of sentence A, REPLACE the activation with
             the activation from sentence B at layer L at the verb_token_idx of sentence B
           - acts_A_intervened[verb_idx_A, L, :] = acts_B[verb_idx_B, L, :]
        4. Run the model forward FROM layer L using acts_A_intervened
           - This means: take acts_A_intervened at layer L, pass through transformer layers L+1, L+2, ... 40
           - Collect the final representation at the last token position (period) at the final layer
        5. Save three representations:
           - rep_A_original: sentence A's original final representation (period token, last layer)
           - rep_B_original: sentence B's original final representation (period token, last layer)
           - rep_A_intervened: sentence A's representation after the verb swap

**Implementation note on running from a specific layer:**

The cleanest way to do this is:
1. Run sentence A normally, save activations at ALL layers and ALL token positions
2. Modify the activation at (verb_token_idx, layer_L) 
3. Re-run the forward pass from layer L onward with the modified activation

In practice with HuggingFace, you can do this with forward hooks:
```python
def make_hook(layer_idx, token_idx, replacement_vector):
    def hook_fn(module, input, output):
        # output is typically (hidden_states, ...) 
        # Modify the hidden states at the target token position
        hidden_states = output[0]
        hidden_states[0, token_idx, :] = replacement_vector
        return (hidden_states,) + output[1:]
    return hook_fn
```

Register the hook on the target layer's transformer block, run the full forward pass, and the hook will inject the replacement at the right point. The model will then process the modified activation through all subsequent layers naturally.

### Measuring Swap Success

For each swap (A, B, layer L), compute:

```python
import torch.nn.functional as F

# Cosine similarities
sim_to_B = F.cosine_similarity(rep_A_intervened, rep_B_original, dim=0)
sim_to_A = F.cosine_similarity(rep_A_intervened, rep_A_original, dim=0)

# Swap success: how much did the representation shift toward B and away from A?
# Range: -1 (moved away from B) to +1 (perfectly became B)
swap_success = (sim_to_B - sim_to_A).item()
```

Alternative metric (normalized):
```python
# Proportion of the way from A to B
# 0 = still at A, 1 = fully at B, >1 = overshot
shift = (sim_to_B - sim_to_A) / (sim_to_A_B_original_distance)
```

Where sim_to_A_B_original_distance = 1 - F.cosine_similarity(rep_A_original, rep_B_original, dim=0)

### Which Pairs to Swap

You don't need to swap every possible pair of 336 sentences (that's 112,000+ pairs). Instead, structure the swaps to test specific hypotheses.

**Within-condition swaps (diagonal of the transfer matrix):**
For each condition (C1 through C6):
    For each pair of items (i, j) where i < j within the same condition:
        Swap verb of item j into item i
        Swap verb of item i into item j
    Total: 56 choose 2 = 1,540 pairs per condition, but this is too many.
    SUBSAMPLE: For each item i, swap with 10 randomly selected other items j.
    Total: 56 × 10 = 560 swaps per condition.

**Cross-condition swaps (off-diagonal):**
For each pair of conditions (source_cond, target_cond):
    For each item i:
        Swap the verb from source_cond item i into target_cond item i (SAME item, different condition)
        This controls for the object — same object noun in both conditions
    Total: 56 swaps per condition pair.

ALSO do mismatched items:
    For each item i:
        Swap the verb from source_cond item j (j ≠ i, randomly selected) into target_cond item i
        This tests whether the swap works even when the object differs
    Total: 56 swaps per condition pair (one random j per i).

### The 6×6 Transfer Matrix

For each pair of conditions (source, target), average the swap_success across all swaps at each layer. This gives a 6×6 matrix per layer.

```
transfer_matrix[source_cond, target_cond, layer] = mean swap_success
```

**Finding the best layer:** For each condition pair, the swap success will vary by layer. Start by finding the layer where within-C1 swaps work best. Use that layer (or a small range of layers) for the main analysis.

## Step 2: Analyze the Transfer Matrix

### 2a. Visualize the 6×6 Matrix

At the best layer, plot the 6×6 transfer matrix as a heatmap. Rows = source condition, columns = target condition. Color = mean swap success.

Label the axes:
```
C1: He+mental+obj
C2: mental+obj
C3: obj+mental (scrambled)
C4: He+action+obj
C5: action+obj
C6: obj+action (scrambled)
```

### 2b. Key Comparisons

Compute the following contrasts and test significance:

**Test 1: Within-mental vs cross-type (verb binding specificity)**
- within_mental = mean(C1→C1)
- cross_to_action = mean(C1→C4) 
- Difference = within_mental - cross_to_action
- If positive: mental verbs swap better into mental contexts than action contexts
- Significance: permutation test — shuffle condition labels 10,000 times, recompute difference, get p-value

**Test 2: Within-action vs cross-type**
- within_action = mean(C4→C4)
- cross_to_mental = mean(C4→C1)
- Same logic as Test 1 for the action side

**Test 3: C1→C1 vs C1→C2 (does the subject matter for binding?)**
- If C1→C1 > C1→C2: the verb binding in a full attribution (with subject) is different from the verb binding in a disembodied sentence
- If C1→C1 ≈ C1→C2: the subject doesn't change how the verb is bound — verb binding is the same with or without "He"

**Test 4: C1→C2 vs C1→C4 (mental verb presence vs subject presence)**
- C1→C2: swap mental verb into context that has same verb type but no subject
- C1→C4: swap mental verb into context that has subject but different verb type
- If C1→C2 > C1→C4: verb TYPE matters more than subject presence for binding compatibility
- If C1→C4 > C1→C2: subject presence matters more — the binding mechanism cares about syntactic frame

**Test 5: The attribution-specific test**
- (C1→C1 minus C1→C2) vs (C4→C4 minus C4→C5)
- This is the INTERACTION: does removing the subject hurt swap success MORE for mental state sentences than for action sentences?
- C1→C1 minus C1→C2 = how much the subject matters for mental verb binding
- C4→C4 minus C4→C5 = how much the subject matters for action verb binding
- If the mental difference is LARGER than the action difference: the subject is more important for mental state binding than for action binding — evidence of attribution-specific binding
- Significance: permutation test on the interaction term

### 2c. Block Structure Analysis

Test whether the transfer matrix has a 2×2 block structure (mental vs action):

```python
# Mental block: C1,C2,C3 source/target pairs
mental_block = transfer_matrix[0:3, 0:3].mean()

# Action block: C4,C5,C6 source/target pairs
action_block = transfer_matrix[3:6, 3:6].mean()

# Cross blocks
cross_block = transfer_matrix[0:3, 3:6].mean()  # mental→action
cross_block2 = transfer_matrix[3:6, 0:3].mean()  # action→mental

# Within-type mean
within_type = (mental_block + action_block) / 2

# Cross-type mean
cross_type = (cross_block + cross_block2) / 2

# Block structure index
block_index = within_type - cross_type
```

If block_index > 0 and significant (permutation test): swaps work better within verb type than across. This means mental and action verbs are bound in representationally distinct ways.

BUT this alone doesn't prove attribution-specific binding — it could just be verb semantic similarity (mental verbs are more similar to other mental verbs than to action verbs in embedding space, so swaps transfer better).

### 2d. Control for Verb Semantic Similarity

To rule out the verb-similarity confound:

1. Get the static embeddings (layer 0) of all 56 mental verbs and 56 action verbs
2. Compute pairwise cosine similarity between all verb pairs
3. For each swap pair, record the embedding similarity of the two verbs being swapped
4. Run a regression:
   ```
   swap_success ~ verb_embedding_similarity + same_condition + same_verb_type
   ```
5. The coefficient on same_condition (controlling for embedding similarity) tells you: do within-condition swaps work better BEYOND what verb similarity predicts?

If same_condition is significant after controlling for verb similarity: the binding mechanism is condition-specific, not just driven by how similar the verbs are in meaning.

## Step 3: Subject Token Swaps

This is the most direct test of attribution-specific binding. Swap the "He" activation between C1 and C4 sentences.

### Procedure

For each item i:
    For each layer L:
        1. Run C1 sentence: "He [mental_verb] the [object]."
        2. Run C4 sentence: "He [action_verb] the [object]."
        3. Both have "He" at token index 0.
        4. Swap: replace "He" activation in C1 at layer L with "He" activation from C4 at layer L
        5. Also do the reverse: replace "He" in C4 with "He" from C1
        6. Measure: how much does the representation change?

### Measuring the Effect of Subject Swaps

Since "He" is the same word in both sentences, any difference in the "He" activation between C1 and C4 is due to what information has been written back to the subject position by attention from subsequent tokens (verb, object). This only happens at layers where attention has had a chance to propagate information backward.

```python
# After swapping "He" from C4 into C1:
rep_C1_with_C4_subject = get_final_representation(C1_intervened)
rep_C1_original = get_final_representation(C1_original)

# How much did the swap change the representation?
subject_swap_effect = 1 - F.cosine_similarity(rep_C1_with_C4_subject, rep_C1_original, dim=0)
```

### Key Comparison

Compute subject_swap_effect for EVERY item at each layer.

Also compute a CONTROL: swap "He" between two C1 sentences (same condition, different items). This tells you how much "He" varies within C1.

```
cross_type_effect = mean(subject_swap_effect for C1↔C4 swaps)
within_type_effect = mean(subject_swap_effect for C1↔C1 swaps)
```

If cross_type_effect > within_type_effect: the "He" in mental state sentences carries different information than the "He" in action sentences. The model is writing verb-type-specific information back to the subject position — evidence that the subject is bound differently depending on whether it's part of a mental state attribution or an action.

Layer profile: this effect should be absent at early layers (before attention propagates verb info to subject position) and emerge at mid-to-late layers.

## Step 4: Layer Profile and Summary

### 4a. Plot Layer Profiles

For each key metric, plot as a function of layer:

1. Within-C1 swap success (verb swaps)
2. Within-C4 swap success (verb swaps)
3. C1→C4 swap success (cross-type verb swaps)
4. C4→C1 swap success (cross-type verb swaps)
5. Block structure index (within-type minus cross-type)
6. Subject swap effect (cross-type vs within-type)
7. Attribution interaction (Test 5 from Step 2b)

### 4b. Statistical Summary Table

At the peak layer, report:

| Comparison | Value | p-value | Interpretation |
|---|---|---|---|
| C1→C1 swap success | | | Mental verb swaps within attribution |
| C4→C4 swap success | | | Action verb swaps within action |
| C1→C4 swap success | | | Mental verb into action context |
| C4→C1 swap success | | | Action verb into mental context |
| Within minus Cross (block index) | | | Verb type binding specificity |
| C1→C1 minus C1→C2 | | | Subject contribution to mental binding |
| C4→C4 minus C4→C5 | | | Subject contribution to action binding |
| Interaction (Test 5) | | | Attribution-specific subject binding |
| Subject swap effect (C1↔C4 minus C1↔C1) | | | Subject carries verb-type-specific info |

### 4c. The 6×6 Heatmap

At the peak layer, display the full 6×6 transfer matrix with annotations showing which comparisons are significant.

## Expected Results Under Different Hypotheses

**If mental state attribution has dedicated binding:**
- C1→C1 > C1→C4 (mental verbs swap better into mental contexts)
- C4→C4 > C4→C1 (action verbs swap better into action contexts)
- Block structure index is positive and significant
- The interaction (Test 5) is significant — subject matters MORE for mental binding
- Subject swap effect is larger for C1↔C4 than C1↔C1 at mid-late layers
- The 6×6 matrix shows clear block structure with C1 potentially forming its own sub-block

**If binding is generic (no attribution-specific mechanism):**
- All swap successes are roughly equal across conditions
- Block structure index near zero
- Subject swap has no cross-condition effect beyond within-condition
- The 6×6 matrix is roughly uniform

**If the difference is just verb semantics (not binding):**
- Block structure exists (mental vs action blocks) BUT
- The regression in Step 2d shows same_condition is NOT significant after controlling for verb embedding similarity
- Verbs that are semantically similar swap better, regardless of condition membership

## Implementation Notes

- Model: LLaMA-2-13B-Chat, HuggingFace Transformers
- Use forward hooks to inject replacement activations at specific token positions and layers
- For efficiency: don't recompute the full forward pass for every swap. Cache activations from the original forward pass and only recompute from the intervention layer onward
- For the 6×6 transfer matrix with 56 items: you need 56 × (number of swap partners) × 30 condition pairs × 41 layers interventions. This is computationally expensive. Start with a subset:
  - Pick 5-10 layers spanning early/mid/late (e.g., layers 5, 10, 15, 20, 25, 30, 35, 40)
  - Run the full analysis at those layers first
  - Find the peak, then fill in surrounding layers at full resolution
- For verb swaps across conditions of the SAME item: both sentences share the same object, so the swap is cleaner. Prioritize same-item swaps.
- For the subject swap (Step 3): this only applies to C1 and C4 (the conditions with "He"). All 56 items can be tested.
- Permutation tests: 10,000 iterations for final results, 1,000 for initial exploration
- Save all swap_success values (not just means) for later analysis and visualization