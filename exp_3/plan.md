# Plan: Concept-Conversation Alignment Analysis (Exp 3, Phase 8)

## Context

Current exp_3 alignment analysis (2a) compares standalone concept vectors to **probe weights** — a learned contrastive representation. The interpretation is muddy: "this concept aligns with the direction a classifier uses to distinguish human from AI." We want a more direct question: **"Is this concept more present/active in human conversations than AI conversations?"**

New approach: compare standalone concept activation vectors directly to conversation activation vectors. For each concept, compute alignment with every conversation, yielding a distribution (~500+ per condition). Then compare human vs AI distributions statistically.

Three sub-approaches handle the concern that 40 prompts may encode linguistic features beyond the concept:
- **A (full prompt set)**: Mean of 40 prompts → align to each conversation. Baseline.
- **C (concept-contrastive)**: Subtract mean of other concepts to isolate what's unique. Controls for shared structure.
- **D (prompt-level)**: Don't average — compute per-prompt alignment, treat prompt as random effect. Separates prompt-level noise from concept-level signal.
- **B (controlled prompts)**: Linguistically matched prompts. Follow-up after we see results from A/C/D.

## Prerequisites & Existing Data

- **Standalone concept activations** — EXIST at `exp_3/data/concept_activations/standalone/{dim}/concept_activations.npz`, shape `(40, 41, 5120)` per concept, 24 dimensions
- **Mean concept vectors** — EXIST at `exp_3/data/concept_activations/standalone/{dim}/mean_vectors_per_layer.npz`, shape `(41, 5120)`
- **Conversation data** — EXIST as per-subject CSVs in `exp_1/versions/{version}/data/meta-llama-Llama-2-13b-chat-hf/0.8/s*.csv`
- **Conversation activations** — DO NOT EXIST. Must be extracted (GPU job).

## Implementation: 3 New Scripts + 2 SLURM Wrappers

**NOTE:** All paths below are pre-restructure. Update after project restructure is complete.

### Step 1: `8a_extract_conversation_activations.py` (GPU)

Extracts and caches conversation activations from LLaMA-2-13B-Chat. Reuses `exp_2/code/src/dataset.py` logic for prompt formatting (`llama_v2_prompt`) and activation extraction.

**Token position:** Operational (last token of formatted conversation, no suffix). This captures the general conversation state rather than a specific probing context. We can add metacognitive position later if needed.

**Args:**
- `--version` (required, e.g. `balanced_gpt`)
- `--turn` (default 5)

**Process:**
1. Load model + tokenizer (same as probe training)
2. For each subject CSV, for each conversation row:
   - Format conversation up to specified turn via `llama_v2_prompt()`
   - Forward pass, extract last-token hidden states from all layers
   - Record condition (human/ai), subject, topic as metadata
3. Save to disk

**Output:**
```
exp_3/data/conversation_activations/{version}/turn_{turn}/
    activations.npz          # shape (n_conversations, 41, 5120), float16 to save space
    metadata.csv             # columns: conv_idx, condition, subject, topic
```

**Size estimate:** ~1000 convs × 41 × 5120 × 2 bytes (float16) ≈ 420 MB per condition, ~840 MB total. Manageable.

**SLURM:** GPU job, `--gres=gpu:1 --mem=48G --time=6:00:00`

### Step 2: `8b_concept_conversation_alignment.py` (CPU)

Core analysis. Computes alignment between standalone concept vectors and conversation activations, with all three sub-approaches.

**Args:**
- `--version` (required)
- `--turn` (default 5)
- `--approaches` (default: all of `a c d`)
- `--dim_ids` (optional filter)
- `--n_bootstrap` (default 1000, for CIs on condition difference)

**Process per approach:**

**Approach A — Full prompt-set mean:**
1. Load mean concept vector per layer: `mean_vectors_per_layer.npz["mean_concept"]` → `(41, 5120)`
2. Load conversation activations → `(n_convs, 41, 5120)`
3. For each conversation, each layer (6-40): `cosine(concept_mean[layer], conv_act[layer])`
4. Average cosine across layers → one scalar per conversation per concept
5. Split by condition, t-test(human_cosines, ai_cosines) per concept
6. Bootstrap the mean difference for CIs
7. FDR correction across concepts

**Approach C — Concept-contrastive:**
1. For each concept k, compute: `contrastive_k[layer] = concept_k[layer] - mean(all_other_concepts[layer])`
2. Unit-normalize the contrastive vector
3. Same alignment + stats as approach A but using contrastive vectors

**Approach D — Prompt-level:**
1. Load per-prompt activations: `concept_activations.npz["activations"]` → `(40, 41, 5120)`
2. For each prompt p, each conversation c, each layer: `cosine(prompt_p[layer], conv_c[layer])`
3. Average across layers → `(40, n_convs)` per concept
4. Average across prompts per conversation → one scalar (compare to approach A which averages prompts before cosine)
5. Also: mixed-effects model `alignment ~ condition + (1|prompt)` using the full (40 × n_convs) matrix
6. Report both the simple and mixed-effects results

**Output per approach:**
```
results/concept_conversation/versions/{version}/turn_{turn}/
    approach_a/
        alignment_scores.npz    # per-concept arrays of per-conv cosines + layer-resolved
        stats.csv               # concept, condition_means, diff, t, p, p_fdr, cohen_d
        figures/
            alignment_by_concept.png        # bar chart: H-A diff per concept
            alignment_distributions.png     # violin/box per concept × condition
    approach_c/
        (same structure)
    approach_d/
        (same + mixed_effects_results.csv)
    cross_approach_summary.csv   # side-by-side comparison of A vs C vs D
```

**SLURM:** CPU job, `--mem=32G --time=3:00:00` (loading ~1.7 GB activations + 24 concept files + bootstrap)

### Step 3: `8c_concept_conversation_report.py` (login node)

Generates HTML/MD report from saved stats and figures. Lightweight, runs from CSVs.

**Sections:**
- Overview table: per-concept H-vs-A difference, significance, effect size
- Approach comparison: does A, C, and D agree? Where do they diverge?
- Layer-resolved profiles: which layers drive the effect?
- Category-level summary: mental vs physical vs pragmatic vs controls
- Controls check: shapes/biological should show weaker effects than mental concepts

## Key Reuse

| Source | What to reuse |
|--------|--------------|
| `exp_2/code/src/dataset.py` | `TextDatasetCSV._extract_activations()`, `llama_v2_prompt()` for conversation formatting |
| `exp_3/data/concept_activations/standalone/` | Pre-computed concept activations (40 prompts × 41 layers × 5120) |
| `exp_3/code/config.py` | Paths, layer constants, version management, `DIMENSION_CATEGORIES` for grouping |
| `exp_3/code/analysis/alignment/2a_alignment_analysis.py` | Bootstrap CI computation pattern, layer filtering (layers 6-40) |

## Execution Order

```
1. sbatch 8a (GPU) — extract conversation activations (~2-4 hours)
2. sbatch 8b (CPU, depends on 8a) — compute alignment + stats (~1-2 hours)
3. python 8c (login node) — generate report (~1 min)
```

## Verification

1. `conversation_activations/{version}/turn_5/activations.npz` exists with expected shape
2. `stats.csv` for each approach has 24 rows (one per concept dimension)
3. Approaches A and D give similar (but not identical) results — D averages in cosine space, A in activation space
4. Control concepts (15_shapes, 14_biological) show weaker H-vs-A differences than mental concepts (1_phenomenology, 2_emotions, etc.)
5. Cross-approach summary shows convergence or informative divergence

## Design Discussion (for reference)

**Why this approach vs current probe-based alignment:**
- Probes are compressed/denoised (trained to find the H-vs-A direction), so probe-based alignment has better signal-to-noise
- But probe-based alignment has confusing interpretation ("concept aligns with learned classifier direction")
- Direct conversation comparison is noisier but more interpretable ("concept pattern is more present in human conversations")
- With N=500+ conversations per condition, statistical power should overcome noise
- The biggest risk is linguistic confounds (shared syntax/style rather than shared concepts) — approaches C and D address this
- If direct comparison works, it's a stronger result; if too noisy, that itself is informative about representational structure
