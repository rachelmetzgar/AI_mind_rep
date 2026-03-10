# Exp 3: Top-1 Prompt Pipeline ("_1") + Pending Tasks

## Context

Two workstreams:

**Part A (pending from previous session):** Standalone concept steering (job 3855322) completed for dims 18, 16, 1, 15, 14. Need to run behavioral analysis on those results. Also need to regenerate the contrasts behavioral_summary.csv with FDR correction code that was written but not yet run.

**Part B (new):** Design and implement a "_1" pipeline variant. Instead of using 40 prompts per concept, select the single most representative standalone prompt from each concept. Compute a new kind of contrast: `activation(top_prompt_X) - mean(activation(top_prompt_Y) for all Y≠X)`. This isolates what's unique about each concept's representation. Run the full pipeline: elicitation, alignment, overlap, steering, conversation alignment, behavioral analysis, and HTML reports.

---

## Part A: Pending Tasks

### A1: Standalone Behavioral Analysis
Results exist at `results/llama2_13b_chat/balanced_gpt/concept_steering/v1_standalone/` for dims 14, 15, 16, 18, 1.

```bash
cd /mnt/cup/labs/graziano/rachel/mind_rep/exp_3
module load pyger && conda activate llama2_env
python code/4a_concept_steering_behavior.py --version balanced_gpt --mode standalone
```

This is CPU-only and should run in ~10 min (5 dims × 2 strategies × ~20 measures). Run on login node.

### A2: FDR-Corrected Contrasts Behavioral Summary
FDR code already added to `4a_concept_steering_behavior.py` (multipletests import + FDR block after pairwise tests). Need to regenerate:

```bash
python code/4a_concept_steering_behavior.py --version balanced_gpt --mode contrasts
```

Also CPU-only, ~30 min (more dims). Run on login node.

### A3: Regenerate Steering HTML
After A1 and A2:
```bash
python code/4b_concept_steering_summary_generator.py --version balanced_gpt
```

---

## Part B: Top-1 Prompt Pipeline ("_1")

### Design Overview

**Prompt selection:** For each concept with standalone activations, load existing activations from `concept_activations/standalone/{dim}/concept_activations.npz`. Select the prompt whose activation has highest mean cosine similarity to the concept centroid (averaged over layers 20-40). This is the most geometrically representative prompt.

**Contrast computation:** After selecting 1 top prompt per concept (N concepts total), compute:
- `direction_X = activation(top_X) - mean(activation(top_Y) for all Y ≠ X)`
- This isolates what makes concept X unique relative to other concepts.

**Included concepts:** All standalone dims except sysprompt variants (20-23). That gives dims 1-18, 25-27, 30-32 (~26 concepts).

**Pipeline steps for _1:**
| Step | Script | GPU? | Notes |
|------|--------|------|-------|
| Elicit | `1_compute_top1_vectors.py` (new) | No | Reads existing acts, computes top prompts + contrasts |
| Alignment | `2a_alignment_analysis.py --variant _1` | No | cosine²(concept_vec, probe_weight) |
| Overlap | `2f_concept_overlap.py --variant _1` | No | Pairwise cosine between _1 concept vectors |
| Conv. alignment | `9b_concept_conversation_alignment.py --variant _1` | No | Project conv acts onto _1 vectors |
| Steering | `4_concept_steering_generate.py --variant _1` | Yes | Inject _1 concept directions |
| Behavior | `4a_concept_steering_behavior.py --variant _1` | No | Linguistic analysis of steering output |
| Probes | SKIP | — | N=1, can't train classifier |
| Lexical | SKIP | — | Only 1 prompt, no lexical variation |

---

### B1: New Script — `1_compute_top1_vectors.py`

**Location:** `exp_3/code/1_compute_top1_vectors.py`

**Logic:**
```python
# 1. Scan standalone activations for available dims (exclude sysprompt 20-23)
# 2. For each dim:
#    - Load concept_activations.npz (activations key, shape: n_prompts × 41 × 5120)
#    - Load concept_prompts.json (prompt texts)
#    - Compute centroid = mean(activations, axis=0)  # shape (41, 5120)
#    - For each prompt, compute mean cosine(act[layer], centroid[layer]) for layers 20-40
#    - Select top prompt = argmax(cosine_scores)
#    - Store: {dim_name: {prompt_idx, prompt_text, cosine_score, activation (41, 5120)}}
#
# 3. After all dims processed:
#    - top_activations = {dim: act[top_idx]}  # each (41, 5120)
#    - For each dim X:
#        other_mean = mean(top_activations[Y] for Y != X)
#        direction_X = top_activations[X] - other_mean
#    - Save per dim to concept_activations_1/contrasts/{dim_name}/:
#        concept_vector_per_layer.npz  (concept_direction, norms)
#    - Save per dim to concept_activations_1/standalone/{dim_name}/:
#        mean_vectors_per_layer.npz    (mean_concept = single prompt activation)
#    - Save top_prompt_selections.json at concept_activations_1/
```

**Output structure** (matches existing format for downstream compatibility):
```
results/llama2_13b_chat/concept_activations_1/
├── top_prompt_selections.json
├── contrasts/
│   ├── 1_phenomenology/
│   │   ├── concept_vector_per_layer.npz  (keys: concept_direction, norms)
│   │   └── concept_prompts.json          (single prompt metadata)
│   ├── 2_emotions/
│   │   └── ...
│   └── ...
└── standalone/
    ├── 1_phenomenology/
    │   └── mean_vectors_per_layer.npz    (key: mean_concept)
    ├── 2_emotions/
    │   └── ...
    └── ...
```

**No GPU needed** — just loading existing .npz files and computing means/cosines.

---

### B2: Config Changes — `config.py`

Add variant support:

```python
# Module-level variable
_active_variant = ""

def set_variant(variant):
    """Set active variant suffix (e.g., '_1'). Updates concept activation + output paths."""
    global _active_variant
    _active_variant = variant
    config.RESULTS._update_model_paths()
    if _active_version:
        config.RESULTS._update_version_paths(_active_version)

def add_variant_argument(parser):
    """Add optional --variant argument to argparse parser."""
    parser.add_argument(
        "--variant", type=str, default="",
        help="Variant suffix for concept paths (e.g., '_1' for top-1 pipeline)"
    )
```

In `OutputPaths._update_model_paths()`, change:
```python
self.concept_activations = model_root / f"concept_activations{_active_variant}"
self.concept_activations_contrasts = self.concept_activations / "contrasts"
self.concept_activations_standalone = self.concept_activations / "standalone"
self.concept_overlap = model_root / f"concept_overlap{_active_variant}"
```

In `OutputPaths._update_version_paths()`, change:
```python
self.alignment = ver_root / f"alignment{_active_variant}"
self.alignment_versions = ver_root / f"alignment{_active_variant}"
self.alignment_contrasts = self.alignment / "contrasts"
self.alignment_contrasts_raw = self.alignment / "contrasts" / "raw"
self.alignment_contrasts_residual = self.alignment / "contrasts" / "residual"
self.alignment_standalone = self.alignment / "standalone"
self.alignment_layer_profiles = self.alignment / "layer_profiles"
self.alignment_sysprompt = self.alignment / "sysprompt"
self.concept_steering = ver_root / f"concept_steering{_active_variant}"
```

---

### B3: Downstream Script Changes

Each script gets ~5 lines added. Pattern:
```python
from config import add_variant_argument, set_variant
# In argparse setup:
add_variant_argument(parser)
# After parsing:
if args.variant:
    set_variant(args.variant)
```

Scripts to modify:
1. **`2a_alignment_analysis.py`** — add `--variant`, call `set_variant()` before path access
2. **`2f_concept_overlap.py`** — same
3. **`2g_concept_overlap_standalone.py`** — same
4. **`4_concept_steering_generate.py`** — add `--variant`, update `_init_paths()` to use config paths with variant
5. **`4a_concept_steering_behavior.py`** — add `--variant`, update path logic
6. **`9b_concept_conversation_alignment.py`** — add `--variant`
7. **`2e_cross_dimension_summary_generator.py`** — add `--variant` for reading _1 alignment stats
8. **`2f_concept_overlap_summary_generator.py`** — add `--variant`
9. **`4b_concept_steering_summary_generator.py`** — add `--variant`
10. **`9c_concept_conversation_report.py`** — add `--variant`

---

### B4: SLURM Scripts

**`slurm/top1_elicit.sh`** — CPU-only, ~16G, ~30 min
```bash
python code/1_compute_top1_vectors.py
```

**`slurm/top1_alignment.sh`** — CPU-only, ~32G, ~2h
```bash
# Run alignment for _1 variant, all turns × both versions
VERSION=$VERSION TURN=$TURN python code/2a_alignment_analysis.py \
    --version $VERSION --turn $TURN --variant _1
```

**`slurm/top1_overlap.sh`** — CPU-only, ~16G, ~30 min
```bash
python code/2f_concept_overlap.py --variant _1
```

**`slurm/top1_steering.sh`** — GPU, ~64G, ~6h, array job over dims
```bash
python code/4_concept_steering_generate.py \
    --version $VERSION --dim_id $DIM_ID --variant _1 \
    --strategies exp2_peak upper_half --strengths 4
```

**`slurm/top1_conv_alignment.sh`** — CPU-only, ~32G, ~1h
```bash
python code/9b_concept_conversation_alignment.py \
    --version $VERSION --turn $TURN --variant _1
```

---

### B5: Run Order

1. `1_compute_top1_vectors.py` — CPU, login node OK (~5 min, just npz loading + cosine)
2. `2a` alignment — SLURM, 10 jobs (5 turns × 2 versions)
3. `2f` overlap — SLURM, 1 job
4. `9b` conv alignment — SLURM, 2 jobs (2 versions)
5. `4_steering` — SLURM GPU, array job (~26 dims, or subset of interesting ones)
6. `4a` behavioral — CPU, after steering
7. HTML generators — login node, after all above

### B6: HTML Reports

Generators read from computed stats, not concept vectors. Run each with `--variant _1`:
```bash
python code/2e_cross_dimension_summary_generator.py --variant _1 --turn 5
python code/2f_concept_overlap_summary_generator.py --variant _1
python code/4b_concept_steering_summary_generator.py --version balanced_gpt --variant _1
python code/9c_concept_conversation_report.py --version balanced_gpt --turn 5 --variant _1
```

---

## Execution Order

1. **Part A first** (quick, no code changes needed):
   - A1: Run standalone behavioral analysis
   - A2: Run FDR-corrected contrasts behavioral analysis
   - A3: Regenerate steering HTML

2. **Part B** (code + compute):
   - B1: Write `1_compute_top1_vectors.py`
   - B2: Config changes
   - B3: Add `--variant` to downstream scripts
   - B4: Write SLURM scripts
   - Run B5 pipeline
   - B6: Generate HTML reports

---

## Key Files

| File | Action |
|------|--------|
| `exp_3/code/1_compute_top1_vectors.py` | CREATE — new elicitation script |
| `exp_3/code/config.py` | MODIFY — add variant support |
| `exp_3/code/2a_alignment_analysis.py` | MODIFY — add `--variant` arg |
| `exp_3/code/2f_concept_overlap.py` | MODIFY — add `--variant` arg |
| `exp_3/code/2g_concept_overlap_standalone.py` | MODIFY — add `--variant` arg |
| `exp_3/code/4_concept_steering_generate.py` | MODIFY — add `--variant` arg |
| `exp_3/code/4a_concept_steering_behavior.py` | MODIFY — add `--variant` arg |
| `exp_3/code/9b_concept_conversation_alignment.py` | MODIFY — add `--variant` arg |
| `exp_3/code/2e_cross_dimension_summary_generator.py` | MODIFY — add `--variant` arg |
| `exp_3/code/2f_concept_overlap_summary_generator.py` | MODIFY — add `--variant` arg |
| `exp_3/code/4b_concept_steering_summary_generator.py` | MODIFY — add `--variant` arg |
| `exp_3/code/9c_concept_conversation_report.py` | MODIFY — add `--variant` arg |
| `exp_3/code/slurm/top1_*.sh` | CREATE — 5 new SLURM scripts |

## Verification

1. After B1: Check `concept_activations_1/top_prompt_selections.json` — verify each dim has 1 selected prompt with cosine score
2. After B1: Check `concept_activations_1/contrasts/1_phenomenology/concept_vector_per_layer.npz` exists with correct shape (41, 5120)
3. After alignment: Check `alignment_1/turn_5/contrasts/raw/summary.json` has entries for _1 dims
4. After steering: Check `concept_steering_1/v1/` has per-dim result directories
5. After behavior: Check behavioral_summary.csv in `concept_steering_1/v1/` has FDR columns
6. After HTML: Open reports, verify _1 data appears with correct dim names
