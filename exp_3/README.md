# Experiment 3: Concept-of-Mind Representations

**Author**: Rachel C. Metzgar, Princeton University
**Last Updated**: March 2026

---

## Overview

This pipeline investigates whether large language models (LLMs) form internal representations of "what human and AI minds are like" and whether these concept-of-mind representations align with and influence conversational behavior.

**Key Question**: When an LLM adjusts its communication style based on partner identity (human vs AI), is it drawing on its general semantic knowledge about what humans and AIs are, and how exactly is it distinguishing them?

**Approach**:
1. Extract the model's internal representations of mental concepts (phenomenology, emotions, agency, etc.)
2. Test alignment with Exp 2 conversational partner-identity representations
3. Causally intervene by injecting concept vectors into generation
4. Analyze behavioral effects and cross-domain generalization

---

## How Concept Vectors Are Made

The pipeline extracts concept vectors from LLaMA-2-13B-Chat by presenting it with concept-related prompts and recording residual-stream activations at the last token across all 41 layers, yielding vectors of shape `(41, 5120)`.

**Two prompt modes** define how vectors are computed from these activations:

- **Contrasts** (`concepts/contrasts/`): Each concept has 40 human-framed and 40 AI-framed prompts (e.g., "Think about how a human experiences emotions" vs "Think about how an AI experiences emotions"). The concept direction is `mean(human_prompts) - mean(AI_prompts)` per layer — a vector pointing from "AI version of this concept" toward "human version."
- **Standalone** (`concepts/standalone/`): Each concept has 40 prompts with no entity framing (e.g., "Imagine what it is like to see the color red for the first time"). The concept vector is `mean(all_prompts)` per layer — an undirected activation centroid.

Contrasts isolate what the model represents as *different* about humans vs AIs on each concept. Standalone captures the concept's *general* activation pattern, independent of any entity comparison.

### Variant Approaches

Four variants of concept vector construction test robustness to methodological choices. All share the same output directories; files are distinguished by **filename suffixes** managed through `config.py`'s variant system (`--variant` flag → `variant_filename()`).

| Variant | Flag | Suffix | Description |
|---------|------|--------|-------------|
| **Full 40-prompt** | *(default)* | *(none)* | All 40 prompts averaged. Maximum statistical power. |
| **Top-1 aligned** | `_1` | `_top_align` | Single most representative prompt per concept (highest cosine to centroid, layers 20-40). Contrast computed as `top_X - mean(top_Y for Y≠X)`. Tests whether averaging creates lexical artifacts. |
| **Simple/syntactic** | `_simple` | `_simple` | 153 concepts × 1 template: "Think about what it is like to have [X]." Grouped into 10 categories. Syntactic control (sort of). |
| **Other-focused** | `_other` | `_other` | Same 25 standalone dims rewritten with "someone" as subject. Tests whether alignment depends on self vs other perspective. |

---

## Alignment Analyses

### Probe Weight Alignment (Phase 2a)

The core question: do concept vectors point in the same direction as the Exp 2 partner-identity probe weights? Measured as cosine similarity (reported as R² = cosine²) between concept vectors and probe weight vectors at each layer.

Three sub-analyses address different confound concerns:

| Analysis | Input | Method | Question |
|----------|-------|--------|----------|
| **Raw** | Contrast directions (H − AI) | Direct cosine with probe weights | Is the concept direction aligned with the identity probe? |
| **Residual** | Contrast directions minus baseline (dim 0) projection | Project out entity/subject framing, then cosine | Does alignment survive removing generic "human vs AI" framing? |
| **Standalone** | Mean concept vectors (no entity framing) | Cosine with probe weights | Do entity-free concept activations still align? |

All three use bootstrap resampling (1000 iterations) for 95% CIs. Results are saved per version per turn: `results/{model}/{version}/alignment/turn_{turn}/{contrasts/{raw,residual},standalone}/`.

### Conversation Activation Alignment (Phase 9)

A complementary analysis that tests whether concept vectors are align during real conversations, rather than just geometrically aligned with probe weights.

1. **Phase 9a**: Extract activations from actual Exp 1 conversations (human-directed vs AI-directed)
2. **Phase 9b**: Compute cosine between each concept vector and each conversation's activation, then compare human vs AI conversations (t-test, effect size)

| Aspect | Probe Alignment (Phase 2a) | Conversation Alignment (Phase 9) |
|--------|---------------------------|----------------------------------|
| **Compares** | Concept vectors ↔ probe **weights** | Concept vectors ↔ conversation **activations** |
| **Tests** | Is the geometry aligned? | Are concepts actually present in conversations? |
| **Granularity** | One value per layer per concept | Per-conversation scores, H vs A comparison |

---

## Lexical Confound Analysis

**The concern**: Concept prompts might share vocabulary with probe training conversations, creating alignment through surface-level word overlap rather than genuine conceptual structure.

**Phase 8b** (`8b_cross_dataset_lexical_overlap.py`) is the primary analysis. It builds a word-bias dictionary from Exp 1 conversations (each word scored by how much more it appears in human vs AI conversations), then correlates concept prompts' mean word-bias with their alignment R².

**Key findings**:
- **Contrast prompts**: Significant positive correlation (ρ=+0.61, p=0.001) — lexical confound is plausible for contrasts, since human-labeled prompts naturally use words that appear more in human conversations.
- **Standalone prompts**: Correlation is *negative* (ρ=−0.44, p=0.018) — goes the *opposite* direction from confound prediction. This is the strongest defense against the lexical artifact account.
- **Control concepts** (shapes, granite, squares, horizontal): Same prompt structure as mental concepts but ~10× lower alignment (~0.0002 vs ~0.0022), arguing for content-specificity.
- **Residual analysis**: Projecting out the entity baseline direction drops 60-75% of alignment, but 25-40% survives for mental concepts — legitimate signal beyond entity framing.

---

## Concept Steering (Phase 4)

Tests causal relevance by injecting concept vectors into the model's residual stream during generation:

```
h'[layer, last_token] = h[layer, last_token] + sign × strength × unit_direction[layer]
```

where `sign = +1` (steer toward "human") or `-1` (steer toward "AI"), and `strength ∈ {1, 2, 4, 8}`.

**Three layer selection strategies** determine which layers receive the injection:
- **exp2_peak**: Top 15 layers by Exp 2 metacognitive probe accuracy
- **upper_half**: Layers 20-40 (simple deep-layer heuristic)
- **concept_aligned**: Top 15 layers by |cosine| between concept vector and probe weight (per-concept)

**Two steering modes**: `v1` uses contrast directions; `v1_standalone` uses standalone mean vectors.

---

## Data Versions

Analyses that interact with Exp 2 data (alignment, cross-prediction) require a `--version` flag specifying which Exp 2 data version to use. Results are saved to version-specific subdirectories under `results/{model}/{version}/`.

| Version | Description |
|---------|-------------|
| `labels` | AI/human labels instead of names |
| `balanced_names` | Balanced gender names |
| `balanced_gpt` | Balanced names + GPT-4 replacement |
| `names` | Original Sam/Casey names (deprecated) |
| `nonsense_codeword` | Tokens framed as arbitrary session code |
| `nonsense_ignore` | Tokens framed with "ignore" instruction |

---

## Quick Start

### Environment Setup
```bash
module load pyger
conda activate llama2_env   # For all phases except behavioral analysis
conda activate behavior_env # For phase 6 (behavioral analysis)
```

### Run Full Pipeline
```bash
# Phase 1: Extract concept activations (contrasts mode)
sbatch code/slurm/1_elicit_contrasts.sh

# Phase 2: Alignment analysis
sbatch --export=VERSION=balanced_gpt code/slurm/2a_alignment_analysis.sh

# Phase 3: Train concept probes
python code/3_train_concept_probes.py --version balanced_gpt --dim_id 1

# Phase 3a-3b: Probe alignment stats + figures
sbatch --export=VERSION=balanced_gpt code/slurm/3a_contrast_pipeline.sh

# Phase 4: Concept steering (V1 mode)
sbatch code/slurm/4a_concept_steering_v1.sh

# Phase 5: Concept intervention
sbatch code/slurm/5a_concept_intervention.sh

# Phase 6: Behavioral analysis
sbatch code/slurm/6_behavior_analysis.sh

# Phase 7: Cross-prediction
python code/7_cross_prediction.py --version balanced_gpt
```

---

## Directory Structure

```
exp_3/
├── README.md
├── code/
│   ├── config.py                              # Central configuration
│   ├── utils/                                 # Shared utilities
│   │   ├── dataset.py                         #   Dataset loaders
│   │   ├── probes.py                          #   Probe architectures
│   │   ├── losses.py                          #   Loss functions
│   │   ├── train_test_utils.py                #   Training loops
│   │   ├── intervention_utils.py              #   Steering functions
│   │   └── prompt_utils.py                    #   Prompt templates
│   ├── slurm/                                 # SLURM batch scripts
│   ├── causality_questions.txt                # V1 test questions
│   │
│   ├── 1_elicit_concept_vectors.py            # Phase 1: Extract concept activations
│   ├── 1_compute_top1_vectors.py              #          Top-1 aligned variant
│   ├── 1_elicit_simple_vectors.py             #          Simple/syntactic-control variant
│   ├── 2a_alignment_analysis.py               # Phase 2: Core alignment (raw/residual/standalone)
│   ├── 2b_layer_profile_analysis.py           #          Layer profiles
│   ├── 2c_elicit_sysprompt_vectors.py         #          System prompt vectors
│   ├── 2d_sysprompt_alignment.py              #          System prompt alignment
│   ├── 2e_cross_dimension_summary_generator.py #         Cross-dimension summary
│   ├── 2f_concept_overlap.py                  #          Concept overlap (contrasts)
│   ├── 2f_concept_overlap_summary_generator.py
│   ├── 2g_concept_overlap_standalone.py       #          Concept overlap (standalone)
│   ├── 2g_concept_overlap_standalone_summary_generator.py
│   ├── 2h_concept_aligned_layers.py           #          Pre-compute aligned layers
│   ├── 2i_alignment_comparison_summary_generator.py #    Cross-version comparisons
│   ├── 2j_raw_comparison_summary_generator.py
│   ├── 2k_pairwise_tests.py
│   ├── 2k_pairwise_summary_generator.py
│   ├── 3_train_concept_probes.py              # Phase 3: Train probes on concepts
│   ├── 3a_compute_alignment_stats.py          #          Probe alignment stats
│   ├── 3b_alignment_figures_summary_generator.py #       Probe alignment figures
│   ├── 4_concept_steering_generate.py         # Phase 4: Steer generation
│   ├── 4a_concept_steering_behavior.py        #          Steering behavioral analysis
│   ├── 4b_concept_steering_summary_generator.py #        Steering figures/report
│   ├── 4c_shapes_flip_test.py                 #          Flipped shapes control (dim 29)
│   ├── 5_concept_intervention.py              # Phase 5: Causal intervention
│   ├── 6_behavior_analysis.py                 # Phase 6: Linguistic feature analysis
│   ├── 7_cross_prediction.py                  # Phase 7: Cross-domain probe evaluation
│   ├── 8_lexical_distinctiveness.py           # Phase 8: Lexical confound analysis
│   ├── 8a_lexical_summary_generator.py        #          Lexical report
│   ├── 8b_cross_dataset_lexical_overlap.py    #          Word-bias correlation analysis
│   ├── 9a_extract_conversation_activations.py # Phase 9: Conversation alignment
│   ├── 9b_concept_conversation_alignment.py   #          Concept–conversation cosines
│   └── 9c_concept_conversation_report.py      #          Conversation alignment report
│
├── concepts/                                  # Concept prompt definitions
│   ├── contrasts/                             #   Human vs AI paired prompts
│   ├── standalone/                            #   Concept-only prompts (self-focused)
│   └── other/standalone/                      #   Other-focused prompts ("someone")
│
├── data/                                      # Reserved for external/input data
│
├── results/
│   └── llama2_13b_chat/                       # Model-scoped results
│       ├── concept_activations/               #   Extracted activations (Phase 1)
│       │   ├── contrasts/                     #     Human-AI contrast prompts
│       │   └── standalone/                    #     Concept-only prompts
│       ├── concept_probes/                    #   Trained concept probes (Phase 3)
│       ├── {version}/                         #   Version-specific
│       │   ├── alignment/                     #     Concept-probe alignment (Phase 2)
│       │   │   ├── contrasts/{raw,residual}/  #       Raw & residual contrast alignment
│       │   │   └── standalone/                #       Standalone alignment
│       │   ├── concept_steering/              #     Steering outputs (Phase 4)
│       │   ├── concept_conversation/          #     Conversation activation alignment (Phase 9)
│       │   ├── interventions/                 #     Causal intervention results (Phase 5)
│       │   ├── behavioral/                    #     Behavioral analysis (Phase 6)
│       │   └── cross_prediction/              #     Cross-domain results (Phase 7)
│       ├── comparisons/                       #   Cross-version comparisons
│       │   └── alignment/
│       ├── concept_overlap/                   #   Concept overlap analysis
│       ├── sysprompt/                         #   System prompt analysis
│       └── lexical/                           #   Lexical overlap analysis
│
├── logs/
│   └── archive_pre_refactor/
├── writeup/
└── archive/
    └── pre_refactor/                          # Pre-refactoring code snapshot
```

---

## Configuration

All paths and hyperparameters are centralized in `code/config.py`:

```python
from config import config, set_version, set_model, add_version_argument

# Set model (default: llama2_13b_chat)
set_model("llama2_13b_chat")

# Set Exp 2 data version (required before accessing Exp 2 paths)
set_version("balanced_gpt", turn=5)

# Key paths (version-dependent, set after set_version)
config.RESULTS.alignment       # → results/llama2_13b_chat/balanced_gpt/alignment/
config.PATHS.exp2_probes        # → exp_2/results/llama2_13b_chat/balanced_gpt/probe_training/data/

# Key paths (model-scoped, version-independent)
config.RESULTS.concept_overlap  # → results/llama2_13b_chat/concept_overlap/
config.RESULTS.comparisons      # → results/llama2_13b_chat/comparisons/

# Variant management (optional — default is "" for standard 40-prompt approach)
from config import set_variant, variant_filename, get_variant_suffix
set_variant("_1")                              # Activate top-1 variant
variant_filename("alignment", ".npz")          # → "alignment_top_align.npz"
```

---

## Dimension Registry

| ID | Name | Category | Description |
|----|------|----------|-------------|
| 0 | baseline | Baseline | Human vs AI (entity framing only, no concept content) |
| 1 | phenomenology | Mental | Conscious experience, qualia |
| 2 | emotions | Mental | Emotional states |
| 3 | agency | Mental | Free will, autonomy |
| 4 | intentions | Mental | Goals, desires |
| 5 | prediction | Mental | Forecasting, anticipation |
| 6 | cognitive | Mental | Reasoning, problem-solving |
| 7 | social | Mental | Relationships, empathy |
| 8 | embodiment | Physical | Physical presence |
| 9 | roles | Physical | Social roles, identity |
| 10 | animacy | Physical | Aliveness |
| 11 | formality | Pragmatic | Communication style |
| 12 | expertise | Pragmatic | Knowledge domain |
| 13 | helpfulness | Pragmatic | Assistance behavior |
| 14 | biological | Bio Ctrl | Biological features (control) |
| 15 | shapes | Shapes | Round vs angular (negative control) |
| 16 | mind_holistic | Meta | Pooled mind dimensions 1-10 |
| 17 | attention | Mental | Focus, awareness |
| 18 | sysprompt_contrasts | SysPrompt | System prompt (contrasts mode) |
| 20-23 | sysprompt_* | SysPrompt | System prompt standalone variants |
| 25 | beliefs | Mental | Epistemic states |
| 26 | desires | Mental | Motivational states |
| 27 | goals | Mental | Objective-directed planning |
| 29 | shapes_flip | Shapes | Flipped polarity shapes control |
| 30 | granite_sandstone | Shapes | Rock types (orthogonal control) |
| 31 | squares_triangles | Shapes | Geometric shapes (orthogonal control) |
| 32 | horizontal_vertical | Shapes | Spatial orientation (orthogonal control) |

---

## Dependencies

### Internal Dependencies
- **Exp 2 probes**: Required for alignment analysis and cross-prediction
  - Path: `exp_2/results/llama2_13b_chat/{version}/probe_training/data/turn_5/{operational,metacognitive}/`
- **Exp 1 utils**: Linguistic analysis functions used by behavioral analysis scripts
  - Path: `exp_1/code/`

### External Dependencies
- **LLaMA-2-13B-Chat**: HuggingFace transformers model
- **PyTorch**: GPU-accelerated tensor operations
- **NumPy, SciPy**: Numerical computing
- **Matplotlib**: Figure generation

---

## Changelog

### March 2026: Other-Focused Standalone Variant (`_other`)
- **Purpose**: Tests whether concept alignment depends on self vs other perspective
- **Prompts**: 25 files in `concepts/other/standalone/` — same dims as original standalone but with "someone" as subject instead of impersonal/generic framing
- **Variant suffix**: `_other` on all data files (e.g., `concept_activations_other.npz`, `summary_other.json`)
- **Pipeline**: Elicitation via `slurm/other_elicit_standalone.sh`, alignment via `slurm/other_alignment.sh`
- **Scope**: Standalone analysis only (contrast prompts already use third-person entity framing)

### March 2026: Structural Refactoring
- **Code flattened**: `pipeline/`, `analysis/alignment/`, `analysis/probes/`, `analysis/lexical/` merged into `code/`
- **Renamed `src/` → `utils/`**: Shared utility modules now in `code/utils/`
- **Model dimension added**: `--model` flag (default: `llama2_13b_chat`), `set_model()` / `add_model_argument()`
- **Results restructured**: `results/alignment/versions/{v}/` → `results/llama2_13b_chat/{v}/alignment/`
- **Config rewritten**: Model-scoped output paths, fixed exp_2 probe paths to post-refactor structure
- **SLURM updated**: `/jukebox/` → `/mnt/cup/`, script paths updated for flat structure
- **Old structure archived**: `archive/pre_refactor/`

### February 2026: Layer Filtering + Nonsense Versions
- Excluded layers 0-5 from alignment analysis
- Added `nonsense_codeword` and `nonsense_ignore` versions
- Multi-version support via `--version` flag and `set_version()`
- Script renumbering to reflect logical pipeline order
- Centralized configuration in `config.py`
