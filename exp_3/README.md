# Experiment 3: Concept-of-Mind Representations

**Author**: Rachel C. Metzgar, Princeton University
**Last Updated**: March 2026

---

## Overview

This pipeline investigates whether large language models (LLMs) form internal representations of "what human and AI minds are like" and whether these concept-of-mind representations align with and influence conversational behavior.

**Key Question**: When an LLM adjusts its communication style based on partner identity (human vs AI), is it drawing on its general semantic knowledge about what humans and AIs are, or using a task-specific behavioral switch?

**Approach**:
1. Extract the model's internal representations of mental concepts (phenomenology, emotions, agency, etc.)
2. Test alignment with Exp 2 conversational partner-identity representations
3. Causally intervene by injecting concept vectors into generation
4. Analyze behavioral effects and cross-domain generalization

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
│   ├── 5_concept_intervention.py              # Phase 5: Causal intervention
│   ├── 6_behavior_analysis.py                 # Phase 6: Linguistic feature analysis
│   ├── 7_cross_prediction.py                  # Phase 7: Cross-domain probe evaluation
│   ├── 8_lexical_distinctiveness.py           # Lexical: Compute Jaccard similarity
│   └── 8a_lexical_summary_generator.py        #          Generate lexical reports
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
│       │   ├── alignment/                     #     Concept-probe alignment
│       │   ├── concept_steering/              #     Steering outputs
│       │   ├── interventions/                 #     Causal intervention results
│       │   ├── behavioral/                    #     Behavioral analysis
│       │   └── cross_prediction/              #     Cross-domain results
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
```

---

## Dimension Registry

| ID | Name | Category | Description |
|----|------|----------|-------------|
| 0 | baseline | Baseline | Human vs AI (entity baseline) |
| 1 | phenomenology | Mental | Conscious experience |
| 2 | emotions | Mental | Emotional states |
| 3 | agency | Mental | Free will, autonomy |
| 4 | intentions | Mental | Goals, desires |
| 5 | prediction | Mental | Forecasting, theory of mind |
| 6 | cognitive | Mental | Reasoning, problem-solving |
| 7 | social | Mental | Relationships, empathy |
| 8 | embodiment | Physical | Physical presence |
| 9 | roles | Physical | Social roles, identity |
| 10 | animacy | Physical | Aliveness |
| 11 | formality | Pragmatic | Communication style |
| 12 | expertise | Pragmatic | Knowledge domain |
| 13 | helpfulness | Pragmatic | Assistance behavior |
| 14 | biological | Bio Ctrl | Biological features |
| 15 | shapes | Shapes | Round vs angular (negative control) |
| 16 | mind_holistic | Mental | Pooled mind dimensions 1-10 |
| 17 | attention | Mental | Focus, awareness |
| 18-21 | sysprompt_* | SysPrompt | System prompt variants |

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

---

## Contact

Rachel C. Metzgar
Princeton University
Graziano Lab
