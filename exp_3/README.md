# Experiment 3: Concept-of-Mind Representations

**Author**: Rachel C. Metzgar, Princeton University
**Last Updated**: February 27, 2026

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

Analyses that interact with Exp 2 data (alignment, cross-prediction) require a `--version` flag specifying which Exp 2 data version to use. Results are saved to version-specific subdirectories.

| Version | Description | Probes | Conversations |
|---------|-------------|--------|---------------|
| `labels` | AI/human labels instead of names | Yes (124 files) | No |
| `balanced_names` | Balanced gender names | Yes (124 files) | No |
| `balanced_gpt` | Balanced names + GPT-4 replacement | Yes (124 files) | No |
| `names` | Original Sam/Casey names (deprecated) | Yes (124 files) | Yes (2001 files) |
| `nonsense_codeword` | Tokens framed as arbitrary session code | Yes (124 files) | No |
| `nonsense_ignore` | Tokens framed with "ignore" instruction | Yes (124 files) | No |

**Note**: Only `names` has conversations, so cross-prediction (Phase 7) will skip the "Concept → Conversation" direction for other versions. Nonsense versions serve as controls: `nonsense_codeword` is the clean null (0/23 behavioral effects), confirming that partner identity effects require semantic processing of the label, not just token presence.

---

## Quick Start

### Environment Setup
```bash
# Create environments from YAML files
conda env create -f envs/llama2_env.yml     # For pipeline phases 1-3, 5
conda env create -f envs/behavior_env.yml   # For phase 4 (behavioral analysis)
```

### Run Full Pipeline
```bash
# Phase 1: Extract concept activations (contrasts mode)
sbatch code/slurm/1_elicit_contrasts.sh

# Phase 2: Alignment analysis (activations only)
sbatch --export=VERSION=labels code/slurm/2a_alignment_analysis.sh

# Phase 3: Train concept probes
python code/pipeline/3_train_concept_probes.py --version labels --dim_id 1

# Phase 4: Probe alignment stats + figures
sbatch --export=VERSION=labels code/slurm/4a_contrast_pipeline.sh

# Phase 5: Concept intervention (V1 mode)
sbatch code/slurm/5a_concept_intervention.sh

# Phase 6: Behavioral analysis
sbatch code/slurm/6_behavior_analysis.sh

# Phase 7: Cross-prediction
python code/pipeline/7_cross_prediction.py --version labels
```

---

## Directory Structure (Simplified)

```
exp_3/
├── README.md                  # This file
│
├── code/                      # All code lives here
│   ├── config.py              # Central configuration
│   │
│   ├── pipeline/              # Main pipeline (run sequentially)
│   │   ├── 1_elicit_concept_vectors.py      # Phase 1: Extract concept activations
│   │   ├── 3_train_concept_probes.py        # Phase 3: Train probes on concepts
│   │   ├── 5_concept_intervention.py        # Phase 5: Steer generation with concepts
│   │   ├── 6_behavior_analysis.py           # Phase 6: Linguistic feature analysis
│   │   └── 7_cross_prediction.py            # Phase 7: Cross-domain probe evaluation
│   │
│   ├── analysis/              # Statistical analyses
│   │   ├── alignment/         # Phase 2: 2a-2e alignment analysis, layer profiles, sysprompts
│   │   ├── probes/            # Phase 4: probe alignment stats + figures
│   │   │   ├── 4a_compute_alignment_stats.py    # Stats (contrast + standalone)
│   │   │   └── 4b_generate_alignment_figures.py # Figures (contrast + standalone)
│   │   └── lexical/           # 2 scripts: compute lexical overlap + generate reports
│   │       ├── lexical_distinctiveness.py       # Compute Jaccard similarity
│   │       └── generate_lexical_report.py       # ← UNIFIED reports (HTML + PPTX)
│   │
│   ├── src/                   # Shared utilities
│   │   ├── dataset.py         # Dataset loaders
│   │   ├── probes.py          # Probe architectures
│   │   ├── intervention_utils.py  # Steering functions
│   │   ├── prompt_utils.py    # Prompt templates
│   │   └── train_test_utils.py    # Training loops
│   │
│   ├── slurm/                 # SLURM batch scripts
│   └── utils/                 # Utility functions
│
├── concepts/                  # Concept prompt definitions (Python modules)
│   ├── contrasts/             # Human vs AI paired prompts
│   └── standalone/            # Concept-only prompts (no entity framing)
│
├── data/                      # Intermediate data (activations, probes)
├── results/                   # Final outputs (figures, stats, CSVs)
└── archive/                   # Deprecated versions + obsolete scripts
    ├── names/                 # Original names version (deprecated)
    ├── old/                   # Legacy code
    └── originals/             # Pre-consolidation scripts
```

---

## Main Pipeline (7 Phases)

### Phase 1: Concept Extraction
**Script**: `code/pipeline/1_elicit_concept_vectors.py`

Extract LLaMA-2-13B's internal representations for concepts like "phenomenology", "emotions", "agency", etc.

**Two modes**:
- **Contrasts**: Human vs AI paired prompts (e.g., "Humans have phenomenal consciousness" vs "AIs process information")
- **Standalone**: Concept-only prompts without entity framing (e.g., "Phenomenal consciousness involves...")

```bash
# Single dimension, contrasts mode
python code/pipeline/1_elicit_concept_vectors.py --mode contrasts --dim_id 1

# All dimensions (SLURM array)
sbatch code/slurm/1_elicit_contrasts.sh
```

**Outputs**: `data/concept_activations/{mode}/{dim_name}/concept_activations.npz`

---

### Phase 2: Alignment Analysis (Activations Only)
**Scripts**: `code/analysis/alignment/2a_alignment_analysis.py` through `2e_summarize_cross_dimension.py`

Compute alignment between concept vectors (Phase 1) and Exp 2 conversational probes. No trained concept probes needed.

**Layer filtering**: Alignment is computed over layers 6–40 only (35 of 41). Layers 0–5 are excluded because layer 0 (token embeddings) produces content-independent mean activations that create spurious alignment, and layers 1–5 have prompt-format confounds with near-zero-norm contrast vectors.

```bash
# Core alignment (raw, residual, standalone)
python code/analysis/alignment/2a_alignment_analysis.py --version labels --analysis all

# Layer profiles
python code/analysis/alignment/2b_layer_profile_analysis.py --version labels --analysis all

# System prompt elicitation + alignment
python code/analysis/alignment/2c_elicit_sysprompt_vectors.py
python code/analysis/alignment/2d_sysprompt_alignment.py --analysis all

# Cross-dimension summary
python code/analysis/alignment/2e_summarize_cross_dimension.py --version labels
```

**Outputs**: `results/alignment/{version}/contrasts/{raw,residual}/`, `results/alignment/{version}/standalone/`

---

### Phase 3: Train Concept Probes
**Script**: `code/pipeline/3_train_concept_probes.py`

Train linear probes to classify human vs AI concepts from activations. Includes a quick alignment sanity check against Exp 2 probes.

```bash
python code/pipeline/3_train_concept_probes.py --version labels --dim_id 1
```

**Outputs**: `data/concept_probes/{dim_name}/concept_probe_layer_{N}.pth`

---

### Phase 4: Probe Alignment Analysis (Needs Probes)
**Scripts**: `code/analysis/probes/4a_compute_alignment_stats.py`, `4b_generate_alignment_figures.py`

Compute comprehensive statistics and figures for concept-probe alignment. Requires trained probes from Phase 3.

```bash
# Stats (contrast, standalone, or both)
python code/analysis/probes/4a_compute_alignment_stats.py --version labels --mode both

# Figures
python code/analysis/probes/4b_generate_alignment_figures.py --version labels --mode both
```

**Outputs**:
- `results/concept_probe_alignment/{version}/summaries/` (contrast)
- `results/standalone_alignment/{version}/summaries/` (standalone)
- `results/concept_probe_alignment/{version}/figures/` (contrast figures)
- `results/standalone_alignment/{version}/figures/` (standalone figures)

---

### Phase 5: Concept Intervention
**Script**: `code/pipeline/5_concept_intervention.py`

Steer conversational generation by adding/subtracting concept vectors.

**Two modes**:
- **V1**: Single-turn causal test questions (dose-response sweep: N = 1, 2, 4, 8)
- **V2**: Multi-turn naturalistic conversations (recreates Exp 1 structure)

```bash
# V1 mode: Single dimension, all strengths
python code/pipeline/5_concept_intervention.py --mode V1 --dim_id 7

# V2 mode: Multi-turn, one subject
python code/pipeline/5_concept_intervention.py --mode V2 --dim_id 7 --subject_idx 0
```

**Outputs**: `results/interventions/V{1,2}/{dim_name}/is_{N}/`

---

### Phase 6: Behavioral Analysis
**Script**: `code/pipeline/6_behavior_analysis.py`

Compute linguistic features (hedges, discourse markers, Theory of Mind language, sentiment, etc.) and run statistical tests.

```bash
# Single dimension, single strength
python code/pipeline/6_behavior_analysis.py --version v1 --dim_id 7 --strength 4

# All dimensions, all strengths
python code/pipeline/6_behavior_analysis.py --version v1 --all
```

**Outputs**: `results/behavioral/V{1,2}/{dim_name}/is_{N}/behavioral_stats.txt`

---

### Phase 7: Cross-Prediction
**Script**: `code/pipeline/7_cross_prediction.py`

Test whether concept probes trained on conceptual prompts can predict partner identity in conversations (and vice versa).

```bash
python code/pipeline/7_cross_prediction.py --version labels
python code/pipeline/7_cross_prediction.py --version names   # only version with conversations
```

**Note**: Only `names` has conversations. Other versions will skip "Concept → Conversation" cross-prediction and only run "Conversation → Concept" + cosine alignment.

**Outputs**: `results/cross_prediction/{version}/cross_prediction_results.json`

---

## Lexical Overlap Analysis

#### 1. `code/analysis/lexical/lexical_distinctiveness.py`
Computes Jaccard similarity between human/AI prompt word sets and tests correlation with alignment.

```bash
python code/analysis/lexical/lexical_distinctiveness.py
```

**Outputs**: `results/lexical/lexical_distinctiveness.csv`

---

#### 2. `code/analysis/lexical/generate_lexical_report.py`
Generates comprehensive lexical overlap reports.

**Three formats**:
- **html**: Interactive web report with embedded figures
- **pptx**: PowerPoint presentation
- **both**: Generate both formats

```bash
# HTML report only
python code/analysis/lexical/generate_lexical_report.py --format html

# PowerPoint only
python code/analysis/lexical/generate_lexical_report.py --format pptx

# Both formats
python code/analysis/lexical/generate_lexical_report.py --format both
```

**Outputs**:
- `results/lexical/lexical_overlap_investigation/LEXICAL_OVERLAP_REPORT.html`
- `results/lexical/lexical_overlap_investigation/LEXICAL_OVERLAP_REPORT.pptx`

**Replaces**: `build_lexical_overlap_report.py`, `build_lexical_overlap_pptx.py`

---

## Configuration

All paths and hyperparameters are centralized in `code/config.py`:

```python
from config import config, set_version, add_version_argument

# Set Exp 2 data version (required before accessing Exp 2 paths)
set_version("labels")  # or "balanced_names", "balanced_gpt", "names", "nonsense_codeword", "nonsense_ignore"

# Key settings
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
INPUT_DIM = 5120
N_LAYERS = 41

# Exp 2 paths (set dynamically by set_version)
config.PATHS.exp2_probes     # → exp_2/data/{version}/probe_checkpoints/
config.PATHS.exp2_conversations  # → exp_2/data/{version}/human_ai_conversations/

# Analysis settings
ANALYSIS.n_permutations = 10000
ANALYSIS.n_bootstrap = 1000
ANALYSIS.restricted_layer_start = 6  # Exclude layers 0-5 from alignment
ANALYSIS.seed = 42
```

To change settings, edit `config.py` once instead of modifying 14 scripts.

---

## Dependencies

### Internal Dependencies
- **Exp 2 probes**: Required for alignment analysis and cross-prediction
  - Path: `exp_2/data/{version}/probe_checkpoints/turn_5/{control,reading}_probe/`
  - Available for all 6 versions (labels, balanced_names, balanced_gpt, names, nonsense_codeword, nonsense_ignore)

- **Exp 2 conversations**: Required for cross-prediction "Concept → Conversation" direction (Phase 7)
  - Path: `exp_2/data/{version}/human_ai_conversations/`
  - **Only available for `names` version** (2001 files). Other versions skip this direction.

### External Dependencies
- **LLaMA-2-13B-Chat**: HuggingFace transformers model
- **PyTorch**: GPU-accelerated tensor operations
- **NumPy, SciPy**: Numerical computing
- **Matplotlib**: Figure generation
- **python-pptx**: PowerPoint generation (optional, for `--format pptx`)

---

## Key Results

### Concept Elicitation
- 19 conceptual dimensions (phenomenology, emotions, agency, intentions, etc.)
- 50 prompts per dimension (contrasts mode) or 25-50 prompts (standalone mode)
- Activations extracted across all 41 layers of LLaMA-2-13B-Chat

### Probe Training
- Linear probes trained per layer (41 layers × 19 dimensions)
- Achieves >85% accuracy for most mental state dimensions
- Lower accuracy for control dimensions (shapes, biological features)

### Alignment
- Strong alignment between concept vectors and Exp 2 conversational probes
- Alignment strongest in middle-to-upper layers (layers 20-35)
- Suggests concept-of-mind representations are recruited during conversational adaptation

### Behavioral Effects
- Concept injection produces measurable shifts in linguistic style
- Effects qualitatively similar to Exp 2 conversational steering
- Suggests conversational adaptation draws on general semantic knowledge

---

## Dimension Registry

Dimensions are defined in `concepts/{contrasts,standalone}/`.

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

**Key dimensions**:
- **Dim 0 (baseline)**: Entity baseline for residual analysis
- **Dim 15 (shapes)**: Negative control (semantically irrelevant)
- **Dim 16 (mind_holistic)**: **Excluded from all analyses** (circular: pooled from other dims)

---

## Key Concepts

### Contrasts vs Standalone

**Contrasts**: Human/AI paired prompts (e.g., "Humans have emotions" vs "AIs have emotions")
- Computes **mean-difference vector**: mean_human - mean_ai
- Used for training probes and interventions
- Captures "what's different about human vs AI minds"

**Standalone**: Concept-only prompts (e.g., "Consciousness involves subjective experience")
- Computes **mean activation vector**: average across all prompts
- No entity framing (no "human" or "AI" words)
- Tests if alignment is driven by entity words vs concept content

### Raw vs Residual Alignment

**Raw**: Cosine similarity between concept direction and Exp 2 probe weights
- Direct alignment measurement

**Residual**: Same as raw, but after projecting out entity baseline (dim 0)
- Removes shared "human vs AI" component
- Tests if alignment is concept-specific

### Control Dimensions

**Dim 0 (baseline)**: Entity labels only ("this is a human/AI"), no concept content
- Floor for alignment (any concept should beat this)

**Dim 15 (shapes)**: Round vs angular shapes
- Negative control (semantically irrelevant to human/AI distinction)
- Alignment should be near zero

---

## Troubleshooting

**Import errors**: Use `from config import config` instead of hardcoded paths. Check `sys.path` if imports fail after reorganization.

**File not found**: Verify prerequisite phases have been run. Check paths in `config.py`.

**CUDA out of memory**: Reduce batch size in `config.py`. Analysis scripts don't need GPU — use `DEVICE="cpu"`.

**Results in wrong directory**: Ensure scripts use `config.RESULTS.*` paths, not `data/`.

---

## Changelog

### February 27, 2026: Layer Filtering Fix + Nonsense Versions
- **Fixed spurious alignment artifact**: Excluded layers 0–5 from alignment analysis (`2a_alignment_analysis.py`). Layer 0 (token embedding) produced content-independent mean activations that inflated standalone alignment, particularly for `balanced_gpt` (control probe R² was 5–17× higher than other versions due to a chance alignment at layer 0). Layers 1–5 excluded for prompt-format confounds.
- **Added nonsense versions**: `nonsense_codeword` and `nonsense_ignore` now documented as supported data versions throughout
- **Updated Exp 2 paths**: Corrected path references from old `exp_2/{version}/llama_exp_2b-13B-chat/data/` to current `exp_2/data/{version}/`
- **Regenerated comparison reports**: `results/alignment/comparisons/{raw,residual,standalone}_comparison.html` updated with layer-filtered values

### February 22, 2026: Promote labels/ to Top Level
- **Removed `labels/` nesting**: `exp_3/labels/{code,concepts,data,results,logs}` promoted to `exp_3/`
- **Archive**: Deprecated `names/` and `old/` directories moved to `exp_3/archive/`; `archive_originals/` renamed to `archive/originals/`
- **Fixed `config.py` ROOT_DIR bug**: Was resolving to `code/` instead of `exp_3/`. Now uses `.resolve().parent.parent` to correctly point to `exp_3/`
- **Updated all 12 SLURM scripts**: Replaced `exp_3/labels`, `exp_3/llama_exp_3-13B-chat`, and `exp_3-13B-chat_mind` paths with `exp_3`
- **No Python script changes needed**: Internal `sys.path` and `config.py`-based imports still work since `code/` structure is preserved

### February 21, 2026: Script Renumbering
- **Renumbered all scripts** to reflect logical pipeline order:
  - Phase 1: Elicit concept vectors (unchanged)
  - Phase 2: Alignment analysis (was 1b-1e) — only needs activations
  - Phase 3: Train concept probes (was Phase 2)
  - Phase 4: Probe alignment stats/figures (was 2f/3c) — needs trained probes
  - Phase 5: Concept intervention (was Phase 3)
  - Phase 6: Behavioral analysis (was Phase 4)
  - Phase 7: Cross-prediction (was Phase 5)
- **SLURM scripts** renumbered to match (e.g., `1b_alignment_analysis.sh` → `2a_alignment_analysis.sh`)
- **Fixed remaining `llama_exp_3-13B-chat`** PROJECT_ROOT paths in SLURM scripts (2c, 2d, 3, 5a, 6)
- **Cleaned up docs**: Deleted obsolete CONSOLIDATION_PLAN.md, MIGRATION_COMPLETE.md, merged PIPELINE.md unique content into README

### February 21, 2026: Multi-Version Support
- **Added `--version` flag** to all 7 version-dependent scripts (alignment, probe stats/figures, cross-prediction, probe training, layer profiles, cross-dimension summary)
- **`config.py`**: Added `set_version()`, `add_version_argument()`, `get_version_results_dir()` — Exp 2/Exp 1 paths are now set dynamically instead of hardcoded to `balanced_names`
- **Version-specific output directories**: Results now saved to `{base}/{version}/` subdirectories (e.g., `results/alignment/labels/`)
- **Cross-prediction graceful fallback**: Versions without conversations (all except `names`) skip "Concept → Conversation" direction with a warning
- **SLURM scripts**: Updated to require `VERSION` env var, fixed `PROJECT_ROOT` path (was pointing to non-existent `llama_exp_3-13B-chat`)
- **Existing results**: Unversioned results from pre-migration runs are left in place alongside new versioned subdirectories

### February 20, 2026: Code Directory Reorganization
- **Moved all code into `code/` directory**:
  - `config.py`, `pipeline/`, `analysis/`, `src/`, `slurm/`, `utils/` → `code/`
  - Keeps top level clean with only: data/, results/, concepts/, docs
  - Updated all SLURM scripts to call scripts with `code/` prefix
  - Updated all documentation to reflect new paths

- **Benefits**: Cleaner project root, better separation of code vs data

### February 19, 2026: Script Consolidation
- **Consolidated probe analysis scripts** (6 → 2):
  - Created `4a_compute_alignment_stats.py` (unified stats for contrast + standalone modes)
  - Created `4b_generate_alignment_figures.py` (unified figures for contrast + standalone modes)

- **Consolidated lexical report scripts** (3 → 2):
  - Created `generate_lexical_report.py` (unified HTML + PPTX generation)

### February 19, 2026: Config Migration
- Migrated all 20 scripts to use centralized `code/config.py`
- Standardized paths, hyperparameters, and directory structure

---

## References

**TalkTuner Framework:**
Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... & Viégas, F. (2024). Designing a dashboard for transparency and control of conversational AI. *arXiv preprint arXiv:2406.07882*.

**Linguistic Analysis:**
- Demir, C. (2018). Hedging and academic writing. *JLLS*, 14(4), 74-92.
- Fung, L. & Carter, R. (2007). Discourse markers and spoken English. *Applied Linguistics*, 28(3), 410-439.
- Wagovich, S. A. et al. (2024). Mental state verbs taxonomy.

---

## Contact

Rachel C. Metzgar
Princeton University
Graziano Lab
