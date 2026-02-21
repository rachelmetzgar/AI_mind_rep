# Experiment 3: Concept-of-Mind Representations

**Author**: Rachel C. Metzgar, Princeton University
**Last Updated**: February 19, 2026

---

## Overview

This pipeline investigates whether large language models (LLMs) form internal representations of "what human and AI minds are like" and whether these concept-of-mind representations align with and influence conversational behavior.

**Key Question**: When an LLM adjusts its communication style based on partner identity (human vs AI), is it drawing on its general semantic knowledge about what humans and AIs are, or using a task-specific behavioral switch?

**Approach**:
1. Extract the model's internal representations of mental concepts (phenomenology, emotions, agency, etc.)
2. Test alignment with Exp 2 conversational partner-identity representations
3. Causally intervene by injecting concept vectors into generation
4. Analyze behavioral effects and cross-domain generalization

See **[PIPELINE.md](PIPELINE.md)** for detailed phase-by-phase instructions.

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

# Phase 2: Train concept probes
sbatch code/slurm/2_train_concept_probes.py

# Phase 3: Concept intervention (V1 mode)
sbatch code/slurm/3_concept_intervention_1.sh

# Phase 4: Behavioral analysis
sbatch code/slurm/behavior_analysis.sh

# Phase 5: Cross-prediction
python code/pipeline/5_cross_prediction.py
```

---

## Directory Structure (Simplified)

```
exp_3/labels/
├── README.md                  # This file
├── PIPELINE.md                # Detailed pipeline documentation
├── CONSOLIDATION_PLAN.md      # Script consolidation rationale
├── MIGRATION_COMPLETE.md      # Config migration documentation
│
├── code/                      # All code lives here
│   ├── config.py              # Central configuration
│   │
│   ├── pipeline/              # Main 5-phase pipeline (run sequentially)
│   │   ├── 1_elicit_concept_vectors.py      # Extract concept activations
│   │   ├── 2_train_concept_probes.py        # Train probes on concepts
│   │   ├── 3_concept_intervention.py        # Steer generation with concepts
│   │   ├── 4_behavior_analysis.py           # Linguistic feature analysis
│   │   └── 5_cross_prediction.py            # Cross-domain probe evaluation
│   │
│   ├── analysis/              # Parallel statistical analyses
│   │   ├── alignment/         # 5 scripts: alignment analysis, layer profiles, sysprompts
│   │   ├── probes/            # 2 scripts (consolidated): stats + figures
│   │   │   ├── compute_alignment_stats.py       # ← UNIFIED stats (contrast + standalone)
│   │   │   └── generate_alignment_figures.py    # ← UNIFIED figures (contrast + standalone)
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
└── archive_originals/         # Obsolete scripts (pre-consolidation)
```

---

## Main Pipeline (5 Phases)

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

### Phase 2: Probe Training
**Script**: `code/pipeline/2_train_concept_probes.py`

Train linear probes to classify human vs AI concepts from activations.

```bash
python code/pipeline/2_train_concept_probes.py --dim_id 1
```

**Outputs**: `data/concept_probes/{dim_name}/concept_probe_layer_{N}.pth`

---

### Phase 3: Concept Intervention
**Script**: `code/pipeline/3_concept_intervention.py`

Steer conversational generation by adding/subtracting concept vectors.

**Two modes**:
- **V1**: Single-turn causal test questions (dose-response sweep: N = 1, 2, 4, 8)
- **V2**: Multi-turn naturalistic conversations (recreates Exp 1 structure)

```bash
# V1 mode: Single dimension, all strengths
python code/pipeline/3_concept_intervention.py --mode V1 --dim_id 7

# V2 mode: Multi-turn, one subject
python code/pipeline/3_concept_intervention.py --mode V2 --dim_id 7 --subject_idx 0
```

**Outputs**: `results/interventions/V{1,2}/{dim_name}/is_{N}/`

---

### Phase 4: Behavioral Analysis
**Script**: `code/pipeline/4_behavior_analysis.py`

Compute linguistic features (hedges, discourse markers, Theory of Mind language, sentiment, etc.) and run statistical tests.

```bash
# Single dimension, single strength
python code/pipeline/4_behavior_analysis.py --version v1 --dim_id 7 --strength 4

# All dimensions, all strengths
python code/pipeline/4_behavior_analysis.py --version v1 --all
```

**Outputs**: `results/behavioral/V{1,2}/{dim_name}/is_{N}/behavioral_stats.txt`

---

### Phase 5: Cross-Prediction
**Script**: `code/pipeline/5_cross_prediction.py`

Test whether concept probes trained on conceptual prompts can predict partner identity in conversations (and vice versa).

```bash
python code/pipeline/5_cross_prediction.py
```

**Outputs**: `results/cross_prediction/cross_prediction_results.json`

---

## Statistical Analysis Scripts (Consolidated)

### Probe Alignment Analysis

**Two unified scripts** replace 6 separate scripts:

#### 1. `code/analysis/probes/compute_alignment_stats.py`
Computes comprehensive statistics for concept-probe alignment.

**Two modes**:
- **contrast**: Human vs AI contrasts (permutation tests)
- **standalone**: Concept-only (bootstrap tests)

```bash
# Contrast mode (human vs AI differences)
python code/analysis/probes/compute_alignment_stats.py --mode contrast

# Standalone mode (concept-only projections)
python code/analysis/probes/compute_alignment_stats.py --mode standalone

# Both modes
python code/analysis/probes/compute_alignment_stats.py --mode both
```

**Outputs**:
- `results/probes/alignment/summaries/alignment_stats.json` (contrast)
- `results/probes/standalone_alignment/summaries/standalone_alignment_stats.json` (standalone)

**Replaces**: `2b_summarize_concept_probes.py`, `2c_permutation_tests.py`, `2d_concept_probe_stats.py`, `3a_standalone_stats.py`

---

#### 2. `code/analysis/probes/generate_alignment_figures.py`
Generates publication-quality figures for alignment analysis.

```bash
# Contrast mode figures
python code/analysis/probes/generate_alignment_figures.py --mode contrast

# Standalone mode figures
python code/analysis/probes/generate_alignment_figures.py --mode standalone

# Both modes
python code/analysis/probes/generate_alignment_figures.py --mode both
```

**Outputs**:
- `results/probes/alignment/figures/` (contrast)
- `results/probes/standalone_alignment/figures/` (standalone)

**Replaces**: `2e_concept_probe_figures.py`, `3b_standalone_figures.py`

---

### Lexical Overlap Analysis

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

## Script Count Summary

### Before Consolidation
- **Pipeline**: 5 scripts
- **Alignment analysis**: 5 scripts
- **Probe analysis**: 6 scripts
- **Lexical analysis**: 3 scripts
- **Total**: 19 scripts

### After Consolidation
- **Pipeline**: 5 scripts
- **Alignment analysis**: 5 scripts
- **Probe analysis**: 2 scripts ✅ (-4 scripts)
- **Lexical analysis**: 2 scripts ✅ (-1 script)
- **Total**: 14 scripts ✅ (-26% reduction)

**Consolidation benefits**:
- Unified interface with `--mode` and `--format` flags
- Reduced code duplication (~65K lines)
- Clearer usage patterns
- Easier maintenance

---

## Configuration

All paths and hyperparameters are centralized in `code/config.py`:

```python
# Key settings
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
INPUT_DIM = 5120
N_LAYERS = 41

# Paths
PATHS.exp2_probes = "/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2/balanced_names/..."
PATHS.concept_activations_contrasts = "data/concept_activations/contrasts"
PATHS.concept_activations_standalone = "data/concept_activations/standalone"

# Analysis settings
ANALYSIS.n_permutations = 10000
ANALYSIS.n_bootstrap = 10000
ANALYSIS.seed = 42

# Training settings
TRAINING.n_epochs = 50
TRAINING.batch_size = 16
TRAINING.learning_rate = 0.001
```

To change settings, edit `config.py` once instead of modifying 14 scripts.

---

## Dependencies

### Internal Dependencies
- **Exp 2 probes**: Required for alignment analysis and cross-prediction
  - Path: `/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/data/probe_checkpoints/`
  - Files: `control_probe/`, `reading_probe/`

- **Exp 2 conversations**: Required for cross-prediction (Phase 5)
  - Path: `/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/data/human_ai_conversations/`

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

## Changelog

### February 20, 2026: Code Directory Reorganization
- **Moved all code into `code/` directory**:
  - `config.py`, `pipeline/`, `analysis/`, `src/`, `slurm/`, `utils/` → `code/`
  - Keeps top level clean with only: data/, results/, concepts/, docs
  - Updated all SLURM scripts to call scripts with `code/` prefix
  - Updated all documentation to reflect new paths

- **Benefits**: Cleaner project root, better separation of code vs data

### February 19, 2026: Script Consolidation
- **Consolidated probe analysis scripts** (6 → 2):
  - Created `compute_alignment_stats.py` (unified stats for contrast + standalone modes)
  - Created `generate_alignment_figures.py` (unified figures for contrast + standalone modes)
  - Archived obsolete scripts: `2b`, `2c`, `2d`, `2e`, `3a`, `3b`

- **Consolidated lexical report scripts** (3 → 2):
  - Created `generate_lexical_report.py` (unified HTML + PPTX generation)
  - Archived obsolete scripts: `build_lexical_overlap_report.py`, `build_lexical_overlap_pptx.py`

- **Benefits**: 26% reduction in script count, clearer usage patterns, reduced code duplication

### February 19, 2026: Config Migration
- Migrated all 20 scripts to use centralized `code/config.py`
- Standardized paths, hyperparameters, and directory structure
- Created comprehensive documentation (PIPELINE.md, MIGRATION_COMPLETE.md, CONSOLIDATION_PLAN.md)

---

## Documentation

- **[PIPELINE.md](PIPELINE.md)**: Detailed phase-by-phase pipeline instructions
- **[MIGRATION_COMPLETE.md](MIGRATION_COMPLETE.md)**: Config migration status and redundancy analysis
- **[CONSOLIDATION_PLAN.md](CONSOLIDATION_PLAN.md)**: Script consolidation rationale and implementation

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
