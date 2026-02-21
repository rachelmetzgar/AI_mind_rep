# Experiment 3: Concept-of-Mind Pipeline

## Overview

This pipeline tests whether LLMs form internal representations of "what human/AI minds are like" and whether these representations causally influence conversational behavior.

**Key Question**: Do concept-of-mind representations align with partner-identity representations from naturalistic conversations (Exp 2)?

---

## Directory Structure

```
exp_3/labels/
├── config.py                   # Central configuration (paths, hyperparameters)
├── run_pipeline.py            # Master runner script (TODO)
│
├── pipeline/                   # Main 5-phase pipeline (run sequentially)
│   ├── 1_elicit_concept_vectors.py
│   ├── 2_train_concept_probes.py
│   ├── 3_concept_intervention.py
│   ├── 4_behavior_analysis.py
│   └── 5_cross_prediction.py
│
├── analysis/                   # Parallel analyses (run after Phase 1 or 2)
│   ├── alignment/
│   │   ├── 1b_alignment_analysis.py
│   │   ├── 1c_layer_profile_analysis.py
│   │   ├── 1d_elicit_sysprompt_vectors.py
│   │   ├── 1e_sysprompt_alignment.py
│   │   └── summarize_cross_dimension.py
│   │
│   ├── probes/
│   │   ├── 2b_summarize_concept_probes.py
│   │   ├── 2c_permutation_tests.py
│   │   ├── 2d_concept_probe_stats.py     # Contrasts mode
│   │   ├── 2e_concept_probe_figures.py  # Contrasts mode
│   │   ├── 3a_standalone_stats.py       # Standalone mode
│   │   └── 3b_standalone_figures.py      # Standalone mode
│   │
│   └── lexical/
│       ├── lexical_distinctiveness.py
│       ├── build_lexical_overlap_report.py
│       └── build_lexical_overlap_pptx.py
│
├── results/                    # All outputs go here (not in data/)
│   ├── concept_activations/
│   ├── alignment/
│   ├── interventions/
│   ├── behavioral/
│   ├── lexical/
│   └── figures/
│
├── data/                       # Inputs only (activations from Phase 1, probes from Phase 2)
│   ├── concept_activations/
│   ├── concept_probes/
│   └── causality_test_questions/
│
├── src/                        # Shared utilities (dataset, probes, intervention)
│   ├── dataset.py
│   ├── probes.py
│   ├── intervention_utils.py
│   └── ...
│
├── slurm/                      # SLURM batch scripts
│   ├── 1_elicit_contrasts.sh
│   ├── 2_train_concept_probes.py
│   ├── 3_concept_intervention_1.sh
│   └── ...
│
└── concepts/                   # Concept prompt definitions
    ├── contrasts/             # Human vs AI paired prompts
    │   ├── 0_baseline.py
    │   ├── 1_phenomenology.py
    │   └── ...
    │
    └── standalone/            # Concept-only prompts (no entity framing)
        ├── 1_phenomenology.py
        └── ...
```

---

## Main Pipeline (Run Sequentially)

### Phase 1: Extract Concept Activations
**Script**: `pipeline/1_elicit_concept_vectors.py`

Extracts LLaMA-2-13B internal representations for concept prompts (e.g., "Humans have phenomenal consciousness").

**Modes**:
- `--mode contrasts`: Human/AI paired prompts → computes mean-difference vector
- `--mode standalone`: Concept-only prompts (no entity framing) → computes mean activation

**Usage**:
```bash
# Single dimension
python pipeline/1_elicit_concept_vectors.py --mode contrasts --dim_id 1

# SLURM array (all dimensions)
sbatch slurm/1_elicit_contrasts.sh
```

**Outputs**:
- `data/concept_activations/{mode}/{dim_name}/concept_activations.npz`
- `data/concept_activations/{mode}/{dim_name}/concept_vector_per_layer.npz`
- `data/concept_activations/{mode}/{dim_name}/concept_prompts.json`

**Dependencies**: None

---

### Phase 2: Train Concept Probes
**Script**: `pipeline/2_train_concept_probes.py`

Trains linear probes to classify human vs AI based on concept activations.

**Usage**:
```bash
python pipeline/2_train_concept_probes.py --dim_id 1
```

**Outputs**:
- `data/concept_probes/{dim_name}/concept_probe_layer_{N}.pth`
- `data/concept_probes/{dim_name}/accuracy_summary.pkl`

**Dependencies**: Phase 1 (contrasts mode)

---

### Phase 3: Concept Intervention (Generation)
**Script**: `pipeline/3_concept_intervention.py`

Steers conversational generation by adding/subtracting concept vectors.

**Modes**:
- `--mode V1`: Single-turn causal test questions (sweeps N = 1, 2, 4, 8)
- `--mode V2`: Multi-turn naturalistic conversations (recreates Exp 1)

**Usage**:
```bash
# V1: Single dimension, all strengths
python pipeline/3_concept_intervention.py --mode V1 --dim_id 7

# V2: Multi-turn, one subject
python pipeline/3_concept_intervention.py --mode V2 --dim_id 7 --subject_idx 0
```

**Outputs**:
- `results/interventions/V{1,2}/{dim_name}/is_{N}/intervention_responses.csv`
- `results/interventions/V{1,2}/{dim_name}/is_{N}/generation_config.json`

**Dependencies**: Phase 2

---

### Phase 4: Behavioral Analysis
**Script**: `pipeline/4_behavior_analysis.py`

Computes linguistic features (hedges, discourse markers, Theory of Mind phrases, etc.) and statistical tests.

**Usage**:
```bash
# Single dimension, single strength
python pipeline/4_behavior_analysis.py --version v1 --dim_id 7 --strength 4

# All dimensions, all strengths
python pipeline/4_behavior_analysis.py --version v1 --all
```

**Outputs**:
- `results/behavioral/V{1,2}/{dim_name}/is_{N}/behavioral_stats.txt`
- `results/behavioral/V{1,2}/{dim_name}/is_{N}/utterance_level_metrics.csv`
- `results/behavioral/V{1,2}/cross_dimension_summary.csv`

**Dependencies**: Phase 3

---

### Phase 5: Cross-Prediction
**Script**: `pipeline/5_cross_prediction.py`

Tests whether concept probes can predict partner identity in conversations (and vice versa).

**Usage**:
```bash
python pipeline/5_cross_prediction.py
```

**Outputs**:
- `results/cross_prediction/cross_prediction_results.json`
- `results/cross_prediction/cosine_alignment.json`
- `results/cross_prediction/combined_analysis_plot.png`

**Dependencies**: Phase 2 + Exp 2b conversational data

---

## Parallel Analyses (Run After Phase 1 or 2)

These can run in parallel with each other, but have dependencies on main pipeline phases.

### Alignment Analyses (After Phase 1)

**1b. Compute Alignment (Raw/Residual/Standalone)**
```bash
python analysis/alignment/1b_alignment_analysis.py --analysis raw
python analysis/alignment/1b_alignment_analysis.py --analysis residual  # Entity baseline projected out
python analysis/alignment/1b_alignment_analysis.py --analysis standalone
```

**Outputs**: `results/alignment/{raw,residual,standalone}/`
**Dependencies**: Phase 1

---

**1c. Layer Profile Analysis**
```bash
python analysis/alignment/1c_layer_profile_analysis.py
```

Analyzes layer-by-layer alignment profiles, peak layers, alignment onset.

**Outputs**: `results/alignment/layer_profiles/`
**Dependencies**: Phase 1 + 1b

---

**1d. System Prompt Variants**
```bash
python analysis/alignment/1d_elicit_sysprompt_vectors.py --dim_id 18
```

Extracts concept vectors for different system prompt phrasings (dim 18-21).

**Outputs**: `data/concept_activations/standalone/18_sysprompt_*/`
**Dependencies**: None (standalone data collection)

---

**1e. System Prompt Alignment**
```bash
python analysis/alignment/1e_sysprompt_alignment.py
```

Compares alignment of different system prompt variants.

**Outputs**: `results/alignment/sysprompt/`
**Dependencies**: 1d

---

**Summarize Cross-Dimension Alignment**
```bash
python analysis/alignment/summarize_cross_dimension.py
```

Collects alignment results from all dimensions, creates heatmaps and bar charts.

**Outputs**: `results/alignment/summary/`
**Dependencies**: Phase 2 (quick alignment check)

---

### Probe Analyses (After Phase 2)

**2b. Summarize Probe Training**
```bash
python analysis/probes/2b_summarize_concept_probes.py
```

Summarizes probe accuracies across dimensions, statistical tests vs baseline/shapes.

**Outputs**: `results/concept_probes/summaries/`
**Dependencies**: Phase 2

---

**2c. Permutation Tests**
```bash
python analysis/probes/2c_permutation_tests.py
```

Permutation testing for alignment significance (shuffles labels, recomputes alignment).

**Outputs**: `results/alignment/permutation_tests/`
**Dependencies**: Phase 2 + 1b

---

**2d. Concept Probe Stats (Contrasts)**
```bash
python analysis/probes/2d_concept_probe_stats.py --dim_id 1
```

Bootstrap confidence intervals, pairwise comparisons, category-level analysis.

**Outputs**: `results/alignment/contrasts/stats/`
**Dependencies**: Phase 2 + 1b

---

**2e. Concept Probe Figures (Contrasts)**
```bash
python analysis/probes/2e_concept_probe_figures.py --dim_id 1
```

Generates bar plots, heatmaps, strip plots, scatter plots.

**Outputs**: `results/alignment/contrasts/figures/`
**Dependencies**: 2d

---

**3a. Standalone Stats**
```bash
python analysis/probes/3a_standalone_stats.py --dim_id 1
```

Same as 2d but for standalone mode (bootstrap against zero, no permutation).

**Outputs**: `results/alignment/standalone/stats/`
**Dependencies**: Phase 1 (standalone mode) + 1b (standalone analysis)

---

**3b. Standalone Figures**
```bash
python analysis/probes/3b_standalone_figures.py --dim_id 1
```

Same as 2e but for standalone mode.

**Outputs**: `results/alignment/standalone/figures/`
**Dependencies**: 3a

---

### Lexical Analyses (After Phase 1 or 2)

**Compute Distinctiveness**
```bash
python analysis/lexical/lexical_distinctiveness.py
```

Computes Jaccard similarity, entity word contamination, correlations with alignment.

**Outputs**: `results/lexical/distinctiveness/`
**Dependencies**: Phase 1 + 1b

---

**Generate Reports**
```bash
# HTML report
python analysis/lexical/build_lexical_overlap_report.py

# PowerPoint slides
python analysis/lexical/build_lexical_overlap_pptx.py
```

**Outputs**:
- `results/lexical/LEXICAL_OVERLAP_REPORT.html`
- `results/lexical/LEXICAL_OVERLAP_REPORT.pptx`

**Dependencies**: lexical_distinctiveness.py

---

### Publication Figures

**Make Publication Figures**
```bash
python results/figures/make_pub_figures.py
```

Creates publication-quality figures with consistent styling.

**Outputs**: `results/figures/publication/`
**Dependencies**: All phases complete

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

## Configuration (`config.py`)

All paths and hyperparameters are centralized in `config.py`.

**Key sections**:
- `config.PATHS`: Input paths (concepts, Exp 2b probes, etc.)
- `config.RESULTS`: Output paths (all results/ subdirectories)
- `config.TRAINING`: Probe training hyperparameters
- `config.GEN_V1` / `config.GEN_V2`: Generation hyperparameters
- `config.ANALYSIS`: Bootstrap, permutation, statistical settings

**Usage**:
```python
from config import config

# Access paths
model_name = config.MODEL_NAME
probe_dir = config.PATHS.concept_probes

# Access hyperparameters
n_bootstrap = config.ANALYSIS.n_bootstrap
```

---

## Typical Workflows

### Full Pipeline (One Dimension)

```bash
# Phase 1: Extract concept activations (contrasts + standalone)
python pipeline/1_elicit_concept_vectors.py --mode contrasts --dim_id 7
python pipeline/1_elicit_concept_vectors.py --mode standalone --dim_id 7

# Phase 2: Train concept probes
python pipeline/2_train_concept_probes.py --dim_id 7

# Phase 1b: Compute alignment
python analysis/alignment/1b_alignment_analysis.py --analysis raw
python analysis/alignment/1b_alignment_analysis.py --analysis residual
python analysis/alignment/1b_alignment_analysis.py --analysis standalone

# Phase 3: Generate steered conversations (V1)
python pipeline/3_concept_intervention.py --mode V1 --dim_id 7

# Phase 4: Behavioral analysis
python pipeline/4_behavior_analysis.py --version v1 --dim_id 7

# Phase 5: Cross-prediction (run once after all dims trained)
python pipeline/5_cross_prediction.py
```

---

### SLURM Array Jobs (All Dimensions)

```bash
# Phase 1: Contrasts mode
sbatch slurm/1_elicit_contrasts.sh       # Array job: dim_id = 0-17

# Phase 1: Standalone mode
sbatch slurm/1_elicit_standalone.py      # Array job: dim_id = 0-17

# Phase 2: Train probes
sbatch slurm/2_train_concept_probes.py   # Array job: dim_id = 0-17

# Phase 3: V1 generation
sbatch slurm/3_concept_intervention_1.sh  # Array job: dim_id × strength

# Phase 3: V2 generation
sbatch slurm/3_concept_intervention_2.sh  # Array job: dim_id × subject_id
```

---

### Alignment Analysis Pipeline

```bash
# 1. Extract activations (all dims)
sbatch slurm/1_elicit_contrasts.sh
sbatch slurm/1_elicit_standalone.py

# 2. Compute alignment (all three analyses)
sbatch slurm/1b_alignment_analysis.sh raw
sbatch slurm/1b_alignment_analysis.sh residual
sbatch slurm/1b_alignment_analysis.sh standalone

# 3. Statistical analysis + figures (contrasts)
sbatch slurm/2f_concept_probe_pipeline.sh    # Runs 2d + 2e for all dims

# 4. Statistical analysis + figures (standalone)
sbatch slurm/3c_standalone_pipeline.sh        # Runs 3a + 3b for all dims

# 5. Layer profile analysis
python analysis/alignment/1c_layer_profile_analysis.py

# 6. Lexical analysis
python analysis/lexical/lexical_distinctiveness.py
python analysis/lexical/build_lexical_overlap_report.py
```

---

## Key Concepts

### Modes: Contrasts vs Standalone

**Contrasts**: Human/AI paired prompts (e.g., "Humans have emotions" vs "AIs have emotions")
- Computes **mean-difference vector**: mean_human - mean_ai
- Used for training probes and interventions
- Captures "what's different about human vs AI minds"

**Standalone**: Concept-only prompts (e.g., "Consciousness involves subjective experience")
- Computes **mean activation vector**: average across all prompts
- No entity framing (no "human" or "AI" words)
- Tests if alignment is driven by entity words vs concept content

### Alignment Analyses: Raw vs Residual

**Raw**: Cosine similarity between concept direction and Exp 2b probe weights
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

### Common Issues

**1. Import errors after reorganization**
- Update `sys.path.insert(0, os.path.dirname(__file__))` to account for new locations
- Use `from config import config` instead of hardcoded paths

**2. File not found errors**
- Check that you've run prerequisite phases
- Verify paths in `config.py` point to correct locations

**3. CUDA out of memory**
- Reduce batch size in `config.py`
- Use `DEVICE="cpu"` for analysis scripts (no GPU needed)

**4. Results in wrong directory**
- Check that scripts use `config.RESULTS.*` paths, not `data/`
- Old scripts may still write to `data/` — update import to use `config.py`

---

## Migration from Old Structure

Original scripts are still in root directory. New structure:
- `pipeline/` contains main 5-phase scripts (copied from root)
- `analysis/` contains parallel analyses (copied from root)
- All scripts updated to use `config.py` for paths

**Testing strategy**:
1. Run new script on one dimension
2. Compare outputs to original script (should be identical)
3. Once validated, use new scripts exclusively

---

## Next Steps

1. **Create `utils/` package** with shared functions:
   - `utils/stats.py`: Bootstrap, permutation tests
   - `utils/plotting.py`: Shared plot templates
   - `utils/io_helpers.py`: File loading/saving

2. **Merge duplicate analysis scripts**:
   - `2d + 3a` → `analysis/probes/compute_stats.py` (with `--mode` flag)
   - `2e + 3b` → `analysis/probes/generate_figures.py` (with `--mode` flag)

3. **Create `run_pipeline.py` master runner**:
   - One command to run full pipeline for a dimension
   - Automatic dependency checking

---

## Contact

Questions? Check the session summary in `/usr/people/rm9561/.claude/projects/-mnt-cup-labs-graziano-rachel/memory/exp3_session_summary.md`

Last updated: Feb 19, 2026
