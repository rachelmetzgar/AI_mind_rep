# Experiment 4: Mind Perception Geometry

**Author:** Rachel C. Metzgar, Princeton University

## Reports

### Core Pipeline — Gray Replication
- [Cross-model behavioral summary](results/comparisons/behavioral_summary_report.html) — 10 publication figures comparing both models: factor structure, human correlations, entity placements, RSA
- [Base model results](results/llama2_13b_base/gray_replication/behavior/results_report.html) — Pairwise and individual rating analyses for the base model
- [Analysis explainer](results/llama2_13b_base/gray_replication/behavior/analysis_explainer.html) — Step-by-step walkthrough of the PCA / varimax / factor-score methodology

### Gray Simple — Internal Representations
- [Chat RSA report](results/llama2_13b_chat/gray_simple/internals/full_dataset/rsa_report.html) — RSA-by-dimension analysis for the chat model
- [Base RSA report](results/llama2_13b_base/gray_simple/internals/rsa_report.html) — RSA-by-dimension analysis for the base model

### Human-AI Adaptation (30 AI/human characters)
- [Base PCA report](results/llama2_13b_base/human_ai_adaptation/behavior/full_dataset/gray_chars_pca_report.html) — PCA on 30 characters rated on 18 Gray capacities
- [Base detailed report](results/llama2_13b_base/human_ai_adaptation/behavior/full_dataset/gray_chars_detailed_report.html) — Per-character ratings breakdown
- [Base RSA report](results/llama2_13b_base/human_ai_adaptation/behavior/full_dataset/gray_chars_rsa_report.html) — Character activation RSA

### Expanded Mental Concepts (Exp 3 bridge)
- [Base behavioral PCA](results/llama2_13b_base/expanded_mental_concepts/behavior/pca/full_dataset/) — 4 reports: behavioral PCA, matched PCA, attribution analysis, detailed responses
- [Base activation RSA](results/llama2_13b_base/expanded_mental_concepts/internals/rsa/full_dataset/activation_rsa_report.html) — Activation-space RSA for 28 characters
- [Chat activation RSA](results/llama2_13b_chat/expanded_mental_concepts/internals/rsa/full_dataset/activation_rsa_report.html) — Chat model activation RSA
- [Base contrast alignment](results/llama2_13b_base/expanded_mental_concepts/internals/contrast_alignment/contrast_alignment_report.html) — Exp 3 contrast vectors projected onto character space
- [Chat contrast alignment](results/llama2_13b_chat/expanded_mental_concepts/internals/contrast_alignment/contrast_alignment_report.html) — Chat model contrast alignment
- [Base standalone alignment](results/llama2_13b_base/expanded_mental_concepts/internals/standalone_alignment/standalone_alignment_report.html) — Exp 3 standalone vectors projected onto character space
- [Chat standalone alignment](results/llama2_13b_chat/expanded_mental_concepts/internals/standalone_alignment/standalone_alignment_report.html) — Chat model standalone alignment
- [Base concept RSA summary](results/llama2_13b_base/expanded_mental_concepts/internals/concept_rsa/data/cross_concept_rsa_summary.md) — Per-dimension concept-specific RSA
- [Chat concept RSA summary](results/llama2_13b_chat/expanded_mental_concepts/internals/concept_rsa/data/cross_concept_rsa_summary.md) — Chat model concept RSA

### Archive
- [Results walkthrough (pre-refactor)](archive/exp4_results_walkthrough.html) — Combined results across both models (legacy format)

---

## Motivation

Experiments 1-3 treat partner identity as a binary (human vs. AI). Human folk psychology is far richer. Gray, Gray, & Wegner (2007, *Science*) showed that humans perceive minds along two orthogonal dimensions: **Experience** (the capacity to feel -- hunger, fear, pain, pleasure, joy) and **Agency** (the capacity to plan and act -- self-control, morality, memory, planning, thought). ~2,400 participants rated 13 diverse entities via pairwise comparisons on 18 mental capacities, and PCA with varimax rotation recovered this two-factor structure, explaining 97% of variance.

Exp 4 asks: **do large language models have an implicit folk psychology of mind that mirrors this human structure?** If the model's representational geometry over diverse entities (baby, dog, robot, God, adults, etc.) resembles the human Experience/Agency space, it would suggest the model has internalized a continuous, multi-dimensional folk psychology -- not just a binary human/AI switch.

---

## Design

Four experimental branches at increasing scope:

| Branch | Entities | Prompts | Behavior? | Internals? |
|--------|----------|---------|-----------|------------|
| `gray_replication` | 13 Gray entities | Pairwise on 18 capacities | Yes | — |
| `gray_simple` | 13 Gray entities | "Think about {entity}" | — | Yes (RSA, neural PCA) |
| `human_ai_adaptation` | 30 AI/human characters | Pairwise on 18 Gray capacities | Yes | — |
| `expanded_mental_concepts` | 28 AI/human characters | Pairwise on ~27 Exp 3 concept dims + activation extraction | Yes | Yes |

Both chat/instruct and base model variants are tested, since RLHF safety training causes refusals on ethically sensitive entities. The full pipeline runs on four model variants: LLaMA-2-13B-Chat, LLaMA-2-13B (Base), LLaMA-3-8B-Instruct, and LLaMA-3-8B (Base).

---

## Directory Structure

```
exp_4/
├── README.md
├── code/
│   ├── config.py                               # Central config (set_model, paths, constants)
│   ├── utils/
│   │   ├── utils.py                            # Shared: varimax, PCA, RDM, RSA, correlation
│   │   └── report_utils.py                     # HTML report scaffolding, CSS, figure encoding
│   ├── entities/
│   │   ├── gray_entities.py                    # Gray et al. scores, prompts, descriptions
│   │   └── characters.py                       # 30 AI/human character definitions
│   ├── comparisons/
│   │   ├── 1_behavioral_summary_figures_generator.py
│   │   └── 1a_behavioral_summary_report_generator.py
│   │
│   ├── gray_replication/                       # Gray et al. 13-entity behavioral replication
│   │   └── behavior/
│   │       ├── 1_pairwise_replication.py       # 66 pairs × 2 orders × 18 capacities
│   │       ├── 2_debiasing_reanalysis.py       # Analytical debiasing (base-only, CPU)
│   │       ├── 3_individual_ratings.py         # Likert ratings (base-only, GPU)
│   │       ├── compute_excl_pca.py, compute_human_comparisons.py
│   │       └── make_condition_reports.py, make_loadings_bar_chart.py
│   │
│   ├── gray_simple/                            # Simple entity activation extraction + RSA
│   │   └── internals/
│   │       ├── 1_extract_entity_representations.py
│   │       ├── 1a_rsa_report_generator.py
│   │       ├── 2_neural_pca.py                 # PCA + Procrustes + MDS on entity activations
│   │       └── 2a_neural_pca_report_generator.py
│   │
│   ├── human_ai_adaptation/                    # 30 AI/human characters on Gray capacities
│   │   └── behavior/
│   │       ├── 1_gray_with_characters.py       # Pairwise on 18 capacities
│   │       ├── 2_gray_names_only.py            # Same but descriptions omitted
│   │       ├── 1a_gray_chars_pca_report_generator.py
│   │       ├── 1b_gray_chars_detailed_report_generator.py
│   │       └── 1c_gray_chars_rsa_report_generator.py
│   │
│   ├── expanded_mental_concepts/               # Exp 3 bridge: 28 chars × ~27 concept dims
│   │   ├── concepts.py                         # Concept dimension definitions
│   │   ├── behavior/
│   │   │   └── pca/                            # 3 scripts + 3 report generators
│   │   └── internals/
│   │       ├── pca/                            # activation_pca, matched_activation_pca
│   │       ├── rsa/                            # activation_rsa, matched_rsa
│   │       ├── concept_rsa/                    # Per-dimension concept-specific RSA
│   │       ├── contrast_alignment/             # Exp 3 contrast vectors → character space
│   │       └── standalone_alignment/           # Exp 3 standalone vectors → character space
│   │
│   └── slurm/                                  # Consolidated SLURM scripts
│       ├── gray_replication/
│       ├── gray_simple/
│       ├── human_ai_adaptation/
│       └── expanded_mental_concepts/
│
├── results/                                    # Branch-first, model-scoped
│   ├── {llama2_13b_chat,llama2_13b_base,llama3_8b_instruct,llama3_8b_base}/
│   │   ├── gray_replication/behavior/{with,without}_self/{data,figures}/
│   │   ├── gray_simple/internals/{with,without}_self/{data,figures}/
│   │   ├── human_ai_adaptation/behavior/{data,names_only/data}/
│   │   └── expanded_mental_concepts/
│   │       ├── behavior/pca/{data,full_dataset}/
│   │       └── internals/{rsa,pca,concept_rsa,contrast_alignment,standalone_alignment}/
│   └── comparisons/figures/
├── writeup/
├── archive/
└── logs/{gray_replication,gray_simple,human_ai_adaptation,expanded_mental_concepts}/
```

---

## Scripts

All scripts use `--model {llama2_13b_chat,llama2_13b_base,llama3_8b_instruct,llama3_8b_base}`. Run from `exp_4/code/`.

### Gray Simple — Internals Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `gray_simple/internals/1_extract_entity_representations.py` | Extract last-token activations for 13 entities, compute RDMs, run RSA | Yes | `slurm/gray_simple/1_extract_entities_*.sh` |
| `gray_simple/internals/1a_rsa_report_generator.py` | HTML report: layerwise RSA profiles, RDM heatmaps | No | -- |
| `gray_simple/internals/2_neural_pca.py` | PCA + Procrustes + MDS on entity activations vs human 2D | No | `slurm/gray_simple/2_neural_pca_*.sh` |
| `gray_simple/internals/2a_neural_pca_report_generator.py` | HTML report: scree plots, PC correlations, entity scatter | No | -- |

### Gray Replication — Behavior Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `gray_replication/behavior/1_pairwise_replication.py` | Core Gray et al. replication: 66 pairs × 2 orders × 18 capacities | Yes | `slurm/gray_replication/1_pairwise_*.sh` |
| `gray_replication/behavior/2_debiasing_reanalysis.py` | Analytical debiasing + log-odds reanalysis (base only) | No | -- |
| `gray_replication/behavior/3_individual_ratings.py` | Individual Likert ratings per entity per capacity (base only) | Yes | `slurm/gray_replication/3_individual_*.sh` |

### Human-AI Adaptation — Behavior Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `human_ai_adaptation/behavior/1_gray_with_characters.py` | 30 AI/human chars on 18 Gray capacities | Yes | `slurm/human_ai_adaptation/1_gray_chars_*.sh` |
| `human_ai_adaptation/behavior/2_gray_names_only.py` | Same as 1 but descriptions omitted | Yes | `slurm/human_ai_adaptation/2_gray_names_only_*.sh` |
| `human_ai_adaptation/behavior/1a-1c_*_report_generator.py` | PCA, detailed, and RSA reports | No | -- |

### Expanded Mental Concepts Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `expanded_mental_concepts/behavior/pca/behavioral_pca.py` | 28 chars × ~27 concepts pairwise | Yes | `slurm/expanded_mental_concepts/behavioral_pca_*.sh` |
| `expanded_mental_concepts/internals/rsa/activation_rsa.py` | Activation RSA for 28 characters | Yes | `slurm/expanded_mental_concepts/activation_rsa_*.sh` |
| `expanded_mental_concepts/internals/concept_rsa/concept_rsa.py` | Per-dimension concept-specific RSA | Yes | `slurm/expanded_mental_concepts/concept_rsa_*.sh` |
| `expanded_mental_concepts/internals/contrast_alignment/contrast_alignment.py` | Exp 3 contrast vectors → character space | Yes | `slurm/expanded_mental_concepts/contrast_alignment_*.sh` |
| `expanded_mental_concepts/internals/standalone_alignment/standalone_alignment.py` | Exp 3 standalone vectors → character space | Yes | `slurm/expanded_mental_concepts/standalone_alignment_*.sh` |

---

## Execution

```bash
# From exp_4/code/

# === Gray Simple (Internals) ===
sbatch slurm/gray_simple/1_extract_entities_chat.sh
sbatch slurm/gray_simple/1_extract_entities_base.sh
# After completion:
python gray_simple/internals/1a_rsa_report_generator.py --model llama2_13b_chat
python gray_simple/internals/1a_rsa_report_generator.py --model llama2_13b_base
# Neural PCA (CPU-only, reads saved activations)
sbatch slurm/gray_simple/2_neural_pca_chat.sh
sbatch slurm/gray_simple/2_neural_pca_base.sh
# After completion:
python gray_simple/internals/2a_neural_pca_report_generator.py --model llama2_13b_chat
python gray_simple/internals/2a_neural_pca_report_generator.py --model llama2_13b_base

# === Gray Replication (Behavior) ===
sbatch slurm/gray_replication/1_pairwise_chat.sh
sbatch slurm/gray_replication/1_pairwise_base.sh
python gray_replication/behavior/2_debiasing_reanalysis.py --model llama2_13b_base --both
sbatch slurm/gray_replication/3_individual_base.sh

# === Human-AI Adaptation (Behavior) ===
sbatch slurm/human_ai_adaptation/1_gray_chars_chat.sh
sbatch slurm/human_ai_adaptation/1_gray_chars_base.sh
sbatch slurm/human_ai_adaptation/2_gray_names_only_chat.sh

# === Expanded Mental Concepts ===
sbatch slurm/expanded_mental_concepts/behavioral_pca_chat.sh
sbatch slurm/expanded_mental_concepts/behavioral_pca_base.sh
sbatch slurm/expanded_mental_concepts/activation_rsa_chat.sh
sbatch slurm/expanded_mental_concepts/activation_rsa_base.sh
sbatch slurm/expanded_mental_concepts/concept_rsa_chat.sh
sbatch slurm/expanded_mental_concepts/concept_rsa_base.sh
sbatch slurm/expanded_mental_concepts/contrast_alignment_chat.sh
sbatch slurm/expanded_mental_concepts/contrast_alignment_base.sh
sbatch slurm/expanded_mental_concepts/standalone_alignment_chat.sh
sbatch slurm/expanded_mental_concepts/standalone_alignment_base.sh

# === LLaMA-3-8B Replication ===
# Gray Simple
sbatch slurm/gray_simple/1_extract_entities_llama3_instruct.sh
sbatch slurm/gray_simple/1_extract_entities_llama3_base.sh
python gray_simple/internals/1a_rsa_report_generator.py --model llama3_8b_instruct
python gray_simple/internals/1a_rsa_report_generator.py --model llama3_8b_base
sbatch slurm/gray_simple/2_neural_pca_llama3_instruct.sh
sbatch slurm/gray_simple/2_neural_pca_llama3_base.sh
python gray_simple/internals/2a_neural_pca_report_generator.py --model llama3_8b_instruct
python gray_simple/internals/2a_neural_pca_report_generator.py --model llama3_8b_base

# Gray Replication
sbatch slurm/gray_replication/1_pairwise_llama3_instruct.sh
sbatch slurm/gray_replication/1_pairwise_llama3_base.sh
sbatch slurm/gray_replication/3_individual_llama3_base.sh

# Human-AI Adaptation
sbatch slurm/human_ai_adaptation/1_gray_chars_llama3_instruct.sh
sbatch slurm/human_ai_adaptation/1_gray_chars_llama3_base.sh

# Expanded Mental Concepts
sbatch slurm/expanded_mental_concepts/behavioral_pca_llama3_instruct.sh
sbatch slurm/expanded_mental_concepts/behavioral_pca_llama3_base.sh
sbatch slurm/expanded_mental_concepts/activation_rsa_llama3_instruct.sh
sbatch slurm/expanded_mental_concepts/activation_rsa_llama3_base.sh
sbatch slurm/expanded_mental_concepts/concept_rsa_llama3_instruct.sh
sbatch slurm/expanded_mental_concepts/concept_rsa_llama3_base.sh
sbatch slurm/expanded_mental_concepts/contrast_alignment_llama3_instruct.sh
sbatch slurm/expanded_mental_concepts/contrast_alignment_llama3_base.sh
sbatch slurm/expanded_mental_concepts/standalone_alignment_llama3_instruct.sh
sbatch slurm/expanded_mental_concepts/standalone_alignment_llama3_base.sh

# === Cross-model comparison ===
python comparisons/1_behavioral_summary_figures_generator.py
python comparisons/1a_behavioral_summary_report_generator.py
```

---

## Human Ground Truth

Human factor scores from Gray et al. (2007, Figure 1), estimated on a 0-1 scale. **These values were estimated from the published figure and should be verified before publication** (e.g., digitize Figure 1 with WebPlotDigitizer or contact Kurt Gray at UNC Chapel Hill).

| Entity | Experience | Agency |
|--------|----------:|-------:|
| Dead woman | 0.06 | 0.07 |
| Robot (Kismet) | 0.13 | 0.22 |
| Fetus (7 wk) | 0.17 | 0.08 |
| PVS patient | 0.17 | 0.10 |
| God | 0.20 | 0.80 |
| Frog | 0.25 | 0.14 |
| Dog | 0.55 | 0.35 |
| Chimpanzee | 0.63 | 0.48 |
| Baby (5 mo) | 0.71 | 0.17 |
| Girl (5 yo) | 0.84 | 0.62 |
| Adult man | 0.91 | 0.95 |
| Adult woman | 0.93 | 0.91 |
| You (self) | 0.97 | 1.00 |

---

## Environment

- **Models:** LLaMA-2-13B (base + chat), LLaMA-3-8B (base + instruct)
- **Cluster:** Princeton HPC (Scotty), SLURM scheduler
- **Conda env:** `llama2_env` (GPU phases: model loading, forward passes)
- **GPU:** `--gres=gpu:1 --mem=48G` (LLaMA-2-13B), `--mem=32G` (LLaMA-3-8B)

---

## References

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619. https://doi.org/10.1126/science.1134475
