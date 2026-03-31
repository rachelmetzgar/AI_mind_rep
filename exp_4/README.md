# Experiment 4: Implicit Folk Psychology Across Architectures

**Author:** Rachel C. Metzgar, Princeton University

## Reports

### Cross-Model Comparisons (11 models)
- [Results comparison](results/comparisons/results_comparison.html) -- Cross-branch synthesis with publication figures
- [Gray entities behavioral](results/comparisons/gray_replication_summary.html) -- PCA, human correlations, factor structure
- [Gray entities neural](results/comparisons/gray_simple_summary.html) -- Neural RSA, Procrustes alignment
- [Human-AI characters behavioral](results/comparisons/human_ai_summary.html) -- 30-character categorical separation
- [Human-AI characters concepts](results/comparisons/expanded_concepts_summary.html) -- Concept RSA, alignment
- [Status report](results/comparisons/status_report.html) -- Data file audit across all models

---

## Motivation

Experiments 1-3 treat partner identity as a binary (human vs. AI). Human folk psychology is far richer. Gray, Gray, & Wegner (2007, *Science*) showed that humans perceive minds along two orthogonal dimensions: **Experience** (the capacity to feel -- hunger, fear, pain, pleasure, joy) and **Agency** (the capacity to plan and act -- self-control, morality, memory, planning, thought). ~2,400 participants rated 13 diverse entities via pairwise comparisons on 18 mental capacities, and PCA with varimax rotation recovered this two-factor structure, explaining 97% of variance.

Exp 4 asks: **do large language models have an implicit folk psychology of mind that mirrors this human structure?** If the model's representational geometry over diverse entities resembles the human Experience/Agency space, it would suggest the model has internalized a continuous, multi-dimensional folk psychology.

---

## Design

Two entity sets, each with behavioral and neural analyses:

| Entity Set | Entities | Behavioral Analyses | Neural Analyses |
|-----------|---------|-------------------|----------------|
| Gray entities | 13 from Gray et al. (baby, dog, robot, God, etc.) | Pairwise on 18 capacities, individual Likert | RSA + PCA on "Think about {entity}" activations |
| Human-AI characters | 30 (15 AI + 15 human) | Pairwise on 18 capacities + 22 concept dims | RSA + PCA on character activations, concept RSA, Exp 3 alignment |

**Conditions:**
- Gray entities: `with_self` (13 entities) / `without_self` (12, drops "you")
- Human-AI behavioral: `full_description` (character bios) / `names_only` (tests prior knowledge)
- Human-AI neural: `names_only` (current) / `full_description` (planned)

**11 models** across 4 families:

| Family | Instruct/Chat | Base | Params | Hidden Dim | Layers |
|--------|--------------|------|--------|-----------|--------|
| LLaMA-2 | llama2_13b_chat | llama2_13b_base | 13B | 5120 | 40 |
| LLaMA-3 | llama3_8b_instruct | llama3_8b_base | 8B | 4096 | 32 |
| Gemma-2 (small) | gemma2_2b_it | gemma2_2b | 2B | 2304 | 26 |
| Gemma-2 (large) | gemma2_9b_it | gemma2_9b | 9B | 3584 | 42 |
| Qwen-2.5 | qwen25_7b_instruct | qwen25_7b | 7B | 3584 | 28 |
| Qwen3 | qwen3_8b | -- | 8B | 4096 | 36 |

---

## Directory Structure

```
exp_4/
├── code/
│   ├── config.py
│   ├── entities/                        # Entity/character definitions
│   ├── utils/                           # Shared analysis + report utilities
│   ├── expanded_mental_concepts/
│   │   └── concepts.py                  # Exp 3 concept dimension definitions
│   │
│   ├── gray_entities/
│   │   ├── behavioral/                  # 1-7: pairwise, debiasing, individual, RSA, reports
│   │   └── neural/                      # 1-2: activation extraction, RSA, PCA
│   │
│   ├── human_ai_characters/
│   │   ├── behavioral/
│   │   │   ├── gray_capacities/         # 1-2: pairwise (full desc + names only), reports
│   │   │   └── expanded_concepts/       # 1-2: concept PCA, matched PCA, reports
│   │   └── neural/
│   │       ├── 1_extract_character_activations.py
│   │       ├── 2_activation_pca.py
│   │       ├── 3_concept_rsa.py
│   │       ├── 4_contrast_alignment.py
│   │       └── 5_standalone_alignment.py
│   │
│   ├── comparisons/                     # Cross-model summary generators (2-7)
│   └── slurm/
│       ├── gray_entities/
│       └── human_ai_characters/
│
├── results/
│   ├── {model}/
│   │   ├── gray_entities/
│   │   │   ├── behavioral/{with,without}_self/data/
│   │   │   └── neural/{with,without}_self/data/
│   │   └── human_ai_characters/
│   │       ├── behavioral/
│   │       │   ├── gray_capacities/{full_description,names_only}/data/
│   │       │   └── expanded_concepts/full_description/data/
│   │       └── neural/names_only/
│   │           ├── rsa_pca/data/
│   │           ├── concept_rsa/{concept}/data/
│   │           └── alignment/{contrast,standalone}/data/
│   └── comparisons/
├── writeup/
└── archive/
```

---

## Scripts

All scripts use `--model <model_key>`. Run from `exp_4/code/`.

### Gray Entities -- Behavioral

| Script | Description | GPU |
|--------|-------------|-----|
| `gray_entities/behavioral/1_pairwise_replication.py` | 78 pairs x 2 orders x 18 capacities | Yes |
| `gray_entities/behavioral/2_debiasing_reanalysis.py` | Analytical debiasing (base only) | No |
| `gray_entities/behavioral/3_individual_ratings.py` | Likert ratings per entity per capacity | Yes |
| `gray_entities/behavioral/4_behavioral_rsa.py` | RSA on behavioral rating geometry | No |
| `gray_entities/behavioral/5_compute_excl_pca.py` | PCA excluding outliers (fetus, god, dead_woman) | No |
| `gray_entities/behavioral/6_compute_human_comparisons.py` | Spearman, Procrustes, RV vs human data | No |
| `gray_entities/behavioral/7_condition_report_generator.py` | Per-condition HTML reports | No |

### Gray Entities -- Neural

| Script | Description | GPU |
|--------|-------------|-----|
| `gray_entities/neural/1_extract_entity_representations.py` | Extract activations, compute RDMs, RSA | Yes |
| `gray_entities/neural/2_neural_pca.py` | PCA + Procrustes alignment to human 2D | No |

### Human-AI Characters -- Behavioral

| Script | Description | GPU |
|--------|-------------|-----|
| `human_ai_characters/behavioral/gray_capacities/1_gray_with_characters.py` | 30 chars on 18 capacities (full descriptions) | Yes |
| `human_ai_characters/behavioral/gray_capacities/2_gray_names_only.py` | Same with names only | Yes |
| `human_ai_characters/behavioral/expanded_concepts/1_behavioral_pca.py` | 30 chars on 22 concept dimensions | Yes |
| `human_ai_characters/behavioral/expanded_concepts/2_matched_behavioral_pca.py` | Matched concept subsets | No |

### Human-AI Characters -- Neural

| Script | Description | GPU |
|--------|-------------|-----|
| `human_ai_characters/neural/1_extract_character_activations.py` | Extract activations for 30 characters | Yes |
| `human_ai_characters/neural/2_activation_pca.py` | PCA + Procrustes on character activations | No |
| `human_ai_characters/neural/3_concept_rsa.py` | Per-concept contextualized RSA (22 concepts) | Yes |
| `human_ai_characters/neural/4_contrast_alignment.py` | Exp 3 contrast vector projection | Yes |
| `human_ai_characters/neural/5_standalone_alignment.py` | Exp 3 standalone vector projection | Yes |

---

## Execution

```bash
# From exp_4/code/
MODEL=gemma2_9b_it

# Gray entities
sbatch slurm/gray_entities/1_pairwise_${MODEL}.sh
sbatch slurm/gray_entities/1_extract_entities_${MODEL}.sh
sbatch slurm/gray_entities/2_neural_pca_${MODEL}.sh
sbatch slurm/gray_entities/3_individual_${MODEL}.sh

# Human-AI characters
sbatch slurm/human_ai_characters/1_gray_chars_${MODEL}.sh
sbatch slurm/human_ai_characters/2_gray_names_only_${MODEL}.sh
sbatch slurm/human_ai_characters/behavioral_pca_${MODEL}.sh
sbatch slurm/human_ai_characters/activation_rsa_${MODEL}.sh
sbatch slurm/human_ai_characters/concept_rsa_${MODEL}.sh
sbatch slurm/human_ai_characters/contrast_alignment_${MODEL}.sh
sbatch slurm/human_ai_characters/standalone_alignment_${MODEL}.sh

# Post-processing (CPU)
python gray_entities/behavioral/5_compute_excl_pca.py --model $MODEL
python gray_entities/behavioral/6_compute_human_comparisons.py --model $MODEL

# Cross-model reports (after all models complete)
python comparisons/3_gray_replication_summary_generator.py
python comparisons/4_gray_simple_summary_generator.py
python comparisons/5_human_ai_summary_generator.py
python comparisons/6_expanded_concepts_summary_generator.py
python comparisons/7_results_comparison_generator.py
```

---

## Human Ground Truth

Human factor scores from Gray et al. (2007, Figure 1), estimated on a 0-1 scale. **These values were estimated from the published figure and should be verified before publication.**

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

- **Cluster:** Princeton HPC (Scotty), SLURM scheduler
- **Conda env:** `llama2_env` (all phases)
- **GPU:** `--gres=gpu:1` with model-appropriate `--mem`: 64G (13B models), 48G (7-8B), 32G (2B)

---

## References

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619. https://doi.org/10.1126/science.1134475
