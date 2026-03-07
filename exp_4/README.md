# Experiment 4: Mind Perception Geometry

**Author:** Rachel C. Metzgar, Princeton University

## Reports

- [Results walkthrough](archive/exp4_results_walkthrough.html) — Combined results across both models: factor structure, human correlations, entity placements, RSA
- [Base model results](results/llama2_13b_base/behavior/results_report.html) — Pairwise and individual rating analyses for the base model
- [Analysis explainer](results/llama2_13b_base/behavior/analysis_explainer.html) — Step-by-step walkthrough of the PCA / varimax / factor-score methodology
- [Chat RSA report](results/llama2_13b_chat/internals/rsa_report.html) — RSA-by-dimension analysis for the chat model
- [Base RSA report](results/llama2_13b_base/internals/rsa_report.html) — RSA-by-dimension analysis for the base model

---

## Motivation

Experiments 1-3 treat partner identity as a binary (human vs. AI). Human folk psychology is far richer. Gray, Gray, & Wegner (2007, *Science*) showed that humans perceive minds along two orthogonal dimensions: **Experience** (the capacity to feel -- hunger, fear, pain, pleasure, joy) and **Agency** (the capacity to plan and act -- self-control, morality, memory, planning, thought). ~2,400 participants rated 13 diverse entities via pairwise comparisons on 18 mental capacities, and PCA with varimax rotation recovered this two-factor structure, explaining 97% of variance.

Exp 4 asks: **does LLaMA-2-13B have an implicit folk psychology of mind that mirrors this human structure?** If the model's representational geometry over diverse entities (baby, dog, robot, God, adults, etc.) resembles the human Experience/Agency space, it would suggest the model has internalized a continuous, multi-dimensional folk psychology -- not just a binary human/AI switch.

---

## Design

Behavioral replication of Gray et al. (2007):
- **13 entities** from the original study: frog, dog, chimpanzee, 7-week fetus, 5-month baby, 5-year-old girl, adult woman, adult man, PVS patient, dead woman, God, robot (Kismet), and "you (yourself)"
- **18 mental capacities**: 11 Experience items (hunger, fear, pain, pleasure, rage, desire, personality, consciousness, pride, embarrassment, joy) + 7 Agency items (self-control, morality, memory, emotion recognition, planning, communication, thought)
- Verbatim character descriptions and survey prompts from Gray et al. Appendix A/B
- Pairwise comparisons on a 5-point scale, counterbalanced across both presentation orders
- PCA with varimax rotation to recover factor structure
- Spearman correlation of model factor scores with human Experience/Agency scores

Both a chat and base model variant are tested, since the chat model's RLHF safety training causes refusals on ethically sensitive entities.

---

## Directory Structure

```
exp_4/
├── README.md
├── archive/                                    # Old structure preserved (browsable)
│   ├── pre_refactor/                           # Pre-refactoring snapshot
│   ├── llama_exp_4-13B-chat/
│   ├── llama_exp_4-13B-base/
│   ├── write_up/
│   └── exp4_results_walkthrough.html
├── code/
│   ├── config.py                               # Central config (set_model, paths, constants)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py                            # Shared: varimax, PCA, RDM, RSA, correlation
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── gray_entities.py                    # Gray et al. scores, prompts, descriptions
│   │   └── gray2007.txt
│   ├── internals/                              # Activation extraction + RSA pipeline
│   │   ├── 1_extract_entity_representations.py
│   │   ├── 1a_rsa_report_generator.py
│   │   └── slurm/
│   │       ├── 1_extract_entities_chat.sh
│   │       └── 1_extract_entities_base.sh
│   ├── behavior/                               # Behavioral replication pipeline
│   │   ├── 1_pairwise_replication.py
│   │   ├── 2_debiasing_reanalysis.py           # Base-only, CPU
│   │   ├── 3_individual_ratings.py             # Base-only, GPU
│   │   └── slurm/
│   │       ├── 1_pairwise_chat.sh
│   │       ├── 1_pairwise_base.sh
│   │       └── 3_individual_base.sh
│   └── comparisons/                            # Cross-model analyses
│       └── 1_behavioral_summary_figures_generator.py
├── results/                                    # Model-first layout
│   ├── llama2_13b_chat/
│   │   ├── internals/
│   │   │   ├── rsa_report.html
│   │   │   ├── with_self/{data,figures}/
│   │   │   └── without_self/{data,figures}/
│   │   └── behavior/
│   │       ├── with_self/{data}/
│   │       └── without_self/{data}/
│   ├── llama2_13b_base/
│   │   ├── internals/
│   │   │   ├── rsa_report.html
│   │   │   ├── with_self/{data,figures}/
│   │   │   └── without_self/{data,figures}/
│   │   └── behavior/
│   │       ├── results_report.html
│   │       ├── analysis_explainer.html
│   │       ├── with_self/{data}/
│   │       └── without_self/{data}/
│   └── comparisons/
│       └── figures/                            # 10 cross-model publication figures
├── writeup/
│   ├── exp4_methods.html
│   └── exp5_plans.md
└── logs/
    ├── internals/                              # SLURM logs for extraction/RSA
    └── behavior/                               # SLURM logs for behavioral runs
```

---

## Scripts

All scripts use `--model llama2_13b_chat|llama2_13b_base` to select the model variant. Run from `exp_4/code/`.

### Internals Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `internals/1_extract_entity_representations.py` | Extract last-token activations for 13 entities, compute RDMs, run RSA at every layer (3 variants: combined, experience, agency) | Yes | `internals/slurm/1_extract_entities_{chat,base}.sh` |
| `internals/1a_rsa_report_generator.py` | Generate HTML report with FDR correction, layerwise profiles, RDM heatmaps | No | -- |

### Behavior Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `behavior/1_pairwise_replication.py` | Pairwise comparisons (66 pairs x 2 orders x 18 capacities). Chat: text generation. Base: logit extraction. PCA + varimax + human correlation. | Yes | `behavior/slurm/1_pairwise_{chat,base}.sh` |
| `behavior/2_debiasing_reanalysis.py` | Analytical debiasing + log-odds reanalysis of pairwise data (base only) | No | -- |
| `behavior/3_individual_ratings.py` | Individual Likert ratings per entity per capacity (base only) | Yes | `behavior/slurm/3_individual_base.sh` |

### Comparisons

| Script | Description |
|--------|-------------|
| `comparisons/1_behavioral_summary_figures_generator.py` | 10 cross-model publication figures (scree plots, loading comparisons, entity scatter, mind space maps, heatmaps, RSA profiles, correlation summaries, RDM comparisons) |

### Common Arguments

- `--model llama2_13b_chat|llama2_13b_base` — Required. Selects model variant.
- `--include_self` — Include "you_self" entity (default: exclude, giving 12 entities).
- `--both` — Run both with_self and without_self conditions.

---

## Execution

```bash
# From exp_4/code/

# Phase 1: Extract entity representations + RSA
sbatch internals/slurm/1_extract_entities_chat.sh
sbatch internals/slurm/1_extract_entities_base.sh

# Phase 1a: Generate RSA reports (CPU, run directly)
python internals/1a_rsa_report_generator.py --model llama2_13b_chat
python internals/1a_rsa_report_generator.py --model llama2_13b_base

# Phase 2: Behavioral replication
sbatch behavior/slurm/1_pairwise_chat.sh
sbatch behavior/slurm/1_pairwise_base.sh

# Phase 2b: Debiasing reanalysis (CPU, run directly)
python behavior/2_debiasing_reanalysis.py --model llama2_13b_base --both

# Phase 2c: Individual ratings
sbatch behavior/slurm/3_individual_base.sh

# Phase 3: Cross-model figures (CPU, run directly)
python comparisons/1_behavioral_summary_figures_generator.py
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

- **Model:** LLaMA-2-13B (base and chat variants)
- **Cluster:** Princeton HPC (Scotty), SLURM scheduler
- **Conda env:** `llama2_env` (GPU phases: model loading, forward passes)
- **GPU:** `--gres=gpu:1 --mem=48G` (~26GB VRAM for fp16)

---

## References

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619. https://doi.org/10.1126/science.1134475
