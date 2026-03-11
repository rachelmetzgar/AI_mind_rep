# Experiment 4: Mind Perception Geometry

**Author:** Rachel C. Metzgar, Princeton University

## Reports

### Core Pipeline
- [Cross-model behavioral summary](results/comparisons/behavioral_summary_report.html) — 10 publication figures comparing both models: factor structure, human correlations, entity placements, RSA
- [Base model results](results/llama2_13b_base/behavior/results_report.html) — Pairwise and individual rating analyses for the base model
- [Analysis explainer](results/llama2_13b_base/behavior/analysis_explainer.html) — Step-by-step walkthrough of the PCA / varimax / factor-score methodology
- [Chat RSA report](results/llama2_13b_chat/internals/full_dataset/rsa_report.html) — RSA-by-dimension analysis for the chat model
- [Base RSA report](results/llama2_13b_base/internals/rsa_report.html) — RSA-by-dimension analysis for the base model

### Gray-with-Characters (30 AI/human characters)
- [Base PCA report](results/llama2_13b_base/behavior/gray_characters/full_dataset/gray_chars_pca_report.html) — PCA on 30 characters rated on 18 Gray capacities
- [Base detailed report](results/llama2_13b_base/behavior/gray_characters/full_dataset/gray_chars_detailed_report.html) — Per-character ratings breakdown
- [Base RSA report](results/llama2_13b_base/behavior/gray_characters/full_dataset/gray_chars_rsa_report.html) — Character activation RSA

### Concept Geometry (Exp 3 bridge)
- [Base behavioral PCA](results/llama2_13b_base/concept_geometry/pca/behavioral/full_dataset/) — 4 reports: behavioral PCA, matched PCA, attribution analysis, detailed responses
- [Base activation RSA](results/llama2_13b_base/concept_geometry/rsa/activation/full_dataset/activation_rsa_report.html) — Activation-space RSA for 28 characters
- [Chat activation RSA](results/llama2_13b_chat/concept_geometry/rsa/activation/full_dataset/activation_rsa_report.html) — Chat model activation RSA
- [Base contrast alignment](results/llama2_13b_base/concept_geometry/rsa/contrast_alignment/contrast_alignment_report.html) — Exp 3 contrast vectors projected onto character space
- [Chat contrast alignment](results/llama2_13b_chat/concept_geometry/rsa/contrast_alignment/contrast_alignment_report.html) — Chat model contrast alignment
- [Base standalone alignment](results/llama2_13b_base/concept_geometry/rsa/standalone_alignment/standalone_alignment_report.html) — Exp 3 standalone vectors projected onto character space
- [Chat standalone alignment](results/llama2_13b_chat/concept_geometry/rsa/standalone_alignment/standalone_alignment_report.html) — Chat model standalone alignment
- [Base concept RSA summary](results/llama2_13b_base/concept_geometry/rsa/concept/data/cross_concept_rsa_summary.md) — Per-dimension concept-specific RSA
- [Chat concept RSA summary](results/llama2_13b_chat/concept_geometry/rsa/concept/data/cross_concept_rsa_summary.md) — Chat model concept RSA

### Archive
- [Results walkthrough (pre-refactor)](archive/exp4_results_walkthrough.html) — Combined results across both models (legacy format)

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

### Bridge to human and AI entities
How are different humans and AIs represented along these dimensions?
TODO: more details

### Concept Geometry (Exp 3 Bridge)

Tests whether Exp 3 mental-property concept dimensions organize Exp 4's richer entity space. Uses 28 characters (14 AI + 14 human) and ~27 concept dimensions auto-discovered from `exp_3/concepts/standalone/`.
TODO: more details

---

## Directory Structure

```
exp_4/
├── README.md
├── archive/                                    # Old structure preserved (browsable)
│   ├── pre_refactor/
│   ├── llama_exp_4-13B-chat/
│   ├── llama_exp_4-13B-base/
│   ├── write_up/
│   └── exp4_results_walkthrough.html
├── code/
│   ├── config.py                               # Central config (set_model, paths, constants)
│   ├── utils/
│   │   ├── utils.py                            # Shared: varimax, PCA, RDM, RSA, correlation
│   │   └── report_utils.py                     # HTML report scaffolding, CSS, figure encoding
│   ├── entities/
│   │   ├── gray_entities.py                    # Gray et al. scores, prompts, descriptions
│   │   └── gray2007.txt
│   ├── internals/                              # Activation extraction + RSA
│   │   ├── 1_extract_entity_representations.py
│   │   ├── 1a_rsa_report_generator.py
│   │   └── slurm/
│   ├── behavior/                               # Behavioral replication
│   │   ├── 1_pairwise_replication.py           # Core Gray et al. pairwise (both models)
│   │   ├── 2_debiasing_reanalysis.py           # Analytical debiasing (base-only, CPU)
│   │   ├── 3_individual_ratings.py             # Likert ratings (base-only, GPU)
│   │   ├── 4_gray_with_characters.py           # 30 AI/human chars on 18 capacities
│   │   ├── 5_gray_names_only.py                # Same but descriptions omitted
│   │   ├── 4a_gray_chars_pca_report_generator.py
│   │   ├── 4b_gray_chars_detailed_report_generator.py
│   │   ├── 4c_gray_chars_rsa_report_generator.py
│   │   ├── compute_excl_pca.py                 # Re-run PCA excluding fetus/god
│   │   ├── compute_human_comparisons.py        # Stat comparisons with Gray et al.
│   │   ├── make_condition_reports.py           # Per-condition HTML reports
│   │   ├── make_loadings_bar_chart.py          # Varimax loading visualizations
│   │   └── slurm/
│   ├── concept_geometry/                       # Cross-experiment bridge with Exp 3
│   │   ├── characters.py                       # 15 AI + 15 human character definitions
│   │   ├── concepts.py                         # Auto-discovers Exp 3 standalone dims
│   │   ├── pca/
│   │   │   ├── behavioral_pca.py               # Phase A: 28 chars × ~27 concepts pairwise
│   │   │   ├── matched_behavioral_pca.py       # Matched-pair subset reanalysis
│   │   │   ├── activation_pca.py               # Phase B: PCA on character activations
│   │   │   ├── matched_activation_pca.py       # Matched-pair activation reanalysis
│   │   │   ├── behavioral_pca_report_generator.py
│   │   │   ├── behavioral_attribution_report_generator.py
│   │   │   ├── matched_behavioral_pca_report_generator.py
│   │   │   ├── activation_pca_report_generator.py
│   │   │   └── detailed_response_report_generator.py
│   │   ├── rsa/
│   │   │   ├── activation_rsa.py               # Activation RSA for 28 characters
│   │   │   ├── matched_rsa.py                  # Matched-pair RSA
│   │   │   ├── concept_rsa.py                  # Concept-specific RSA per dimension
│   │   │   ├── contrast_alignment.py           # Exp 3 contrast vectors → character space
│   │   │   ├── standalone_alignment.py         # Exp 3 standalone vectors → character space
│   │   │   ├── activation_rsa_report_generator.py
│   │   │   ├── contrast_alignment_report_generator.py
│   │   │   └── standalone_alignment_report_generator.py
│   │   └── slurm/                              # 10 SLURM scripts (5 analyses × 2 models)
│   └── comparisons/
│       ├── 1_behavioral_summary_figures_generator.py   # 10 cross-model pub figures
│       └── 1a_behavioral_summary_report_generator.py   # Wraps into HTML report
├── results/                                    # Model-first layout
│   ├── llama2_13b_chat/
│   │   ├── internals/
│   │   │   ├── full_dataset/rsa_report.html
│   │   │   ├── with_self/data/
│   │   │   └── without_self/data/
│   │   ├── behavior/
│   │   │   ├── with_self/data/
│   │   │   ├── without_self/data/
│   │   │   ├── gray_characters/data/
│   │   │   └── names_only/data/
│   │   └── concept_geometry/
│   │       ├── pca/behavioral/data/
│   │       └── rsa/
│   │           ├── activation/{data,full_dataset/}
│   │           ├── concept/{22 concept dims}/data/  # per-dim RSA
│   │           ├── contrast_alignment/{data,contrast_alignment_report.html}
│   │           └── standalone_alignment/{data,standalone_alignment_report.html}
│   ├── llama2_13b_base/
│   │   ├── internals/
│   │   │   ├── rsa_report.html
│   │   │   ├── with_self/{data,figures}/
│   │   │   └── without_self/{data,figures}/
│   │   ├── behavior/
│   │   │   ├── results_report.html
│   │   │   ├── analysis_explainer.html
│   │   │   ├── with_self/{data,figures}/
│   │   │   ├── without_self/{data,figures}/
│   │   │   ├── gray_characters/{data,full_dataset/}  # 3 HTML reports
│   │   │   └── names_only/data/
│   │   └── concept_geometry/
│   │       ├── pca/
│   │       │   ├── behavioral/{data,full_dataset/}    # 4 HTML reports
│   │       │   └── activation/data/
│   │       └── rsa/
│   │           ├── activation/{data,full_dataset/}
│   │           ├── concept/{22 concept dims}/data/    # per-dim RSA
│   │           ├── contrast_alignment/{data,contrast_alignment_report.html}
│   │           └── standalone_alignment/{data,standalone_alignment_report.html}
│   └── comparisons/
│       ├── behavioral_summary_report.html
│       └── figures/                            # 10 cross-model publication figures
├── writeup/
│   ├── exp4_methods.html
│   └── exp5_plans.md
└── logs/
    ├── internals/
    ├── behavior/
    └── concept_geometry/
```

---

## Scripts

All scripts use `--model llama2_13b_chat|llama2_13b_base` to select the model variant. Run from `exp_4/code/`.

### Common Arguments

- `--model llama2_13b_chat|llama2_13b_base` — Required. Selects model variant.
- `--include_self` — Include "you_self" entity (default: exclude, giving 12 entities).
- `--both` — Run both with_self and without_self conditions.

### Internals Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `internals/1_extract_entity_representations.py` | Extract last-token activations for 13 entities, compute RDMs, run RSA at every layer (3 variants: combined, experience, agency) | Yes | `internals/slurm/1_extract_entities_{chat,base}.sh` |
| `internals/1a_rsa_report_generator.py` | HTML report with FDR correction, layerwise profiles, RDM heatmaps | No | -- |

### Behavior Pipeline

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `behavior/1_pairwise_replication.py` | Core Gray et al. replication: 66 pairs × 2 orders × 18 capacities. Chat: text gen + parsing. Base: logit extraction. PCA + varimax + human correlation | Yes | `behavior/slurm/1_pairwise_{chat,base}.sh` |
| `behavior/2_debiasing_reanalysis.py` | Analytical debiasing + log-odds reanalysis (base only) | No | -- |
| `behavior/3_individual_ratings.py` | Individual Likert ratings per entity per capacity (base only) | Yes | `behavior/slurm/3_individual_base.sh` |
| `behavior/4_gray_with_characters.py` | Gray et al. replication with 30 AI/human characters instead of 13 entities | Yes | `behavior/slurm/4_gray_chars_{chat,base}.sh` |
| `behavior/5_gray_names_only.py` | Same as 4 but character descriptions omitted, names only | Yes | `behavior/slurm/5_gray_names_only_chat.sh` |
| `behavior/4a_gray_chars_pca_report_generator.py` | PCA report for gray-with-characters | No | -- |
| `behavior/4b_gray_chars_detailed_report_generator.py` | Per-character ratings breakdown | No | -- |
| `behavior/4c_gray_chars_rsa_report_generator.py` | Character activation RSA report | No | -- |
| `behavior/compute_excl_pca.py` | Re-run PCA excluding fetus and god | No | -- |
| `behavior/compute_human_comparisons.py` | Statistical comparisons with Gray et al. human scores | No | -- |
| `behavior/make_condition_reports.py` | Per-condition HTML reports (4 conditions × 2 models) | No | -- |
| `behavior/make_loadings_bar_chart.py` | Varimax-rotated capacity loadings bar chart | No | -- |

### Concept Geometry Pipeline (Exp 3 Bridge)

Tests whether Exp 3 mental-property concept dimensions organize Exp 4's richer entity space. Uses 28 characters (14 AI + 14 human) and ~27 concept dimensions auto-discovered from `exp_3/concepts/standalone/`.

**Phase A — Behavioral PCA:**

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `concept_geometry/pca/behavioral_pca.py` | 28 chars × ~27 concepts, pairwise comparisons, PCA + varimax, AI/human group analysis | Yes | `concept_geometry/slurm/behavioral_pca_{chat,base}.sh` |
| `concept_geometry/pca/matched_behavioral_pca.py` | Matched-pair subset reanalysis | No | -- |
| `concept_geometry/pca/behavioral_pca_report_generator.py` | Behavioral PCA HTML report | No | -- |
| `concept_geometry/pca/behavioral_attribution_report_generator.py` | Cross-type pairwise analysis (human vs AI per concept) | No | -- |
| `concept_geometry/pca/matched_behavioral_pca_report_generator.py` | Matched PCA HTML report | No | -- |
| `concept_geometry/pca/detailed_response_report_generator.py` | Per-character response breakdown | No | -- |

**Phase B — Activation Geometry:**

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `concept_geometry/pca/activation_pca.py` | PCA on character hidden-state activations | Yes | -- |
| `concept_geometry/pca/matched_activation_pca.py` | Matched-pair activation PCA | No | -- |
| `concept_geometry/pca/activation_pca_report_generator.py` | Activation PCA HTML report | No | -- |
| `concept_geometry/rsa/activation_rsa.py` | Extract activations for 28 chars, compute RDMs per layer, RSA vs behavioral RDMs | Yes | `concept_geometry/slurm/activation_rsa_{chat,base}.sh` |
| `concept_geometry/rsa/matched_rsa.py` | Matched-pair activation RSA | No | -- |
| `concept_geometry/rsa/activation_rsa_report_generator.py` | Activation RSA HTML report | No | -- |

**Phase C — Concept Alignment:**

| Script | Description | GPU | SLURM |
|--------|-------------|-----|-------|
| `concept_geometry/rsa/concept_rsa.py` | Per-dimension concept-specific RSA | Yes | `concept_geometry/slurm/concept_rsa_{chat,base}.sh` |
| `concept_geometry/rsa/contrast_alignment.py` | Project Exp 3 contrast vectors onto character activation space | Yes | `concept_geometry/slurm/contrast_alignment_{chat,base}.sh` |
| `concept_geometry/rsa/standalone_alignment.py` | Project Exp 3 standalone vectors onto character activation space | Yes | `concept_geometry/slurm/standalone_alignment_{chat,base}.sh` |
| `concept_geometry/rsa/contrast_alignment_report_generator.py` | Contrast alignment HTML report | No | -- |
| `concept_geometry/rsa/standalone_alignment_report_generator.py` | Standalone alignment HTML report | No | -- |

### Comparisons (Cross-Model)

| Script | Description |
|--------|-------------|
| `comparisons/1_behavioral_summary_figures_generator.py` | 10 cross-model publication figures (scree plots, loading comparisons, entity scatter, mind space maps, heatmaps, RSA profiles, correlation summaries, RDM comparisons) |
| `comparisons/1a_behavioral_summary_report_generator.py` | Wraps figures into comprehensive HTML summary report |

---

## Execution

```bash
# From exp_4/code/

# === Core Pipeline ===

# Internals: Extract entity representations + RSA
sbatch internals/slurm/1_extract_entities_chat.sh
sbatch internals/slurm/1_extract_entities_base.sh
# Then generate reports (CPU):
python internals/1a_rsa_report_generator.py --model llama2_13b_chat
python internals/1a_rsa_report_generator.py --model llama2_13b_base

# Behavior: Gray et al. pairwise replication
sbatch behavior/slurm/1_pairwise_chat.sh
sbatch behavior/slurm/1_pairwise_base.sh
# Base-only: debiasing + individual ratings
python behavior/2_debiasing_reanalysis.py --model llama2_13b_base --both
sbatch behavior/slurm/3_individual_base.sh

# Gray-with-characters (30 AI/human characters on Gray capacities)
sbatch behavior/slurm/4_gray_chars_chat.sh
sbatch behavior/slurm/4_gray_chars_base.sh
# Then generate reports (CPU):
python behavior/4a_gray_chars_pca_report_generator.py --model llama2_13b_base
python behavior/4b_gray_chars_detailed_report_generator.py --model llama2_13b_base
python behavior/4c_gray_chars_rsa_report_generator.py --model llama2_13b_base

# Names-only variant
sbatch behavior/slurm/5_gray_names_only_chat.sh

# Cross-model comparison (CPU)
python comparisons/1_behavioral_summary_figures_generator.py
python comparisons/1a_behavioral_summary_report_generator.py

# === Concept Geometry Pipeline (Exp 3 Bridge) ===

# Phase A: Behavioral PCA (28 chars × ~27 concept dims)
sbatch concept_geometry/slurm/behavioral_pca_chat.sh
sbatch concept_geometry/slurm/behavioral_pca_base.sh

# Phase B: Activation RSA
sbatch concept_geometry/slurm/activation_rsa_chat.sh
sbatch concept_geometry/slurm/activation_rsa_base.sh

# Phase C: Concept alignment
sbatch concept_geometry/slurm/concept_rsa_chat.sh
sbatch concept_geometry/slurm/concept_rsa_base.sh
sbatch concept_geometry/slurm/contrast_alignment_chat.sh
sbatch concept_geometry/slurm/contrast_alignment_base.sh
sbatch concept_geometry/slurm/standalone_alignment_chat.sh
sbatch concept_geometry/slurm/standalone_alignment_base.sh

# Report generators (CPU, run after GPU phases complete):
python concept_geometry/pca/behavioral_pca_report_generator.py --model llama2_13b_base
python concept_geometry/pca/activation_pca_report_generator.py --model llama2_13b_base
python concept_geometry/rsa/activation_rsa_report_generator.py --model llama2_13b_base
python concept_geometry/rsa/contrast_alignment_report_generator.py --model llama2_13b_base
python concept_geometry/rsa/standalone_alignment_report_generator.py --model llama2_13b_base
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
