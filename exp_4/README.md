# Experiment 4: Mind Perception Geometry

**Author:** Rachel C. Metzgar, Princeton University

## Reports

- [Results walkthrough](exp4_results_walkthrough.html) — Combined results across both models: factor structure, human correlations, entity placements, RSA
- [Base model results](llama_exp_4-13B-base/results_report.html) — Pairwise and individual rating analyses for the base model
- [Analysis explainer](llama_exp_4-13B-base/analysis_explainer.html) — Step-by-step walkthrough of the PCA / varimax / factor-score methodology

---

## Motivation

Experiments 1-3 treat partner identity as a binary (human vs. AI). Human folk psychology is far richer. Gray, Gray, & Wegner (2007, *Science*) showed that humans perceive minds along two orthogonal dimensions: **Experience** (the capacity to feel — hunger, fear, pain, pleasure, joy) and **Agency** (the capacity to plan and act — self-control, morality, memory, planning, thought). ~2,400 participants rated 13 diverse entities via pairwise comparisons on 18 mental capacities, and PCA with varimax rotation recovered this two-factor structure, explaining 97% of variance.

Exp 4 asks: **does LLaMA-2-13B have an implicit folk psychology of mind that mirrors this human structure?** If the model's representational geometry over diverse entities (baby, dog, robot, God, adults, etc.) resembles the human Experience/Agency space, it would suggest the model has internalized a continuous, multi-dimensional folk psychology — not just a binary human/AI switch.

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
├── README.md                          # This file
├── exp4_results_walkthrough.html      # Visual walkthrough of results across both models
├── llama_exp_4-13B-chat/              # Chat model (RLHF fine-tuned)
└── llama_exp_4-13B-base/              # Base model (pretrained, no RLHF)
```

---

## `llama_exp_4-13B-chat/` — Chat Model

Uses LLaMA-2-13B-Chat to both extract entity representations and run a behavioral replication.

### Approach
- **Phase 1** (`1_extract_entity_representations.py`): Present entity prompts to the chat model, extract last-token residual-stream activations across all 41 layers. Compute representational dissimilarity matrices (RDMs) via cosine distance and compare to the human RDM using RSA (representational similarity analysis) at every layer.
- **Phase 2** (`2_behavioral_replication.py`): Direct behavioral replication — the chat model generates text responses to pairwise comparison prompts (66 pairs x 2 orders x 18 capacities = 2,376 comparisons). Responses are parsed for a 1-5 rating. PCA with varimax rotation on entity-by-capacity means.

### Limitation
RLHF safety training caused high refusal rates on ethically sensitive entities (dead woman ~91%, PVS patient ~82%, God ~77%), making the behavioral replication unreliable. This motivated the base model approach.

### Contents
```
llama_exp_4-13B-chat/
├── README.md
├── 1_extract_entity_representations.py   # Phase 1: activation extraction + RSA
├── 2_behavioral_replication.py           # Phase 2: pairwise comparisons (text generation)
├── entities/
│   ├── gray_entities.py                  # 13 entities, descriptions, capacities, human scores
│   └── gray2007.txt                      # Full text of Gray et al. paper + supplementary materials
├── data/
│   ├── entity_activations/               # Activations + RDMs (with_self/, without_self/)
│   └── behavioral_replication/           # Raw responses, character means, PCA results
├── results/                              # Summary reports + figures (RSA layerwise, RDM comparison)
├── slurm/                                # SLURM job scripts
└── logs/                                 # SLURM output/error logs
```

---

## `llama_exp_4-13B-base/` — Base Model

Uses LLaMA-2-13B (pretrained, no chat fine-tuning) to avoid RLHF refusals. Instead of generating text, ratings are extracted from next-token logits in a single forward pass: the probability distribution over tokens "1"-"5" yields continuous expected ratings with no refusals.

### Approach
- **Pairwise comparisons** (`2_behavioral_replication.py`): Same design as the chat version (66 pairs x 2 orders x 18 capacities), but using logit-based rating extraction. PCA with varimax rotation, Spearman correlation with human scores.
- **Debiasing reanalysis** (`2b_reanalysis_debiased.py`): Tests two strategies (analytical debiasing, log-odds) on the pairwise data to address position bias.
- **Individual Likert ratings** (`2c_individual_ratings.py`): Alternative non-pairwise approach — each entity rated individually on each capacity (1-5 Likert), avoiding pairwise position bias entirely. Deviates from Gray et al. methodology but may produce cleaner signal.
- **Figure generation** (`generate_figures.py`): Publication-quality figures for both base and chat model results (scree plots, loading comparisons, entity scatter plots, mind space comparisons, heatmaps, RSA profiles).

### Key Technical Detail
Prompts must end with `"Rating: "` (trailing space) for correct digit prediction. Without it, the model predicts a space token and bare digit logits are noise.

### Contents
```
llama_exp_4-13B-base/
├── README.md
├── 2_behavioral_replication.py           # Pairwise comparisons (logit-based)
├── 2b_reanalysis_debiased.py             # Debiasing reanalysis (no GPU needed)
├── 2c_individual_ratings.py              # Individual Likert ratings (non-pairwise)
├── generate_figures.py                   # Publication figures for both models
├── entities/
│   └── gray_entities.py                  # Same entity definitions as chat version
├── data/
│   ├── behavioral_replication/           # Pairwise results (with_self/, without_self/)
│   └── individual_ratings/              # Likert results (with_self/, without_self/)
├── results/
│   ├── behavioral_replication/           # Summary reports + figures
│   └── individual_ratings/              # Summary reports + figures
├── results_report.html                   # Visual walkthrough of the analysis pipeline
├── analysis_explainer.html               # Detailed analysis explainer
├── slurm/                                # SLURM job scripts
└── logs/                                 # SLURM output/error logs
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
