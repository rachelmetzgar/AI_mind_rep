# Experiment 4: Mind Perception Geometry — Base Model

**Author:** Rachel C. Metzgar, Princeton University
**Model:** LLaMA-2-13B (base, no chat fine-tuning)

---

## Why base model?

The chat model (`llama_exp_4-13B-chat/`) failed this task: RLHF safety training caused
~50% refusal rates on ethically sensitive entities (dead woman 91%, PVS patient 82%,
God 77%). The base model has no safety training, so it doesn't refuse comparisons.

Instead of generating text responses and parsing for a digit, we extract the probability
distribution over tokens "1"-"5" from the next-token logits in a single forward pass.
This gives continuous expected ratings with no refusals.

---

## Method: Behavioral replication of Gray et al. (2007)

Replicating Gray, Gray, & Wegner (2007) "Dimensions of Mind Perception" (Science, 315, 619).

**Original study:** ~2,400 humans rated 13 entities via pairwise comparisons on 18 mental
capacities. PCA with varimax rotation revealed 2 factors: **Experience** (feeling) and
**Agency** (doing), explaining 97% of variance.

**Our replication:**
- Same 13 entities with verbatim character descriptions from Gray et al. Appendix A
- Same 18 mental capacity survey prompts from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity, both orders (counterbalancing)
- PCA with varimax rotation, regression factor scores rescaled to 0-1
- Spearman correlation with human Experience/Agency scores

---

## Scripts

### `2_behavioral_replication.py` — Pairwise comparisons (Gray et al. method)

12 entities x C(12,2)=66 pairs x 2 orders x 18 capacities = 2,376 comparisons.
Each comparison: single forward pass, extract P("1")...P("5"), compute E[R].

**Results (with_self, 13 entities):**
- 2-factor structure: eigenvalues 15.82 + 1.17 (human: 15.85 + 1.46)
- Factor 2 vs human Experience: **rho=0.72, p=.006**
- Factor 2 vs human Agency: rho=0.50, p=.086
- Factor 1 doesn't map to either human dimension
- 0% argmax order consistency (strong position bias toward entity A)

**Interpretation:** Model recovers the *shape* of mind perception (dominant first factor)
and partially aligns with Experience, but doesn't cleanly separate Agency as a second
dimension. Position bias is severe but algebraically cancelled by counterbalanced scoring.

### `2b_reanalysis_debiased.py` — Debiasing reanalysis

Tested two approaches on existing pairwise data (no rerun):
1. **Analytical debiasing:** R_debiased = (R_AB + (6 - R_BA)) / 2. Identical results
   to original — counterbalancing already cancels constant position bias.
2. **Log-odds:** log(P(A>B) / P(B>A)). Slightly worse — lost graded information.

### `2c_individual_ratings.py` — Individual Likert ratings (non-pairwise)

Alternative approach: rate each entity individually on each capacity (1-5 Likert).
Only 216 forward passes. Avoids pairwise position bias entirely. Deviates from
Gray et al. methodology but may produce cleaner signal.

---

## Key technical details

- **Tokenization:** Prompt must end with `"Rating: "` (trailing space). Without it, the
  model predicts a space token and bare digit logits are noise. This caused 0% order
  consistency and mean rating of 1.46 in the first run.
- **Token IDs:** LLaMA tokenizes digits as two tokens (▁ + digit). The code uses
  `tokenizer.encode(digit)[-1]` to get the bare digit ID.
- **Model loading:** `model.half().to(DEVICE)` — fp16 on GPU (~26GB VRAM).
- **HF cache:** Must set `HF_HOME` / `HF_HUB_CACHE` explicitly in SLURM scripts
  pointing to labs filesystem. Set `HF_HUB_OFFLINE=1` on compute nodes.

---

## Directory structure

```
llama_exp_4-13B-base/
├── README.md
├── 2_behavioral_replication.py       # Pairwise comparisons
├── 2b_reanalysis_debiased.py         # Debiasing analysis
├── 2c_individual_ratings.py          # Individual Likert ratings
├── entities/
│   └── gray_entities.py              # 13 entities, descriptions, capacities, human scores
├── data/
│   ├── behavioral_replication/       # Pairwise results
│   │   ├── with_self/                # raw_responses.json, character_means.npz, pca_results.npz
│   │   └── without_self/
│   └── individual_ratings/           # Likert results
│       ├── with_self/
│       └── without_self/
├── results/
│   ├── behavioral_replication/       # results_summary.md per condition
│   └── individual_ratings/
├── logs/                             # SLURM output/error logs
├── slurm/
│   ├── 2_behavioral_replication.sh
│   └── 2c_individual_ratings.sh
└── analysis_explainer.html           # Visual walkthrough of the analysis pipeline
```

---

## Human factor scores (Gray et al. 2007)

Estimated from Figure 1 (0-1 scale). **VERIFY BEFORE PUBLICATION** — digitize
Figure 1 with WebPlotDigitizer or contact Kurt Gray at UNC Chapel Hill.

| Entity | Experience | Agency |
|--------|----------:|-------:|
| dead_woman | 0.06 | 0.07 |
| frog | 0.25 | 0.14 |
| robot | 0.13 | 0.22 |
| fetus | 0.17 | 0.08 |
| pvs_patient | 0.17 | 0.10 |
| god | 0.20 | 0.80 |
| dog | 0.55 | 0.35 |
| chimpanzee | 0.63 | 0.48 |
| baby | 0.71 | 0.17 |
| girl | 0.84 | 0.62 |
| adult_woman | 0.93 | 0.91 |
| adult_man | 0.91 | 0.95 |
| you_self | 0.97 | 1.00 |

---

## References

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception.
*Science*, 315(5812), 619. https://doi.org/10.1126/science.1134475
