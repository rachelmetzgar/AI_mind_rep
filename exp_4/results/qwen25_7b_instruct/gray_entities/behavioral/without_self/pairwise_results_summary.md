# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Qwen-2.5-7B-Instruct

**Run:** 2026-03-28 14:49:29

---

## What is being tested

Does Qwen-2.5-7B-Instruct's **explicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

This is a direct behavioral replication: the model answers the same pairwise comparison questions that ~2,400 human participants answered in the original study, using the exact character descriptions and mental capacity survey prompts from the supplementary materials.

## Procedure: What matches and what differs

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B ("This survey asks you to judge...")
- 5-point scale anchored by character names ("Much more [Name]" / "Slightly more [Name]" / "Both equally")
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation on capacity correlations across characters, regression-method factor scores rescaled to 0-1

### Differs from original

- **No photos**: original included character images; model gets text descriptions only
- **One deterministic participant**: original averaged over ~100+ respondents per capacity survey; we have one model with greedy decoding
- **Position-bias control**: original counterbalanced left/right across participants; we present each pair in BOTH orders and average to eliminate position bias
- **No survey selection**: original participants chose a capacity to rate; model rates all 18
- **12 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2376
- Successfully parsed: 2376 / 2376 (100.0%)

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1109 | 46.7% |
| 2 | 134 | 5.6% |
| 3 | 52 | 2.2% |
| 4 | 50 | 2.1% |
| 5 | 1031 | 43.4% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1188
- Perfectly consistent (sum = 6): 489 (41.2%)
- Mean |sum - 6|: 1.99

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.98 * | 77.7% | 77.7% |
| PC2 | 1.57 * | 8.7% | 86.4% |
| PC3 | 0.81 | 4.5% | 90.9% |
| PC4 | 0.50 | 2.8% | 93.6% |
| PC5 | 0.38 | 2.1% | 95.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.878 | +0.383 |
| fear | E | +0.660 | +0.608 |
| pain | E | +0.756 | +0.559 |
| pleasure | E | +0.709 | +0.563 |
| rage | E | +0.878 | +0.066 |
| desire | E | +0.497 | +0.818 |
| personality | E | +0.104 | +0.949 |
| consciousness | E | +0.475 | +0.852 |
| pride | E | +0.799 | +0.463 |
| embarrassment | E | +0.876 | +0.395 |
| joy | E | +0.494 | +0.805 |
| self_control | A | +0.681 | +0.390 |
| morality | A | +0.389 | +0.823 |
| memory | A | +0.753 | +0.577 |
| emotion_recognition | A | +0.856 | +0.478 |
| planning | A | +0.513 | +0.788 |
| communication | A | +0.372 | +0.865 |
| thought | A | +0.666 | +0.614 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.518 | 0.465 | 0.06 | 0.07 |
| frog | 0.467 | 0.292 | 0.25 | 0.14 |
| robot | 0.349 | 0.006 | 0.13 | 0.22 |
| fetus | 0.858 | 0.000 | 0.17 | 0.08 |
| pvs_patient | 1.000 | 0.066 | 0.17 | 0.10 |
| god | 0.000 | 0.313 | 0.20 | 0.80 |
| dog | 0.770 | 0.554 | 0.55 | 0.35 |
| chimpanzee | 0.761 | 0.523 | 0.63 | 0.48 |
| baby | 0.773 | 0.571 | 0.71 | 0.17 |
| girl | 0.667 | 0.876 | 0.84 | 0.62 |
| adult_woman | 0.589 | 1.000 | 0.93 | 0.91 |
| adult_man | 0.561 | 0.849 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.130 (p=0.6881) | rho=-0.315 (p=0.3191) |
| Factor 2 | rho=+0.862 (p=0.0003) | rho=+0.671 (p=0.0168) |

