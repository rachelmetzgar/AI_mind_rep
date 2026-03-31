# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Qwen3-8B

**Run:** 2026-03-28 15:04:35

---

## What is being tested

Does Qwen3-8B's **explicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

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
- **13 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2808
- Successfully parsed: 2808 / 2808 (100.0%)

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 637 | 22.7% |
| 2 | 971 | 34.6% |
| 3 | 725 | 25.8% |
| 4 | 110 | 3.9% |
| 5 | 365 | 13.0% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1404
- Perfectly consistent (sum = 6): 553 (39.4%)
- Mean |sum - 6|: 1.22

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.45 * | 74.7% | 74.7% |
| PC2 | 2.58 * | 14.3% | 89.0% |
| PC3 | 0.62 | 3.5% | 92.5% |
| PC4 | 0.44 | 2.5% | 95.0% |
| PC5 | 0.33 | 1.9% | 96.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | -0.414 | -0.860 |
| fear | E | +0.451 | -0.780 |
| pain | E | +0.366 | -0.762 |
| pleasure | E | +0.899 | -0.348 |
| rage | E | +0.260 | -0.770 |
| desire | E | +0.942 | -0.168 |
| personality | E | +0.858 | -0.416 |
| consciousness | E | +0.965 | -0.063 |
| pride | E | +0.953 | -0.200 |
| embarrassment | E | +0.672 | -0.657 |
| joy | E | +0.870 | -0.267 |
| self_control | A | +0.969 | -0.045 |
| morality | A | +0.967 | -0.205 |
| memory | A | +0.953 | -0.189 |
| emotion_recognition | A | +0.916 | -0.285 |
| planning | A | +0.966 | -0.094 |
| communication | A | +0.940 | -0.067 |
| thought | A | +0.981 | -0.064 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.920 | 0.359 | 0.06 | 0.07 |
| frog | 0.296 | 0.533 | 0.25 | 0.14 |
| robot | 0.686 | 0.585 | 0.13 | 0.22 |
| fetus | 0.000 | 0.707 | 0.17 | 0.08 |
| pvs_patient | 0.208 | 0.532 | 0.17 | 0.10 |
| god | 1.000 | 1.000 | 0.20 | 0.80 |
| dog | 0.362 | 0.426 | 0.55 | 0.35 |
| chimpanzee | 0.001 | 0.273 | 0.63 | 0.48 |
| baby | 0.082 | 0.525 | 0.71 | 0.17 |
| girl | 0.545 | 0.000 | 0.84 | 0.62 |
| adult_woman | 0.890 | 0.099 | 0.93 | 0.91 |
| adult_man | 0.727 | 0.431 | 0.91 | 0.95 |
| you_self | 0.611 | 0.515 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.036 (p=0.9077) | rho=+0.407 (p=0.1680) |
| Factor 2 | rho=-0.498 (p=0.0833) | rho=-0.291 (p=0.3344) |

