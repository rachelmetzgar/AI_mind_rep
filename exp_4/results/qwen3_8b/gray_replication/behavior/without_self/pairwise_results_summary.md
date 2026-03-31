# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Qwen3-8B

**Run:** 2026-03-28 14:54:51

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
- **12 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2376
- Successfully parsed: 2376 / 2376 (100.0%)

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 586 | 24.7% |
| 2 | 877 | 36.9% |
| 3 | 471 | 19.8% |
| 4 | 99 | 4.2% |
| 5 | 343 | 14.4% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1188
- Perfectly consistent (sum = 6): 431 (36.3%)
- Mean |sum - 6|: 1.29

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.41 * | 74.5% | 74.5% |
| PC2 | 2.71 * | 15.0% | 89.6% |
| PC3 | 0.62 | 3.4% | 93.0% |
| PC4 | 0.44 | 2.4% | 95.4% |
| PC5 | 0.29 | 1.6% | 97.0% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | -0.420 | -0.855 |
| fear | E | +0.417 | -0.795 |
| pain | E | +0.393 | -0.803 |
| pleasure | E | +0.877 | -0.394 |
| rage | E | +0.232 | -0.780 |
| desire | E | +0.936 | -0.178 |
| personality | E | +0.853 | -0.436 |
| consciousness | E | +0.960 | -0.080 |
| pride | E | +0.946 | -0.236 |
| embarrassment | E | +0.611 | -0.742 |
| joy | E | +0.833 | -0.325 |
| self_control | A | +0.964 | -0.053 |
| morality | A | +0.961 | -0.250 |
| memory | A | +0.958 | -0.203 |
| emotion_recognition | A | +0.914 | -0.303 |
| planning | A | +0.963 | -0.107 |
| communication | A | +0.936 | -0.082 |
| thought | A | +0.978 | -0.098 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.914 | 0.361 | 0.06 | 0.07 |
| frog | 0.306 | 0.548 | 0.25 | 0.14 |
| robot | 0.673 | 0.626 | 0.13 | 0.22 |
| fetus | 0.000 | 0.757 | 0.17 | 0.08 |
| pvs_patient | 0.230 | 0.581 | 0.17 | 0.10 |
| god | 1.000 | 1.000 | 0.20 | 0.80 |
| dog | 0.351 | 0.431 | 0.55 | 0.35 |
| chimpanzee | 0.041 | 0.262 | 0.63 | 0.48 |
| baby | 0.141 | 0.477 | 0.71 | 0.17 |
| girl | 0.518 | 0.000 | 0.84 | 0.62 |
| adult_woman | 0.872 | 0.099 | 0.93 | 0.91 |
| adult_man | 0.717 | 0.428 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.011 (p=0.9741) | rho=+0.420 (p=0.1745) |
| Factor 2 | rho=-0.609 (p=0.0354) | rho=-0.350 (p=0.2652) |

