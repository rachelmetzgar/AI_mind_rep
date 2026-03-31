# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Qwen-2.5-7B-Instruct

**Run:** 2026-03-28 14:52:25

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
- **13 entities**: dropped "You" (no self-referential analog for an LLM)

## Response statistics

- Total comparisons: 2808
- Successfully parsed: 2808 / 2808 (100.0%)

### Rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1169 | 41.6% |
| 2 | 172 | 6.1% |
| 3 | 117 | 4.2% |
| 4 | 78 | 2.8% |
| 5 | 1272 | 45.3% |

### Order consistency

For each pair presented in both orders (A-B and B-A), a consistent model should give opposite ratings (R_AB + R_BA = 6). Deviation from this indicates position bias.

- Pairs with both orders parsed: 1404
- Perfectly consistent (sum = 6): 563 (40.1%)
- Mean |sum - 6|: 2.00

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.84 * | 76.9% | 76.9% |
| PC2 | 1.74 * | 9.7% | 86.5% |
| PC3 | 0.83 | 4.6% | 91.2% |
| PC4 | 0.43 | 2.4% | 93.6% |
| PC5 | 0.40 | 2.2% | 95.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.870 | +0.413 |
| fear | E | +0.715 | +0.572 |
| pain | E | +0.762 | +0.549 |
| pleasure | E | +0.681 | +0.585 |
| rage | E | +0.907 | +0.083 |
| desire | E | +0.531 | +0.787 |
| personality | E | +0.065 | +0.951 |
| consciousness | E | +0.418 | +0.872 |
| pride | E | +0.823 | +0.454 |
| embarrassment | E | +0.921 | +0.277 |
| joy | E | +0.513 | +0.773 |
| self_control | A | +0.641 | +0.450 |
| morality | A | +0.411 | +0.794 |
| memory | A | +0.822 | +0.487 |
| emotion_recognition | A | +0.891 | +0.399 |
| planning | A | +0.519 | +0.796 |
| communication | A | +0.360 | +0.870 |
| thought | A | +0.615 | +0.663 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.504 | 0.478 | 0.06 | 0.07 |
| frog | 0.448 | 0.307 | 0.25 | 0.14 |
| robot | 0.341 | 0.000 | 0.13 | 0.22 |
| fetus | 0.793 | 0.030 | 0.17 | 0.08 |
| pvs_patient | 1.000 | 0.117 | 0.17 | 0.10 |
| god | 0.000 | 0.326 | 0.20 | 0.80 |
| dog | 0.751 | 0.606 | 0.55 | 0.35 |
| chimpanzee | 0.730 | 0.564 | 0.63 | 0.48 |
| baby | 0.751 | 0.648 | 0.71 | 0.17 |
| girl | 0.655 | 0.861 | 0.84 | 0.62 |
| adult_woman | 0.565 | 1.000 | 0.93 | 0.91 |
| adult_man | 0.534 | 0.871 | 0.91 | 0.95 |
| you_self | 0.347 | 0.681 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.008 (p=0.9787) | rho=-0.407 (p=0.1680) |
| Factor 2 | rho=+0.872 (p=0.0001) | rho=+0.698 (p=0.0080) |

