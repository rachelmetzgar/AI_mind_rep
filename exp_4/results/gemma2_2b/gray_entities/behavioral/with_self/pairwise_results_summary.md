# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Gemma-2-2B (Base)

**Run:** 2026-03-29 10:55:06

---

## What is being tested

Does Gemma-2-2B (Base)'s **implicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

Unlike the chat model, the base model has no RLHF safety training, so it does not refuse to make comparisons. Instead of generating text responses, we extract the probability distribution over rating tokens (1-5) from the next-token logits.

## Procedure

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation, regression factor scores rescaled to 0-1

### Differs from original

- **Base model**: pretrained Gemma-2-2B (Base) (no chat/RLHF)
- **Logit-based ratings**: probability distribution over tokens "1"-"5" instead of generated text
- **Expected rating**: continuous E[R] = sum(p_i * i) instead of discrete response
- **No photos**: text descriptions only
- **Completion format**: prompt ends with "Rating:" for natural completion
- **Position-bias control**: each pair in both orders
- **13 entities**

## Response statistics

- Total comparisons: 2808
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 0.1%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.321
- Mean expected rating: 2.72

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1049 | 37.4% |
| 2 | 0 | 0.0% |
| 3 | 1759 | 62.6% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1404
- Mean |E[R_AB] + E[R_BA] - 6|: 0.567
- Argmax perfectly consistent: 735 (52.4%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 17.11 * | 95.0% | 95.0% |
| PC2 | 0.42 | 2.3% | 97.4% |
| PC3 | 0.16 | 0.9% | 98.2% |
| PC4 | 0.13 | 0.7% | 98.9% |
| PC5 | 0.10 | 0.5% | 99.5% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.563 | -0.793 |
| fear | E | +0.753 | -0.642 |
| pain | E | +0.760 | -0.624 |
| pleasure | E | +0.696 | -0.689 |
| rage | E | +0.705 | -0.682 |
| desire | E | +0.729 | -0.667 |
| personality | E | +0.872 | -0.465 |
| consciousness | E | +0.719 | -0.686 |
| pride | E | +0.494 | -0.868 |
| embarrassment | E | +0.617 | -0.775 |
| joy | E | +0.510 | -0.856 |
| self_control | A | +0.766 | -0.618 |
| morality | A | +0.750 | -0.651 |
| memory | A | +0.591 | -0.791 |
| emotion_recognition | A | +0.760 | -0.620 |
| planning | A | +0.837 | -0.524 |
| communication | A | +0.830 | -0.541 |
| thought | A | +0.607 | -0.762 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.095 | 0.000 | 0.06 | 0.07 |
| frog | 0.165 | 0.191 | 0.25 | 0.14 |
| robot | 0.637 | 0.695 | 0.13 | 0.22 |
| fetus | 1.000 | 0.503 | 0.17 | 0.08 |
| pvs_patient | 0.000 | 0.938 | 0.17 | 0.10 |
| god | 0.233 | 0.041 | 0.20 | 0.80 |
| dog | 0.485 | 0.912 | 0.55 | 0.35 |
| chimpanzee | 0.569 | 0.877 | 0.63 | 0.48 |
| baby | 0.232 | 0.895 | 0.71 | 0.17 |
| girl | 0.289 | 0.412 | 0.84 | 0.62 |
| adult_woman | 0.342 | 1.000 | 0.93 | 0.91 |
| adult_man | 0.132 | 0.839 | 0.91 | 0.95 |
| you_self | 0.697 | 0.097 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.182 (p=0.5527) | rho=+0.258 (p=0.3943) |
| Factor 2 | rho=+0.259 (p=0.3936) | rho=+0.066 (p=0.8305) |

