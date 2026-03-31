# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## LLaMA-3-8B (Base)

**Run:** 2026-03-28 13:02:53

---

## What is being tested

Does LLaMA-3-8B (Base)'s **implicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

Unlike the chat model, the base model has no RLHF safety training, so it does not refuse to make comparisons. Instead of generating text responses, we extract the probability distribution over rating tokens (1-5) from the next-token logits.

## Procedure

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation, regression factor scores rescaled to 0-1

### Differs from original

- **Base model**: pretrained LLaMA-3-8B (Base) (no chat/RLHF)
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

- P(top rating) >= 0.5: 12.7%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.410
- Mean expected rating: 2.54

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 2569 | 91.5% |
| 2 | 0 | 0.0% |
| 3 | 230 | 8.2% |
| 4 | 0 | 0.0% |
| 5 | 9 | 0.3% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1404
- Mean |E[R_AB] + E[R_BA] - 6|: 0.916
- Argmax perfectly consistent: 59 (4.2%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 17.05 * | 94.7% | 94.7% |
| PC2 | 0.45 | 2.5% | 97.2% |
| PC3 | 0.18 | 1.0% | 98.2% |
| PC4 | 0.16 | 0.9% | 99.1% |
| PC5 | 0.05 | 0.3% | 99.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.487 | -0.870 |
| fear | E | +0.739 | -0.660 |
| pain | E | +0.779 | -0.613 |
| pleasure | E | +0.761 | -0.630 |
| rage | E | +0.677 | -0.719 |
| desire | E | +0.809 | -0.559 |
| personality | E | +0.778 | -0.543 |
| consciousness | E | +0.819 | -0.571 |
| pride | E | +0.526 | -0.847 |
| embarrassment | E | +0.578 | -0.812 |
| joy | E | +0.536 | -0.841 |
| self_control | A | +0.828 | -0.495 |
| morality | A | +0.804 | -0.564 |
| memory | A | +0.731 | -0.646 |
| emotion_recognition | A | +0.818 | -0.559 |
| planning | A | +0.772 | -0.605 |
| communication | A | +0.823 | -0.546 |
| thought | A | +0.706 | -0.691 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.611 | 1.000 | 0.06 | 0.07 |
| frog | 0.416 | 0.345 | 0.25 | 0.14 |
| robot | 0.382 | 0.000 | 0.13 | 0.22 |
| fetus | 0.418 | 0.968 | 0.17 | 0.08 |
| pvs_patient | 0.591 | 0.573 | 0.17 | 0.10 |
| god | 0.473 | 0.201 | 0.20 | 0.80 |
| dog | 0.886 | 0.317 | 0.55 | 0.35 |
| chimpanzee | 0.636 | 0.429 | 0.63 | 0.48 |
| baby | 0.408 | 0.544 | 0.71 | 0.17 |
| girl | 0.731 | 0.434 | 0.84 | 0.62 |
| adult_woman | 0.513 | 0.777 | 0.93 | 0.91 |
| adult_man | 0.000 | 0.175 | 0.91 | 0.95 |
| you_self | 1.000 | 0.104 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.223 (p=0.4643) | rho=+0.165 (p=0.5905) |
| Factor 2 | rho=-0.253 (p=0.4041) | rho=-0.599 (p=0.0306) |

