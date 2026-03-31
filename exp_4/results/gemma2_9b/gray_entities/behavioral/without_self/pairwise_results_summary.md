# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Gemma-2-9B (Base)

**Run:** 2026-03-29 10:55:43

---

## What is being tested

Does Gemma-2-9B (Base)'s **implicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

Unlike the chat model, the base model has no RLHF safety training, so it does not refuse to make comparisons. Instead of generating text responses, we extract the probability distribution over rating tokens (1-5) from the next-token logits.

## Procedure

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation, regression factor scores rescaled to 0-1

### Differs from original

- **Base model**: pretrained Gemma-2-9B (Base) (no chat/RLHF)
- **Logit-based ratings**: probability distribution over tokens "1"-"5" instead of generated text
- **Expected rating**: continuous E[R] = sum(p_i * i) instead of discrete response
- **No photos**: text descriptions only
- **Completion format**: prompt ends with "Rating:" for natural completion
- **Position-bias control**: each pair in both orders
- **12 entities**

## Response statistics

- Total comparisons: 2376
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 38.4%
- P(top rating) >= 0.7: 7.8%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.461
- Mean expected rating: 2.29

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1964 | 82.7% |
| 2 | 0 | 0.0% |
| 3 | 412 | 17.3% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1188
- Mean |E[R_AB] + E[R_BA] - 6|: 1.418
- Argmax perfectly consistent: 130 (10.9%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.97 * | 88.7% | 88.7% |
| PC2 | 0.68 | 3.8% | 92.6% |
| PC3 | 0.48 | 2.7% | 95.2% |
| PC4 | 0.29 | 1.6% | 96.9% |
| PC5 | 0.17 | 0.9% | 97.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.286 | -0.930 |
| fear | E | +0.794 | -0.490 |
| pain | E | +0.752 | -0.594 |
| pleasure | E | +0.756 | -0.604 |
| rage | E | +0.675 | -0.671 |
| desire | E | +0.916 | -0.316 |
| personality | E | +0.581 | -0.768 |
| consciousness | E | +0.709 | -0.605 |
| pride | E | +0.824 | -0.454 |
| embarrassment | E | +0.662 | -0.700 |
| joy | E | +0.740 | -0.602 |
| self_control | A | +0.759 | -0.570 |
| morality | A | +0.880 | -0.431 |
| memory | A | +0.820 | -0.537 |
| emotion_recognition | A | +0.824 | -0.507 |
| planning | A | +0.872 | -0.463 |
| communication | A | +0.834 | -0.482 |
| thought | A | +0.892 | -0.408 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 1.000 | 1.000 | 0.06 | 0.07 |
| frog | 0.946 | 0.000 | 0.25 | 0.14 |
| robot | 0.517 | 0.358 | 0.13 | 0.22 |
| fetus | 0.000 | 0.515 | 0.17 | 0.08 |
| pvs_patient | 0.440 | 0.765 | 0.17 | 0.10 |
| god | 0.537 | 0.617 | 0.20 | 0.80 |
| dog | 0.725 | 0.702 | 0.55 | 0.35 |
| chimpanzee | 0.615 | 0.509 | 0.63 | 0.48 |
| baby | 0.568 | 0.675 | 0.71 | 0.17 |
| girl | 0.769 | 0.674 | 0.84 | 0.62 |
| adult_woman | 0.823 | 0.566 | 0.93 | 0.91 |
| adult_man | 0.715 | 0.366 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.305 (p=0.3355) | rho=+0.133 (p=0.6806) |
| Factor 2 | rho=-0.182 (p=0.5710) | rho=-0.315 (p=0.3191) |

