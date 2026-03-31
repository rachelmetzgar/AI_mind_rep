# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Gemma-2-2B (Base)

**Run:** 2026-03-29 10:53:07

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
- **12 entities**

## Response statistics

- Total comparisons: 2376
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 0.1%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.323
- Mean expected rating: 2.72

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 850 | 35.8% |
| 2 | 0 | 0.0% |
| 3 | 1526 | 64.2% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1188
- Mean |E[R_AB] + E[R_BA] - 6|: 0.551
- Argmax perfectly consistent: 636 (53.5%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 17.05 * | 94.7% | 94.7% |
| PC2 | 0.50 | 2.8% | 97.5% |
| PC3 | 0.18 | 1.0% | 98.5% |
| PC4 | 0.13 | 0.7% | 99.2% |
| PC5 | 0.06 | 0.3% | 99.6% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.554 | -0.809 |
| fear | E | +0.738 | -0.656 |
| pain | E | +0.759 | -0.624 |
| pleasure | E | +0.702 | -0.686 |
| rage | E | +0.675 | -0.708 |
| desire | E | +0.723 | -0.675 |
| personality | E | +0.901 | -0.412 |
| consciousness | E | +0.727 | -0.680 |
| pride | E | +0.471 | -0.880 |
| embarrassment | E | +0.588 | -0.799 |
| joy | E | +0.491 | -0.865 |
| self_control | A | +0.776 | -0.609 |
| morality | A | +0.757 | -0.639 |
| memory | A | +0.563 | -0.817 |
| emotion_recognition | A | +0.762 | -0.624 |
| planning | A | +0.820 | -0.553 |
| communication | A | +0.829 | -0.539 |
| thought | A | +0.592 | -0.770 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.193 | 0.060 | 0.06 | 0.07 |
| frog | 0.210 | 0.168 | 0.25 | 0.14 |
| robot | 0.656 | 0.677 | 0.13 | 0.22 |
| fetus | 1.000 | 0.492 | 0.17 | 0.08 |
| pvs_patient | 0.000 | 0.946 | 0.17 | 0.10 |
| god | 0.345 | 0.000 | 0.20 | 0.80 |
| dog | 0.503 | 0.934 | 0.55 | 0.35 |
| chimpanzee | 0.596 | 0.883 | 0.63 | 0.48 |
| baby | 0.271 | 0.928 | 0.71 | 0.17 |
| girl | 0.364 | 0.454 | 0.84 | 0.62 |
| adult_woman | 0.310 | 1.000 | 0.93 | 0.91 |
| adult_man | 0.167 | 0.873 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.109 (p=0.7369) | rho=+0.035 (p=0.9141) |
| Factor 2 | rho=+0.434 (p=0.1583) | rho=+0.189 (p=0.5567) |

