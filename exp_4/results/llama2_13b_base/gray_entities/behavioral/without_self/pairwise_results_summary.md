# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## LLaMA-2-13B (Base)

**Run:** 2026-03-28 13:17:50

---

## What is being tested

Does LLaMA-2-13B (Base)'s **implicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

Unlike the chat model, the base model has no RLHF safety training, so it does not refuse to make comparisons. Instead of generating text responses, we extract the probability distribution over rating tokens (1-5) from the next-token logits.

## Procedure

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation, regression factor scores rescaled to 0-1

### Differs from original

- **Base model**: pretrained LLaMA-2-13B (Base) (no chat/RLHF)
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

- P(top rating) >= 0.5: 58.6%
- P(top rating) >= 0.7: 8.1%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.537
- Mean expected rating: 2.24

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 2376 | 100.0% |
| 2 | 0 | 0.0% |
| 3 | 0 | 0.0% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1188
- Mean |E[R_AB] + E[R_BA] - 6|: 1.510
- Argmax perfectly consistent: 0 (0.0%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.54 * | 86.4% | 86.4% |
| PC2 | 1.46 * | 8.1% | 94.5% |
| PC3 | 0.40 | 2.2% | 96.7% |
| PC4 | 0.20 | 1.1% | 97.8% |
| PC5 | 0.16 | 0.9% | 98.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.907 | +0.342 |
| fear | E | +0.755 | +0.620 |
| pain | E | +0.761 | +0.612 |
| pleasure | E | +0.669 | +0.700 |
| rage | E | +0.928 | +0.293 |
| desire | E | +0.664 | +0.719 |
| personality | E | +0.060 | +0.963 |
| consciousness | E | +0.454 | +0.865 |
| pride | E | +0.880 | +0.383 |
| embarrassment | E | +0.890 | +0.373 |
| joy | E | +0.939 | +0.303 |
| self_control | A | +0.669 | +0.699 |
| morality | A | +0.458 | +0.823 |
| memory | A | +0.683 | +0.720 |
| emotion_recognition | A | +0.713 | +0.673 |
| planning | A | +0.481 | +0.837 |
| communication | A | +0.587 | +0.773 |
| thought | A | +0.682 | +0.704 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.308 | 0.644 | 0.06 | 0.07 |
| frog | 0.395 | 0.140 | 0.25 | 0.14 |
| robot | 0.362 | 0.162 | 0.13 | 0.22 |
| fetus | 1.000 | 0.482 | 0.17 | 0.08 |
| pvs_patient | 0.395 | 0.359 | 0.17 | 0.10 |
| god | 0.384 | 0.000 | 0.20 | 0.80 |
| dog | 0.163 | 0.273 | 0.55 | 0.35 |
| chimpanzee | 0.250 | 0.689 | 0.63 | 0.48 |
| baby | 0.147 | 0.625 | 0.71 | 0.17 |
| girl | 0.000 | 0.660 | 0.84 | 0.62 |
| adult_woman | 0.265 | 0.736 | 0.93 | 0.91 |
| adult_man | 0.625 | 1.000 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.378 (p=0.2253) | rho=-0.210 (p=0.5128) |
| Factor 2 | rho=+0.613 (p=0.0341) | rho=+0.371 (p=0.2356) |

