# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Qwen-2.5-7B (Base)

**Run:** 2026-03-29 10:54:43

---

## What is being tested

Does Qwen-2.5-7B (Base)'s **implicit folk psychology** of mind perception match the human folk psychology described by Gray, Gray, & Wegner (2007, Science)?

Unlike the chat model, the base model has no RLHF safety training, so it does not refuse to make comparisons. Instead of generating text responses, we extract the probability distribution over rating tokens (1-5) from the next-token logits.

## Procedure

### Matches original

- Character descriptions: verbatim from Appendix A
- Capacity survey prompts: verbatim from Appendix B
- 5-point scale anchored by character names
- All pairwise comparisons per capacity
- Analysis: PCA with varimax rotation, regression factor scores rescaled to 0-1

### Differs from original

- **Base model**: pretrained Qwen-2.5-7B (Base) (no chat/RLHF)
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
- Mean max P: 0.346
- Mean expected rating: 2.58

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 977 | 41.1% |
| 2 | 0 | 0.0% |
| 3 | 1399 | 58.9% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1188
- Mean |E[R_AB] + E[R_BA] - 6|: 0.836
- Argmax perfectly consistent: 560 (47.1%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 17.18 * | 95.4% | 95.4% |
| PC2 | 0.39 | 2.2% | 97.6% |
| PC3 | 0.15 | 0.8% | 98.4% |
| PC4 | 0.10 | 0.6% | 99.0% |
| PC5 | 0.07 | 0.4% | 99.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.841 | +0.512 |
| fear | E | +0.751 | +0.649 |
| pain | E | +0.691 | +0.714 |
| pleasure | E | +0.716 | +0.687 |
| rage | E | +0.665 | +0.741 |
| desire | E | +0.699 | +0.696 |
| personality | E | +0.430 | +0.882 |
| consciousness | E | +0.697 | +0.687 |
| pride | E | +0.818 | +0.551 |
| embarrassment | E | +0.782 | +0.608 |
| joy | E | +0.846 | +0.517 |
| self_control | A | +0.607 | +0.771 |
| morality | A | +0.559 | +0.815 |
| memory | A | +0.866 | +0.482 |
| emotion_recognition | A | +0.776 | +0.607 |
| planning | A | +0.662 | +0.735 |
| communication | A | +0.713 | +0.692 |
| thought | A | +0.748 | +0.634 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.456 | 0.190 | 0.06 | 0.07 |
| frog | 0.459 | 0.234 | 0.25 | 0.14 |
| robot | 1.000 | 0.314 | 0.13 | 0.22 |
| fetus | 0.000 | 0.000 | 0.17 | 0.08 |
| pvs_patient | 0.300 | 0.382 | 0.17 | 0.10 |
| god | 0.434 | 1.000 | 0.20 | 0.80 |
| dog | 0.865 | 0.357 | 0.55 | 0.35 |
| chimpanzee | 0.922 | 0.261 | 0.63 | 0.48 |
| baby | 0.326 | 0.633 | 0.71 | 0.17 |
| girl | 0.665 | 0.246 | 0.84 | 0.62 |
| adult_woman | 0.398 | 0.542 | 0.93 | 0.91 |
| adult_man | 0.342 | 0.313 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.060 (p=0.8542) | rho=+0.203 (p=0.5273) |
| Factor 2 | rho=+0.301 (p=0.3414) | rho=+0.469 (p=0.1245) |

