# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Qwen-2.5-7B (Base)

**Run:** 2026-03-29 10:56:14

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
- **13 entities**

## Response statistics

- Total comparisons: 2808
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 0.1%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.340
- Mean expected rating: 2.61

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 1115 | 39.7% |
| 2 | 0 | 0.0% |
| 3 | 1691 | 60.2% |
| 4 | 0 | 0.0% |
| 5 | 2 | 0.1% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1404
- Mean |E[R_AB] + E[R_BA] - 6|: 0.775
- Argmax perfectly consistent: 681 (48.5%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 17.13 * | 95.1% | 95.1% |
| PC2 | 0.39 | 2.1% | 97.3% |
| PC3 | 0.16 | 0.9% | 98.2% |
| PC4 | 0.10 | 0.6% | 98.7% |
| PC5 | 0.07 | 0.4% | 99.1% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.860 | +0.485 |
| fear | E | +0.774 | +0.623 |
| pain | E | +0.716 | +0.688 |
| pleasure | E | +0.734 | +0.668 |
| rage | E | +0.711 | +0.681 |
| desire | E | +0.700 | +0.691 |
| personality | E | +0.420 | +0.890 |
| consciousness | E | +0.677 | +0.697 |
| pride | E | +0.828 | +0.533 |
| embarrassment | E | +0.798 | +0.585 |
| joy | E | +0.851 | +0.508 |
| self_control | A | +0.619 | +0.758 |
| morality | A | +0.597 | +0.783 |
| memory | A | +0.865 | +0.483 |
| emotion_recognition | A | +0.773 | +0.606 |
| planning | A | +0.690 | +0.706 |
| communication | A | +0.735 | +0.667 |
| thought | A | +0.737 | +0.648 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.451 | 0.228 | 0.06 | 0.07 |
| frog | 0.507 | 0.221 | 0.25 | 0.14 |
| robot | 1.000 | 0.294 | 0.13 | 0.22 |
| fetus | 0.000 | 0.000 | 0.17 | 0.08 |
| pvs_patient | 0.387 | 0.335 | 0.17 | 0.10 |
| god | 0.513 | 1.000 | 0.20 | 0.80 |
| dog | 0.911 | 0.333 | 0.55 | 0.35 |
| chimpanzee | 0.995 | 0.182 | 0.63 | 0.48 |
| baby | 0.415 | 0.588 | 0.71 | 0.17 |
| girl | 0.708 | 0.197 | 0.84 | 0.62 |
| adult_woman | 0.464 | 0.508 | 0.93 | 0.91 |
| adult_man | 0.394 | 0.235 | 0.91 | 0.95 |
| you_self | 0.753 | 0.163 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.113 (p=0.7137) | rho=+0.374 (p=0.2086) |
| Factor 2 | rho=-0.063 (p=0.8373) | rho=+0.071 (p=0.8166) |

