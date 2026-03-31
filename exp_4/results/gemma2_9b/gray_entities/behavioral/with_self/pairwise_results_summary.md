# Experiment 4: Pairwise Behavioral Replication of Gray et al. (2007)
## Gemma-2-9B (Base)

**Run:** 2026-03-29 10:58:22

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
- **13 entities**

## Response statistics

- Total comparisons: 2808
- All comparisons yield ratings (logit-based, no refusals)

### Rating probability concentration

How confident is the model? Distribution of max P(rating):

- P(top rating) >= 0.5: 40.5%
- P(top rating) >= 0.7: 9.8%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.471
- Mean expected rating: 2.26

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 2349 | 83.7% |
| 2 | 0 | 0.0% |
| 3 | 459 | 16.3% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Order consistency

For each pair in both orders, expected ratings should be complementary: E[R_AB] + E[R_BA] should equal 6.

- Pairs with both orders: 1404
- Mean |E[R_AB] + E[R_BA] - 6|: 1.471
- Argmax perfectly consistent: 144 (10.3%)

## PCA Results

### Eigenvalues and explained variance

Gray et al. found: Experience eigenvalue = 15.85 (88%), Agency eigenvalue = 1.46 (8%), total = 97%.

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.36 * | 85.3% | 85.3% |
| PC2 | 1.15 * | 6.4% | 91.7% |
| PC3 | 0.61 | 3.4% | 95.1% |
| PC4 | 0.26 | 1.4% | 96.5% |
| PC5 | 0.17 | 0.9% | 97.5% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

Compare to Gray et al. Table S1. In the original, Experience capacities loaded .67-.97 on Factor 1 and Agency capacities loaded .73-.97 on Factor 2.

| Capacity | Human factor | F1 loading | F2 loading |
|----------|:------------:|-----------:|-----------:|
| hunger | E | +0.272 | -0.837 |
| fear | E | +0.628 | -0.697 |
| pain | E | +0.904 | -0.329 |
| pleasure | E | +0.914 | -0.314 |
| rage | E | +0.454 | -0.853 |
| desire | E | +0.772 | -0.531 |
| personality | E | +0.709 | -0.596 |
| consciousness | E | +0.844 | -0.423 |
| pride | E | +0.491 | -0.820 |
| embarrassment | E | +0.395 | -0.909 |
| joy | E | +0.523 | -0.826 |
| self_control | A | +0.813 | -0.493 |
| morality | A | +0.769 | -0.570 |
| memory | A | +0.753 | -0.627 |
| emotion_recognition | A | +0.683 | -0.695 |
| planning | A | +0.866 | -0.470 |
| communication | A | +0.865 | -0.449 |
| thought | A | +0.789 | -0.552 |

### Entity positions (factor scores, 0-1 scale)

Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 to match the original figure.

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.625 | 0.772 | 0.06 | 0.07 |
| frog | 1.000 | 0.147 | 0.25 | 0.14 |
| robot | 0.573 | 0.705 | 0.13 | 0.22 |
| fetus | 0.000 | 0.666 | 0.17 | 0.08 |
| pvs_patient | 0.483 | 1.000 | 0.17 | 0.10 |
| god | 0.484 | 0.751 | 0.20 | 0.80 |
| dog | 0.575 | 0.683 | 0.55 | 0.35 |
| chimpanzee | 0.642 | 0.730 | 0.63 | 0.48 |
| baby | 0.606 | 0.881 | 0.71 | 0.17 |
| girl | 0.491 | 0.498 | 0.84 | 0.62 |
| adult_woman | 0.743 | 0.556 | 0.93 | 0.91 |
| adult_man | 0.690 | 0.451 | 0.91 | 0.95 |
| you_self | 0.125 | 0.000 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

Spearman correlations between model factor scores (0-1) and human factor scores. We check all factor-to-dimension correlations since the factor ordering and sign may differ.

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.209 (p=0.4930) | rho=+0.082 (p=0.7890) |
| Factor 2 | rho=-0.586 (p=0.0353) | rho=-0.538 (p=0.0576) |

